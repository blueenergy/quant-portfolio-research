#!/usr/bin/env python3
"""ML 信号持有期匹配：IC 衰减曲线 + 年化净超额曲线（阶段 A）。

预注册（跑数前已冻结，判据见该页，**结果不回改预注册**）：
``projects/trading/research/ml-holding-period-decay-preregistration-2026-11-19.md``

复用 ``analysis/research/score_ic_by_year_regime.py`` 的逐日 IC 与面板构建，
新增三件事：

1. 把 ``quant_ml.model_predictions`` 的模型分并进面板（``analysis/core/model_scores.py``）；
2. 用 **top-N** 而不是分位做成本块（对齐 live 的 ``top_n``，分位在 ~1000 只的池子上
   等于 top-20% ≈ 200 只，与 live 的 top-20 不是一回事）；
3. 给**年化**口径与相对参考 horizon 的比值 —— 决策量是年化净超额，不是 IC。

口径备忘：

* IC 是同日截面的 Spearman（Pearson-on-ranks），只在 score 与 forward return
  **都非空**的子集上算；h>1 时相邻截面的前瞻收益重叠，朴素 t 虚高，
  所以每档都另给「按 h 步长取样的非重叠 t」。
* ``gross_excess(h)`` = top-N 均值 − 全池均值（h 日前瞻收益，按日等权）。
* ``turnover(h)`` = 相邻 h 步调仓日之间 top-N 名单的单边 churn（**名单口径，
  不是权重口径**，是实际权重的代理）。
* ``net_annualized(h) = (gross_excess(h) − turnover(h) × cost) × 252/h``。
* 模型分跨日不可比（原始回归输出），但本脚本全程只用**同日截面**的排序，
  因此不受影响；凡涉及跨日分数阈值的读法都不成立。

Example (inside the quant-scorer image, workdir /app):
    python analysis/research/model_score_holding_period_decay.py \
        --model-id research-csi1000-clean-raw5-oos2025 \
        --start-date 20250102 --end-date 20261113 --universe csi1000 \
        --horizons 1 2 3 5 10 15 20 30 40 60 --top-n 20 \
        --output reports/ml_holding_period_decay_oos2025.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from analysis.core.trade_fee_model import TradeFeeModel
from analysis.research.score_ic_by_year_regime import (
    ic_by_date,
    prepare_panel,
    summarize_with_nonoverlap,
)

ROUND_TRIP_COST_RATE = TradeFeeModel().period_cost_rate(1.0)
TRADING_DAYS_PER_YEAR = 252

MODEL_COLUMN = "model_score"
REFERENCE_HORIZON = 20


# ---------------------------------------------------------------------------
# top-N 成本块
# ---------------------------------------------------------------------------


def _select_topn(
    sub: pd.DataFrame, score_col: str, top_n: int, construction: str
) -> pd.DataFrame:
    """按构造方式选 top-N。

    ``top_n``：直接取分数最高的 N 只。

    ``industry_neutral``：**照抄** ``PortfolioBacktester._select_industry_neutral``
    的口径 —— 按候选池里各行业的占比给每行业分配配额，各行业内取分数最高的若干只，
    名额不满再从全局排名里补。live 用的就是这一档；不加行业约束的 top-N
    会把赌注压在少数行业上，与 live 不可比。
    """
    if construction == "top_n":
        return sub.head(int(top_n))
    if construction != "industry_neutral":
        raise ValueError(f"Unsupported construction: {construction}")
    if "industry" not in sub.columns:
        return sub.head(int(top_n))

    counts = sub.groupby("industry", sort=False).size()
    total = float(counts.sum())
    if total <= 0:
        return sub.head(int(top_n))
    raw = {industry: int(top_n) * count / total for industry, count in counts.items()}
    quotas = {industry: int(value) for industry, value in raw.items()}
    remaining = int(top_n) - sum(quotas.values())
    for industry, _ in sorted(
        raw.items(), key=lambda item: item[1] - int(item[1]), reverse=True
    )[:remaining]:
        quotas[industry] = quotas.get(industry, 0) + 1

    picked: List[pd.DataFrame] = []
    for industry, quota in quotas.items():
        if quota <= 0:
            continue
        group = sub[sub["industry"] == industry]
        if not group.empty:
            picked.append(group.head(int(quota)))
    if not picked:
        return sub.head(int(top_n))
    selected = pd.concat(picked).drop_duplicates(subset=["symbol"], keep="first")
    if len(selected) < int(top_n):
        seen = set(selected["symbol"])
        extra = sub[~sub["symbol"].isin(seen)].head(int(top_n) - len(selected))
        selected = pd.concat([selected, extra])
    return selected.sort_values(score_col, ascending=False).head(int(top_n))


def topn_cost_block(
    dataset: pd.DataFrame,
    score_col: str,
    horizon: int,
    *,
    top_n: int,
    cost_rate: float = ROUND_TRIP_COST_RATE,
    min_names: int = 20,
    construction: str = "top_n",
    n_boot: int = 0,
    block: int = 0,
    seed: int = 20261119,
) -> Optional[Dict[str, Any]]:
    """Top-N 的毛超额、名单换手、扣成本净超额与年化。

    与 ``score_ic_by_year_regime.quantile_cost_block`` 的区别：那里是**分位**
    （固定比例），这里是 **top-N**（绝对只数），才与 live 的 ``top_n`` 同口径。

    换手按「每 horizon 个交易日调仓一次」：相邻两个 h 步调仓日之间，
    top-N 名单的变化比例（单边）。

    额外产出**逐期序列**（每期的毛超额、每对调仓日的换手），供 bootstrap 用：
    两者都在同一张「期」网格上，才能联合重采样。
    """
    ret_col = f"forward_return_{horizon}d"
    if score_col not in dataset.columns or ret_col not in dataset.columns:
        return None
    columns = ["score_date", "symbol", score_col, ret_col]
    if "industry" in dataset.columns:
        columns.insert(2, "industry")
    sub = dataset[columns].dropna()
    if sub.empty or sub["score_date"].nunique() < 2:
        return None

    # 稳定排序 + 组内序号：分数相同的票按 symbol 字典序打破并列，保证可复现。
    sub = sub.sort_values(
        ["score_date", score_col, "symbol"], ascending=[True, False, True]
    ).copy()
    sub["_rank"] = sub.groupby("score_date", sort=False).cumcount() + 1

    top = pd.concat(
        [
            _select_topn(group, score_col, int(top_n), construction)
            for _, group in sub.groupby("score_date", sort=False)
        ]
    )
    if top.empty:
        return None

    per_date = pd.DataFrame(
        {
            "top_mean": top.groupby("score_date")[ret_col].mean(),
            "universe_mean": sub.groupby("score_date")[ret_col].mean(),
        }
    ).dropna()
    if per_date.empty:
        return None

    top_sets: Dict[str, set] = {
        str(date): set(group["symbol"])
        for date, group in top.groupby("score_date")
    }
    rebalance_dates = sorted(top_sets.keys())[:: max(1, int(horizon))]
    turnovers: List[float] = []
    for previous, current in zip(rebalance_dates, rebalance_dates[1:]):
        current_set = top_sets.get(current) or set()
        if not current_set:
            continue
        previous_set = top_sets.get(previous) or set()
        turnovers.append(1.0 - len(current_set & previous_set) / len(current_set))

    mean_turnover = float(np.mean(turnovers)) if turnovers else float("nan")
    gross = float((per_date["top_mean"] - per_date["universe_mean"]).mean())
    annualization = TRADING_DAYS_PER_YEAR / max(1, int(horizon))
    net = None if math.isnan(mean_turnover) else gross - mean_turnover * cost_rate
    result = {
        "horizon": int(horizon),
        "observations": int(len(per_date)),
        "rebalance_periods": int(len(turnovers)),
        "top_excess_gross": gross,
        "mean_turnover": None if math.isnan(mean_turnover) else mean_turnover,
        "cost_rate_per_unit_turnover": float(cost_rate),
        "top_excess_net_per_period": net,
        "gross_annualized": gross * annualization,
        "net_annualized": None if net is None else net * annualization,
        "annualization_factor": annualization,
        "by_year": _by_year_breakdown(per_date, rebalance_dates, turnovers),
    }

    if n_boot and block and turnovers:
        result["bootstrap"] = bootstrap_net_annualized(
            per_date,
            rebalance_dates,
            top_sets,
            horizon,
            cost_rate=cost_rate,
            n_boot=int(n_boot),
            block=int(block),
            seed=int(seed),
        )
    return result


def _by_year_breakdown(
    per_date: pd.DataFrame,
    rebalance_dates: Sequence[str],
    turnovers: Sequence[float],
) -> Dict[str, Any]:
    """逐年拆毛超额与换手。

    **必须有这一步**：2026-11-19 的实测里，全窗口看「Q9 优于 Q10」t=5.52，
    但拆到年就是 2025 +0.998%/期、**2026 −0.261%/期（符号翻转）**。
    不分年就会把单年现象报成结论。
    """
    excess = (per_date["top_mean"] - per_date["universe_mean"])
    out: Dict[str, Any] = {}
    for year, chunk in excess.groupby(excess.index.astype(str).str[:4]):
        turn_year = [
            value
            for date, value in zip(rebalance_dates[1:], turnovers)
            if str(date).startswith(str(year))
        ]
        out[str(year)] = {
            "observations": int(len(chunk)),
            "gross_per_period": float(chunk.mean()),
            "turnover_per_period": float(np.mean(turn_year)) if turn_year else None,
        }
    return out


def _moving_block_mean_ci(
    values: Sequence[float], *, n_boot: int, block: int, seed: int, alpha: float = 0.05
) -> Optional[Dict[str, float]]:
    """移动块 bootstrap：均值的百分位区间。

    块长覆盖序列的自相关长度（h 日前瞻收益相邻期重叠），
    否则朴素重采样会把标准误压到远小于真实值。
    """
    arr = np.asarray([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    n = len(arr)
    if n < 3:
        return None
    block = max(1, min(int(block), n))
    n_blocks = int(math.ceil(n / block))
    last_start = n - block
    if last_start < 0:
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, last_start + 1, size=n_blocks)
        idx = np.concatenate(
            [np.arange(s, s + block) for s in starts]
        )[:n]
        means[b] = float(arr[idx].mean())
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(arr.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(n),
        "block": int(block),
        "n_boot": int(n_boot),
    }


def bootstrap_net_annualized(
    per_date: pd.DataFrame,
    rebalance_dates: Sequence[str],
    top_sets: Dict[str, set],
    horizon: int,
    *,
    cost_rate: float,
    n_boot: int,
    block: int,
    seed: int,
    alpha: float = 0.05,
) -> Optional[Dict[str, Any]]:
    """逐期 bootstrap：年化净超额的区间。

    毛超额按「日 → 期」聚合到调仓期网格，与换手同网格，
    这样两者可以联合重采样（而不是各自独立，��独立会低估组合的不确定性）。
    """
    h = max(1, int(horizon))
    dates = list(per_date.index)
    excess = (per_date["top_mean"] - per_date["universe_mean"]).to_dict()

    # 把日超额摊到「调仓期」：第 i 期覆盖 rebalance_dates[i] 起的 h 个交易日。
    period_excess: List[float] = []
    for i, start in enumerate(rebalance_dates):
        chunk = [excess.get(d) for d in dates if start <= d < _shift_date(dates, start, h)]
        chunk = [v for v in chunk if v is not None]
        if chunk:
            period_excess.append(float(np.mean(chunk)))
    period_turnover: List[float] = []
    for previous, current in zip(rebalance_dates, rebalance_dates[1:]):
        current_set = top_sets.get(current) or set()
        if not current_set:
            continue
        previous_set = top_sets.get(previous) or set()
        period_turnover.append(1.0 - len(current_set & previous_set) / len(current_set))

    n = min(len(period_excess), len(period_turnover))
    if n < 3:
        return None
    gross_ci = _moving_block_mean_ci(
        period_excess[:n], n_boot=n_boot, block=block, seed=seed, alpha=alpha
    )
    turn_ci = _moving_block_mean_ci(
        period_turnover[:n], n_boot=n_boot, block=block, seed=seed + 1, alpha=alpha
    )
    if gross_ci is None or turn_ci is None:
        return None
    annualization = TRADING_DAYS_PER_YEAR / h
    return {
        "periods": int(n),
        "gross_per_period": gross_ci,
        "turnover_per_period": turn_ci,
        # 保守包络：毛超额取下界、换手取上界得净下界；反之得上界。
        "net_annualized_ci_low": (
            gross_ci["ci_low"] - turn_ci["ci_high"] * cost_rate
        ) * annualization,
        "net_annualized_ci_high": (
            gross_ci["ci_high"] - turn_ci["ci_low"] * cost_rate
        ) * annualization,
        "alpha": alpha,
    }


def _shift_date(dates: Sequence[str], start: str, h: int) -> str:
    """返回 dates 中第 ``start`` 之后 h 个交易日的日期（用于切期）。"""
    ordered = list(dates)
    try:
        idx = ordered.index(start)
    except ValueError:
        return ""
    return ordered[min(len(ordered) - 1, idx + h)]



# ---------------------------------------------------------------------------
# 分析
# ---------------------------------------------------------------------------


def analyze_column(
    dataset: pd.DataFrame,
    score_col: str,
    horizons: Sequence[int],
    *,
    top_n: int,
    cost_rate: float,
    min_names: int,
    reference_horizon: int,
    construction: str = "top_n",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "column": score_col,
        "reference_horizon": int(reference_horizon),
        "horizons": {},
    }
    for horizon in horizons:
        h = int(horizon)
        ret_col = f"forward_return_{h}d"
        entry: Dict[str, Any] = {"horizon": h}

        per_date = ic_by_date(dataset, score_col, ret_col, min_names)
        if not per_date.empty:
            entry["ic"] = summarize_with_nonoverlap(per_date, h)
        else:
            entry["ic"] = None

        entry["cost"] = topn_cost_block(
            dataset,
            score_col,
            h,
            top_n=top_n,
            cost_rate=cost_rate,
            min_names=min_names,
            construction=construction,
        )
        out["horizons"][str(h)] = entry

    # 相对参考 horizon 的比值：共同分母 ⇒ 比绝对值稳得多。
    ref = out["horizons"].get(str(int(reference_horizon)), {})
    ref_ic = (ref.get("ic") or {}).get("ic_mean")
    ref_net = ((ref.get("cost") or {}).get("net_annualized"))
    for entry in out["horizons"].values():
        ic_mean = ((entry.get("ic") or {}).get("ic_mean"))
        entry["ic_ratio_vs_reference"] = (
            None
            if (ic_mean is None or not ref_ic)
            else float(ic_mean) / float(ref_ic)
        )
        net = ((entry.get("cost") or {}).get("net_annualized"))
        entry["net_annualized_ratio"] = (
            None
            if (net is None or not ref_net)
            else float(net) / float(ref_net)
        )
    return out


def pick_best(analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """argmax 年化净超额（决策量），只看有值的档位。"""
    best: Optional[Dict[str, Any]] = None
    for entry in analysis.get("horizons", {}).values():
        net = ((entry.get("cost") or {}).get("net_annualized"))
        if net is None:
            continue
        if best is None or float(net) > float(best["net_annualized"]):
            best = {
                "horizon": int(entry["horizon"]),
                "net_annualized": float(net),
                "gross_annualized": (entry.get("cost") or {}).get("gross_annualized"),
                "mean_turnover": (entry.get("cost") or {}).get("mean_turnover"),
                "ic_mean": (entry.get("ic") or {}).get("ic_mean"),
                "nonoverlap_t_stat": (entry.get("ic") or {}).get("nonoverlap_t_stat"),
            }
    return best


def model_score_fill_stats(
    dataset: pd.DataFrame, model_id: str, score_col: str = MODEL_COLUMN
) -> Dict[str, Any]:
    """模型分的覆盖与填充情况（用于判断换手口径有没有被陈旧分污染）。"""
    if score_col not in dataset.columns:
        return {"available": False}
    values = dataset[score_col]
    total = int(len(values))
    non_null = int(values.notna().sum())
    dates = dataset.loc[values.notna(), "score_date"].nunique()
    all_dates = dataset["score_date"].nunique()
    return {
        "available": True,
        "model_id": model_id,
        "rows_total": total,
        "rows_with_score": non_null,
        "rows_filled_ratio": None if not total else (total - non_null) / total,
        "score_dates": int(dates),
        "panel_dates": int(all_dates),
        "dates_covered_ratio": None if not all_dates else float(dates) / float(all_dates),
    }


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def print_table(analysis: Dict[str, Any], title: str) -> None:
    print(f"\n=== {title} ===")
    print(
        f"{'h':>4} {'obs':>5} {'IC':>8} {'no-t':>7} {'IC/IC20':>8} "
        f"{'gross_ann':>10} {'turn':>6} {'net_ann':>9} {'net/20':>7}"
    )
    for key in sorted(analysis.get("horizons", {}), key=lambda k: int(k)):
        entry = analysis["horizons"][key]
        ic = entry.get("ic") or {}
        cost = entry.get("cost") or {}
        print(
            f"{int(key):>4} {int(ic.get('observations', 0)):>5} "
            f"{_fmt(ic.get('ic_mean')):>8} {_fmt(ic.get('nonoverlap_t_stat'), 2):>7} "
            f"{_fmt(entry.get('ic_ratio_vs_reference'), 3):>8} "
            f"{_fmt(cost.get('gross_annualized'), 4):>10} "
            f"{_fmt(cost.get('mean_turnover'), 3):>6} "
            f"{_fmt(cost.get('net_annualized'), 4):>9} "
            f"{_fmt(entry.get('net_annualized_ratio'), 3):>7}"
        )
    # 分年毛超额：不分年会把单年现象读成结论（见 _by_year_breakdown 的 docstring）。
    print(f"  {'h':>4} {'year':>6} {'obs':>5} {'gross/period':>13} {'turnover':>9}")
    for key in sorted(analysis.get("horizons", {}), key=lambda k: int(k)):
        by_year = (analysis["horizons"][key].get("cost") or {}).get("by_year") or {}
        for year in sorted(by_year):
            row = by_year[year]
            print(
                f"  {int(key):>4} {year:>6} {int(row.get('observations', 0)):>5} "
                f"{_fmt(row.get('gross_per_period'), 5):>13} "
                f"{_fmt(row.get('turnover_per_period'), 3):>9}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IC decay + annualized net excess vs holding period for the ML score."
    )
    parser.add_argument("--model-id", default="research-csi1000-clean-raw5-oos2025")
    parser.add_argument("--start-date", default="20250102")
    parser.add_argument("--end-date", default="20261113")
    parser.add_argument("--universe", default="csi1000")
    parser.add_argument(
        "--horizons", nargs="+", type=int, default=[1, 2, 3, 5, 10, 15, 20, 30, 40, 60]
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--construction",
        choices=["top_n", "industry_neutral"],
        default="industry_neutral",
        help="组合构造；live 用的是 industry_neutral，默认与 live 对齐",
    )
    parser.add_argument("--min-names", type=int, default=20)
    parser.add_argument("--reference-horizon", type=int, default=REFERENCE_HORIZON)
    parser.add_argument("--ffill-limit", type=int, default=1)
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["model_score", "growth_cycle_g60_c40_score"],
        help="被评估的列；model_score 由本脚本从 quant_ml 并入",
    )
    parser.add_argument("--no-panel-cache", dest="use_panel_cache", action="store_false")
    parser.set_defaults(use_panel_cache=True)
    # score_ic_by_year_regime.prepare_panel 需要的两个开关（控制派生 blend 列）。
    parser.add_argument(
        "--growth-cycle-weights",
        dest="growth_cycle_weights",
        action="store_true",
        default=True,
    )
    parser.add_argument("--score-spec", nargs="+")
    parser.add_argument("--output", help="JSON 输出路径")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    horizons = tuple(int(h) for h in args.horizons)

    db_manager, dataset, _ = prepare_panel(args)
    print(
        f"[panel] rows={len(dataset)} dates={dataset['score_date'].nunique()}",
        flush=True,
    )

    from analysis.core.model_scores import merge_model_score_into_dataset

    merged = merge_model_score_into_dataset(
        dataset,
        args.model_id,
        start_date=args.start_date,
        end_date=args.end_date,
        column=MODEL_COLUMN,
        ffill_limit=int(args.ffill_limit),
        enabled=True,
    )
    fill_stats = model_score_fill_stats(merged, args.model_id)
    print(f"[model] {json.dumps(fill_stats, ensure_ascii=False)}", flush=True)
    dataset = merged

    report: Dict[str, Any] = {
        "meta": {
            "model_id": args.model_id,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "universe": args.universe,
            "horizons": list(horizons),
            "top_n": int(args.top_n),
            "construction": str(args.construction),
            "min_names": int(args.min_names),
            "ffill_limit": int(args.ffill_limit),
            "reference_horizon": int(args.reference_horizon),
            "round_trip_cost_rate": ROUND_TRIP_COST_RATE,
            "panel_rows": int(len(dataset)),
            "panel_dates": int(dataset["score_date"].nunique()),
        },
        "model_score_fill": fill_stats,
        "columns": {},
    }

    for column in args.columns:
        if column not in dataset.columns:
            print(f"[warn] column missing from panel, skipped: {column}", flush=True)
            continue
        analysis = analyze_column(
            dataset,
            column,
            horizons,
            top_n=int(args.top_n),
            cost_rate=ROUND_TRIP_COST_RATE,
            min_names=int(args.min_names),
            reference_horizon=int(args.reference_horizon),
            construction=str(args.construction),
        )
        analysis["best_by_net_annualized"] = pick_best(analysis)
        report["columns"][column] = analysis
        print_table(analysis, column)

    for column, analysis in report["columns"].items():
        best = analysis.get("best_by_net_annualized")
        if best:
            print(
                f"[best] {column}: h={best['horizon']} "
                f"net_annualized={_fmt(best['net_annualized'])} "
                f"(reference h={args.reference_horizon})",
                flush=True,
            )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[out] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
