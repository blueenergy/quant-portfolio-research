#!/usr/bin/env python3
"""逐年 / 分市况 Rank IC、ICIR 与 horizon decay —— stock_scores 的统计功效主战场。

组合净值曲线一年 ≈ 一个独立数据点，确认年化 SR=0.5>0 需要 ~25 年日频
（见 llm-wiki ``notes/statistical-evidence-of-edge``）。横截面 IC 是唯一能在
可等日历时间内攒够样本量的口径，所以「长期正期望」最严的存在形式是：

    多年 mean IC 显著 > 0  +  逐年/分市况稳定  +  扣成本净 IC  +  decay 合理

本脚本就出这份报告。它不重跑组合回测，只消费 ``stock_scores`` 面板。

口径要点（每条都对应一个踩过的坑）：

* 生产打分行是**派生列** ``growth_cycle_g60_c40_score``，必须用
  ``add_weighted_score_columns`` 现算。漏这步会让结论完全反过来
  （置换检验第一版就是这么错的）。
* h>1 时逐日 IC 的持有期重叠，t 值虚高。所以每格同时给「按 h 步长取样的
  非重叠 t」，两个数并排看。
* 成本用生产同款 ``TradeFeeModel``（默认 7bp / 单位换手），不另立假设。
* 9 列 × 7 年 × 5 horizon 是多重比较，**全量出、不筛选**，好看不好看都留在
  JSON 里。

Example (inside the quant-scorer image, workdir /app):
    python analysis/research/score_ic_by_year_regime.py \
        --start-date 20200201 --end-date 20260930 --universe csi1000 \
        --horizons 1 5 10 20 60 \
        --output reports/score_ic_csi1000_2020_2026.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.core.trade_fee_model import TradeFeeModel
from analysis.factor.ic_metrics import summarize_ic

# 生产配置：growth:cycle = 60:40，两个可交易 horizon 档在生产里是 {10, 20, 30}，
# 这里多扫 1 / 5 / 60 是为了看 decay 形状。
PRODUCTION_GROWTH_CYCLE = "60:40"
PRODUCTION_COLUMN = "growth_cycle_g60_c40_score"

DEFAULT_COLUMNS: Tuple[str, ...] = (
    PRODUCTION_COLUMN,
    "growth_score",
    "cycle_score",
    "value_score",
    "fundamental_score",
    "money_flow_score",
    "technical_score",
    "composite_score",
)

# 生产 cost model 的默认值；写死在这里只是为了让报告能独立说明扣了多少。
ROUND_TRIP_COST_RATE = TradeFeeModel().period_cost_rate(1.0)


# ---------------------------------------------------------------------------
# IC 计算（向量化：每个 (列, horizon) 组合一次 groupby 求和，不做逐日 apply）
# ---------------------------------------------------------------------------


def ic_by_date(
    frame: pd.DataFrame,
    score_col: str,
    ret_col: str,
    min_names: int,
) -> pd.DataFrame:
    """逐日 Spearman IC（Pearson-on-ranks），向量化。

    只在 score 与 forward return **都非空**的子集上算秩与方差 —— 用全样本算
    分母会把尾部那些还没有 forward return 的行算进去，而尾部恰好就是 2026 年，
    正是最需要看准的一段。
    """
    sub = frame[["score_date", score_col, ret_col]].dropna()
    if sub.empty:
        return pd.DataFrame(columns=["score_date", "rank_ic", "count"])
    dates = sub["score_date"]
    grouped = sub.groupby("score_date", sort=False)
    score_rank = grouped[score_col].rank()
    ret_rank = grouped[ret_col].rank()
    demeaned_score = score_rank - score_rank.groupby(dates, sort=False).transform("mean")
    demeaned_ret = ret_rank - ret_rank.groupby(dates, sort=False).transform("mean")

    agg = pd.DataFrame(
        {
            "num": (demeaned_score * demeaned_ret).groupby(dates, sort=True).sum(),
            "var_score": (demeaned_score**2).groupby(dates, sort=True).sum(),
            "var_ret": (demeaned_ret**2).groupby(dates, sort=True).sum(),
            "count": demeaned_score.groupby(dates, sort=True).size(),
        }
    )
    denom = np.sqrt(agg["var_score"] * agg["var_ret"])
    agg["rank_ic"] = agg["num"] / denom.where(denom > 0)
    agg = agg[(agg["count"] >= min_names) & agg["rank_ic"].notna()]
    return agg.reset_index()[["score_date", "rank_ic", "count"]]


def summarize_with_nonoverlap(per_date: pd.DataFrame, horizon: int) -> Dict[str, Any]:
    """全样本汇总 + 非重叠子样本汇总。

    持有期 h>1 时相邻 score_date 的 forward return 重叠，逐日 IC 因此自相关，
    朴素 t 值虚高。按 h 步长取样近似消除重叠，给出一个保守下限。
    """
    base = summarize_ic(per_date)
    base["horizon"] = int(horizon)
    ordered = per_date.sort_values("score_date")
    stride = max(1, int(horizon))
    sub = ordered.iloc[::stride]
    nonoverlap = summarize_ic(sub)
    base["nonoverlap_observations"] = nonoverlap["observations"]
    base["nonoverlap_ic_mean"] = nonoverlap["ic_mean"]
    base["nonoverlap_t_stat"] = nonoverlap["t_stat"]
    return base


def group_summaries(
    per_date: pd.DataFrame,
    horizon: int,
    *,
    by_year: bool = True,
    by_regime: bool = True,
    regime_by_date: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"overall": summarize_with_nonoverlap(per_date, horizon)}
    if by_year and not per_date.empty:
        df = per_date.copy()
        df["year"] = df["score_date"].astype(str).str[:4]
        out["by_year"] = [
            {"year": year, **summarize_with_nonoverlap(group, horizon)}
            for year, group in sorted(df.groupby("year"))
        ]
    if by_regime and regime_by_date is not None and not per_date.empty:
        df = per_date.copy()
        df["regime"] = df["score_date"].astype(str).map(regime_by_date).fillna("uncertain")
        out["by_regime"] = [
            {"regime": label, **summarize_with_nonoverlap(group, horizon)}
            for label, group in sorted(df.groupby("regime"))
        ]
    return out


# ---------------------------------------------------------------------------
# 分位组合：毛超额、换手、扣成本后超额
# ---------------------------------------------------------------------------


def quantile_cost_block(
    dataset: pd.DataFrame,
    score_col: str,
    horizon: int,
    *,
    quantiles: int = 5,
    cost_rate: float = ROUND_TRIP_COST_RATE,
) -> Optional[Dict[str, Any]]:
    """Top 分位的毛/净超额与换手。净额 = 毛超额 − 换手 × 单边成本率。

    换手按「每 horizon 个交易日调仓一次」的口径：t 与 t−h 两期 Top 分位的
    名单差异比例。成本率与组合回测同源（``TradeFeeModel``）。
    """
    ret_col = f"forward_return_{horizon}d"
    if score_col not in dataset.columns or ret_col not in dataset.columns:
        return None
    sub = dataset[["score_date", "symbol", score_col, ret_col]].dropna()
    if sub.empty or sub["score_date"].nunique() < 2:
        return None

    rank_pct = sub.groupby("score_date", sort=False)[score_col].rank(pct=True)
    top_cut = 1.0 - 1.0 / quantiles
    top = sub[rank_pct > top_cut]
    bottom = sub[rank_pct <= 1.0 / quantiles]

    per_date = pd.DataFrame(
        {
            "top_mean": top.groupby("score_date")[ret_col].mean(),
            "bottom_mean": bottom.groupby("score_date")[ret_col].mean(),
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
    spread = float((per_date["top_mean"] - per_date["bottom_mean"]).mean())
    return {
        "horizon": int(horizon),
        "observations": int(len(per_date)),
        "top_excess_gross": gross,
        "top_minus_bottom_gross": spread,
        "mean_turnover": None if math.isnan(mean_turnover) else mean_turnover,
        "cost_rate_per_unit_turnover": cost_rate,
        "top_excess_net": None if math.isnan(mean_turnover) else gross - mean_turnover * cost_rate,
    }


# ---------------------------------------------------------------------------
# 面板与市况
# ---------------------------------------------------------------------------


def load_regime_by_date(
    db_manager: Any,
    universe: str,
    start_date: str,
    score_dates: Sequence[str],
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """用系统自己那套 regime label（K_fast/K_slow EWMA + 确认天数）标注每个 score_date。"""
    from analysis.core.benchmark_prices import load_index_benchmark_prices
    from shared.regime_label import label_index_regime, padded_index_start
    from shared.universe_benchmark import default_index_benchmark_symbol

    symbol = default_index_benchmark_symbol(universe)
    rows = load_index_benchmark_prices(db_manager, symbol, padded_index_start(start_date))
    if not rows:
        return {date: "uncertain" for date in score_dates}, {"error": f"no benchmark prices for {symbol}"}
    labels = label_index_regime(rows)
    if labels.empty:
        return {date: "uncertain" for date in score_dates}, {"error": "empty regime labels"}

    frame = labels[["trade_date", "label"]].copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame = frame[frame["trade_date"].str.len() == 8].sort_values("trade_date")
    frame["_key"] = frame["trade_date"].astype("int64")
    wanted = pd.DataFrame({"score_date": sorted({str(d) for d in score_dates})})
    wanted["_key"] = wanted["score_date"].astype("int64")
    merged = pd.merge_asof(
        wanted.sort_values("_key"),
        frame[["_key", "label"]].sort_values("_key"),
        on="_key",
        direction="backward",
    )
    mapping = {
        str(row.score_date): str(row.label) if isinstance(row.label, str) else "uncertain"
        for row in merged.itertuples()
    }
    distribution: Dict[str, int] = {}
    for label in mapping.values():
        distribution[label] = distribution.get(label, 0) + 1
    return mapping, {
        "benchmark_symbol": symbol,
        "label_distribution_over_score_dates": distribution,
    }


def prepare_panel(args: argparse.Namespace):
    from analysis.core.score_backtester import BacktestConfig, ScoreBacktester
    from analysis.sweep.score_columns import add_weighted_score_columns, build_score_sweeps
    from analysis.sweep.sweep_engine import load_panel_cache, panel_cache_key, save_panel_cache
    from shared.data_access import open_research_data_access
    from shared.industry_universe import attach_sw_l1
    from shared.scores import latest_stock_score_date
    from analysis.sweep.sweep_engine import panel_cache_dir

    universe = [args.universe]
    horizons = tuple(int(h) for h in args.horizons)
    db_manager = open_research_data_access()
    # shared.scores 是 contract 层,research 层可直接调用。
    # tools.process_portfolio_research_jobs._latest_stock_score_date 只是它的
    # 一层包装,而 tools 属于 entry 层,跨过去会被 test_module_boundaries 拦下。
    stock_scores_max_date = latest_stock_score_date(
        getattr(db_manager, "stock_scores", None), args.end_date, universe
    )
    cache_key = panel_cache_key(
        universe_index=universe,
        start_date=args.start_date,
        end_date=args.end_date,
        horizons=horizons,
        stock_scores_max_date=stock_scores_max_date,
    )
    cached = load_panel_cache(cache_key) if args.use_panel_cache else None
    if cached is not None:
        dataset = cached[0]
        print(f"[panel] cache hit {cache_key} rows={len(dataset)}", flush=True)
        if len(dataset):
            # Recompute rather than trust a cached column: panels are cached after
            # the attach, so a stale cache would silently keep pre-PIT membership.
            dataset = attach_sw_l1(dataset, db_manager, as_of_column="score_date")
    else:
        print(
            f"[panel] cache miss {cache_key}, building {args.start_date}-{args.end_date} "
            f"universe={args.universe} horizons={horizons} from Mongo (minutes)",
            flush=True,
        )
        config = BacktestConfig(
            horizons=horizons,
            start_date=args.start_date,
            end_date=args.end_date,
            universe_index_codes=tuple(universe),
            trailing_stop_pct=None,
        )
        backtester = ScoreBacktester(db_manager, config)
        score_panel = backtester.prepare_score_panel()
        dataset = backtester.build_dataset_from_panel(score_panel, trailing_stop_pct=None)
        if args.use_panel_cache and not dataset.empty:
            save_panel_cache(
                cache_key,
                dataset,
                backtester.price_cache,
                score_panel=score_panel,
                cache_root=panel_cache_dir(),
            )
            print(f"[panel] cached as {cache_key}", flush=True)

    if dataset is None or dataset.empty:
        raise SystemExit("panel is empty; check --start-date/--end-date/--universe")

    sweeps = build_score_sweeps(
        PRODUCTION_COLUMN,
        [PRODUCTION_GROWTH_CYCLE] if args.growth_cycle_weights else None,
        args.score_spec,
    )
    dataset = add_weighted_score_columns(dataset, sweeps)
    return db_manager, dataset, horizons


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def build_report(
    dataset: pd.DataFrame,
    horizons: Sequence[int],
    columns: Sequence[str],
    regime_by_date: Dict[str, str],
    *,
    min_names: int,
    cost_columns: Sequence[str],
) -> Dict[str, Any]:
    frame = dataset.copy()
    frame["score_date"] = frame["score_date"].astype(str)

    report_columns: Dict[str, Any] = {}
    for column in columns:
        if column not in frame.columns:
            report_columns[column] = {"error": "column missing from panel"}
            continue
        per_horizon: Dict[str, Any] = {}
        for horizon in horizons:
            ret_col = f"forward_return_{horizon}d"
            if ret_col not in frame.columns:
                continue
            per_date = ic_by_date(frame, column, ret_col, min_names)
            if per_date.empty:
                continue
            block = group_summaries(
                per_date, horizon, regime_by_date=regime_by_date
            )
            per_horizon[f"h{horizon}"] = block
        report_columns[column] = per_horizon

    cost_block: Dict[str, Any] = {}
    for column in cost_columns:
        if column not in dataset.columns:
            continue
        rows = [
            row
            for horizon in horizons
            if (row := quantile_cost_block(dataset, column, horizon)) is not None
        ]
        cost_block[column] = {"quantiles": 5, "by_horizon": rows}

    return {
        "columns": report_columns,
        "top_quantile_cost": cost_block,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def print_report(report: Dict[str, Any], columns: Sequence[str], horizons: Sequence[int], focus: str) -> None:
    print("\n==== decay：全期 mean IC / ICIR / 非重叠 t ====")
    header = f"{'column':<32}" + "".join(f"{'h' + str(h):>22}" for h in horizons)
    print(header)
    for column in columns:
        blocks = report["columns"].get(column) or {}
        if not isinstance(blocks, dict) or "error" in blocks:
            continue
        cells = []
        for horizon in horizons:
            block = blocks.get(f"h{horizon}")
            if not block:
                cells.append(f"{'-':>22}")
                continue
            overall = block["overall"]
            cells.append(
                f"{_fmt(overall['ic_mean'], 4):>7}/{_fmt(overall['ic_ir'], 2):>5}"
                f"/{_fmt(overall['nonoverlap_t_stat'], 2):>6}"
            )
        print(f"{column:<32}" + "".join(f"{c:>22}" for c in cells))

    focus_blocks = report["columns"].get(focus) or {}
    if isinstance(focus_blocks, dict) and "error" not in focus_blocks:
        for horizon in horizons:
            block = focus_blocks.get(f"h{horizon}")
            if not block:
                continue
            print(f"\n==== {focus} @ h={horizon}：逐年 ====")
            print(f"{'year':<8}{'n':>6}{'IC':>10}{'ICIR':>9}{'t':>8}{'no-t':>8}{'pos%':>8}")
            for row in block.get("by_year", []):
                print(
                    f"{row['year']:<8}{row['observations']:>6}{_fmt(row['ic_mean'], 4):>10}"
                    f"{_fmt(row['ic_ir'], 2):>9}{_fmt(row['t_stat'], 2):>8}"
                    f"{_fmt(row['nonoverlap_t_stat'], 2):>8}{_fmt(row['positive_ratio'], 2):>8}"
                )
            print(f"---- {focus} @ h={horizon}：分市况 ----")
            print(f"{'regime':<12}{'n':>6}{'IC':>10}{'ICIR':>9}{'t':>8}{'no-t':>8}{'pos%':>8}")
            for row in block.get("by_regime", []):
                print(
                    f"{row['regime']:<12}{row['observations']:>6}{_fmt(row['ic_mean'], 4):>10}"
                    f"{_fmt(row['ic_ir'], 2):>9}{_fmt(row['t_stat'], 2):>8}"
                    f"{_fmt(row['nonoverlap_t_stat'], 2):>8}{_fmt(row['positive_ratio'], 2):>8}"
                )

    if report.get("top_quantile_cost"):
        print(f"\n==== Top 分位超额（扣成本，cost={ROUND_TRIP_COST_RATE:.4%}/单位换手）====")
        print(f"{'column':<32}{'h':>4}{'毛超额':>12}{'换手':>9}{'净超额':>12}{'Q5-Q1':>12}")
        for column, payload in report["top_quantile_cost"].items():
            for row in payload["by_horizon"]:
                print(
                    f"{column:<32}{row['horizon']:>4}"
                    f"{_fmt(row['top_excess_gross'], 4):>12}"
                    f"{_fmt(row['mean_turnover'], 3):>9}"
                    f"{_fmt(row['top_excess_net'], 4):>12}"
                    f"{_fmt(row['top_minus_bottom_gross'], 4):>12}"
                )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-year / per-regime Rank IC, ICIR and horizon decay for stock_scores."
    )
    parser.add_argument("--start-date", default="20200201")
    parser.add_argument("--end-date", default="20260930")
    parser.add_argument("--universe", default="csi1000")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10, 20, 60])
    parser.add_argument("--min-names", type=int, default=20)
    parser.add_argument(
        "--columns",
        nargs="+",
        default=list(DEFAULT_COLUMNS),
        help="评估列；默认生产 blend + 六个维度分 + composite_score",
    )
    parser.add_argument(
        "--cost-columns",
        nargs="+",
        default=[PRODUCTION_COLUMN],
        help="算分位超额/换手的列（默认只算生产 blend，其余列靠 IC 表看）",
    )
    parser.add_argument("--focus", default=PRODUCTION_COLUMN, help="终端详细表的列")
    parser.add_argument(
        "--growth-cycle-weights",
        dest="growth_cycle_weights",
        action="store_true",
        default=True,
        help="派生 growth_cycle_g60_c40_score（默认开；关掉的话生产列会是空的）",
    )
    parser.add_argument("--score-spec", nargs="+", help="替代 --columns 的加权口径，如 growth=0.6 cycle=0.4")
    parser.add_argument("--no-panel-cache", dest="use_panel_cache", action="store_false")
    parser.set_defaults(use_panel_cache=True)
    parser.add_argument("--output", help="JSON 输出路径")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    db_manager, dataset, horizons = prepare_panel(args)
    columns = [str(c) for c in args.columns]
    missing = [c for c in columns if c not in dataset.columns]
    if missing:
        print(f"[warn] columns missing from panel, skipped: {missing}", flush=True)

    score_dates = sorted({str(d) for d in dataset["score_date"].unique()})
    regime_by_date, regime_meta = load_regime_by_date(
        db_manager, args.universe, args.start_date, score_dates
    )
    print(f"[panel] rows={len(dataset)} dates={len(score_dates)} "
          f"range={score_dates[0]}..{score_dates[-1]} regime={regime_meta}", flush=True)

    report = {
        "config": {
            "universe": args.universe,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "horizons": list(horizons),
            "min_names_per_date": args.min_names,
            "columns": columns,
            "production_growth_cycle": PRODUCTION_GROWTH_CYCLE,
            "cost_rate_per_unit_turnover": ROUND_TRIP_COST_RATE,
            "rows": int(len(dataset)),
            "distinct_score_dates": len(score_dates),
            "score_date_range": [score_dates[0], score_dates[-1]],
        },
        "regime": regime_meta,
        "regime_by_score_date": regime_by_date,
    }
    report.update(
        build_report(
            dataset,
            horizons,
            columns,
            regime_by_date,
            min_names=args.min_names,
            cost_columns=[str(c) for c in args.cost_columns],
        )
    )

    print_report(report, columns, horizons, args.focus)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        print(f"\n[report] written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
