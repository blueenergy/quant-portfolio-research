#!/usr/bin/env python3
"""诊断两个因子（默认 growth × cycle）的协同来源：真·正交增量/交互 还是 起点运气。

只读、不写库：复用 ``ScoreBacktester.build_dataset`` 取 base/overlay 两个因子分
与前向收益，再按截面逐日计算四类指标并跨日聚合：

1. corr(rank base, rank overlay)  —— 两者打分相关性，越低，分散增益越成立。
2. 正交增量 IC                     —— overlay 对 base 做截面线性回归取残差后的 IC，
                                      验证 overlay 中独立于 base 的部分是否有正信息。
3. 条件 IC                         —— 仅在 base 头部/尾部子集内算 overlay 的 IC，对比全市场，
                                      验证 base×overlay 的交互（确认/防价值陷阱）效应。
4. 混合 IC 曲线                     —— 对 base:overlay 各权重直接算混合分的全截面 IC，
                                      看中间权重是否出现凸起。

可用 --base-factor / --overlay-factor 切换任意两个已落库的因子列；--overlay-reversal
对叠加因子取反号(100-x)以检验反转方向的协同。

Example:
    python analysis/growth_cycle_synergy_check.py \
        --start-date 20250101 --end-date 20260501 \
        --universe-index csi1000 --horizons 5 10 20 \
        --base-factor growth_score --overlay-factor technical_score \
        --out reports/synergy_growth_technical_csi1000.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def _agg(vals: Sequence[float]) -> Dict[str, Optional[float]]:
    vals = [float(v) for v in vals if v is not None and not pd.isna(v)]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "icir": None, "pos_ratio": None}
    m = mean(vals)
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    return {
        "n": len(vals),
        "mean": m,
        "std": sd,
        "icir": (m / sd) if sd else None,
        "pos_ratio": sum(1 for v in vals if v > 0) / len(vals),
    }


def _spearman(a: pd.Series, b: pd.Series) -> Optional[float]:
    # Ranks + Pearson is Spearman without scipy, which the worker image does
    # not install. The ranks have to come from the pairwise-complete rows,
    # which is the subset corr() would have used.
    pair = pd.concat([a, b], axis=1).dropna()
    if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return None
    return float(pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank()))


def parse_weights(tokens: Sequence[str]) -> List[tuple]:
    """支持 '60:40' 或单值 base 权重 '0.6'（overlay=1-base）。返回 [(bw, ow), ...]。"""
    out: List[tuple] = []
    for tok in tokens:
        if ":" in tok:
            g, c = tok.split(":", 1)
            gw, cw = float(g), float(c)
            total = gw + cw
            if total > 1.5:  # 形如 60:40 -> 归一
                gw, cw = gw / total, cw / total
        else:
            gw = float(tok)
            cw = 1.0 - gw
        out.append((round(gw, 4), round(cw, 4)))
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose two-factor synergy (orthogonal increment & interaction).")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--universe-index", nargs="+", default=["csi1000"])
    p.add_argument("--horizons", nargs="+", type=int, default=[5, 10, 20])
    p.add_argument("--base-factor", default="growth_score", help="主因子列名（默认 growth_score）")
    p.add_argument("--overlay-factor", default="cycle_score", help="叠加因子列名（默认 cycle_score）")
    p.add_argument(
        "--overlay-reversal",
        action="store_true",
        help="对叠加因子取反号(100-x)后再分析，用于检验反转方向的协同",
    )
    p.add_argument("--growth-quantile", type=float, default=0.7, help="主因子头部子集阈值分位（默认前 30%%）")
    p.add_argument(
        "--weights",
        nargs="+",
        default=["0:100", "20:80", "40:60", "50:50", "60:40", "80:20", "100:0"],
        help="混合 IC 曲线的 base:overlay 权重",
    )
    p.add_argument("--min-names", type=int, default=5)
    p.add_argument("--out", help="结果 JSON 输出路径")
    return p.parse_args()


def main() -> int:
    from analysis.core.price_cache import VolumePriceCacheProxy
    from analysis.core.score_backtester import BacktestConfig, ScoreBacktester
    from shared.data_access import open_research_data_access_for_cli
    from datetime import datetime, timedelta
    from utils.dates import normalize_date

    args = _parse_args()
    weights = parse_weights(args.weights)

    db_real = open_research_data_access_for_cli(enable_tushare=False)
    history_start = (
        datetime.strptime(normalize_date(args.start_date), "%Y%m%d") - timedelta(days=400)
    ).strftime("%Y%m%d")
    db = VolumePriceCacheProxy(db_real, history_start)

    g_col, c_col = args.base_factor, args.overlay_factor
    overlay_label = f"{c_col}{'(reversal)' if args.overlay_reversal else ''}"
    config = BacktestConfig(
        horizons=tuple(args.horizons),
        start_date=args.start_date,
        end_date=args.end_date,
        universe_index_codes=tuple(args.universe_index),
        score_columns=(g_col, c_col),
    )
    print(
        f"构建数据集: universe={','.join(args.universe_index)} range={args.start_date}~{args.end_date} "
        f"base={g_col} overlay={overlay_label} horizons={args.horizons} ...",
        flush=True,
    )
    dataset = ScoreBacktester(db, config).build_dataset()
    if dataset.empty:
        print("❌ 数据集为空")
        return 1

    for col in (g_col, c_col):
        if col not in dataset.columns:
            print(f"❌ 数据集缺少列 {col}（可选: growth_score/cycle_score/technical_score/money_flow_score/...）")
            return 1
        dataset[col] = pd.to_numeric(dataset[col], errors="coerce")
    if args.overlay_reversal:
        dataset[c_col] = 100.0 - dataset[c_col]

    # 指标 1：base/overlay 打分相关性（与 horizon 无关）。
    corr_vals: List[float] = []
    for _, grp in dataset.groupby("score_date"):
        v = grp[[g_col, c_col]].dropna()
        if len(v) >= args.min_names:
            r = _spearman(v[g_col], v[c_col])
            if r is not None:
                corr_vals.append(r)
    corr_summary = _agg(corr_vals)

    result: Dict[str, object] = {
        "config": {
            "universe": args.universe_index,
            "range": [args.start_date, args.end_date],
            "horizons": args.horizons,
            "base_factor": g_col,
            "overlay_factor": overlay_label,
            "growth_quantile": args.growth_quantile,
            "n_dates": int(dataset["score_date"].nunique()),
            "n_rows": int(len(dataset)),
        },
        "rank_corr_base_overlay": corr_summary,
        "by_horizon": {},
    }

    print(
        f"\nrank_corr({g_col}, {overlay_label}): mean={_f(corr_summary['mean'])} "
        f"std={_f(corr_summary['std'])} n={corr_summary['n']}",
        flush=True,
    )

    for h in args.horizons:
        ret_col = f"forward_return_{h}d"
        if ret_col not in dataset.columns:
            continue
        ic_base: List[float] = []
        ic_overlay: List[float] = []
        ic_resid: List[float] = []
        ic_cond_top: List[float] = []
        ic_cond_bottom: List[float] = []
        blend_ic: Dict[str, List[float]] = {f"{int(gw*100)}:{int(cw*100)}": [] for gw, cw in weights}

        for _, grp in dataset.groupby("score_date"):
            v = grp[[g_col, c_col, ret_col]].dropna()
            if len(v) < args.min_names or v[ret_col].nunique() < 2:
                continue
            g, c, r = v[g_col], v[c_col], v[ret_col]

            ig = _spearman(g, r)
            ic = _spearman(c, r)
            if ig is not None:
                ic_base.append(ig)
            if ic is not None:
                ic_overlay.append(ic)

            # 正交增量：overlay 对 base 截面线性回归取残差，再算残差 IC。
            if g.nunique() >= 2:
                a, b = np.polyfit(g.to_numpy(float), c.to_numpy(float), 1)
                resid = c - (a * g + b)
                ir = _spearman(resid, r)
                if ir is not None:
                    ic_resid.append(ir)

            # 条件 IC：base 头部 / 尾部子集内 overlay 的 IC。
            thr_hi = g.quantile(args.growth_quantile)
            thr_lo = g.quantile(1.0 - args.growth_quantile)
            top = v[v[g_col] >= thr_hi]
            bot = v[v[g_col] <= thr_lo]
            if len(top) >= args.min_names and top[ret_col].nunique() >= 2:
                ict = _spearman(top[c_col], top[ret_col])
                if ict is not None:
                    ic_cond_top.append(ict)
            if len(bot) >= args.min_names and bot[ret_col].nunique() >= 2:
                icb = _spearman(bot[c_col], bot[ret_col])
                if icb is not None:
                    ic_cond_bottom.append(icb)

            # 混合 IC 曲线（原始 0-100 加权，base:overlay，和 sweep 口径一致）。
            for gw, cw in weights:
                blend = gw * g + cw * c
                bic = _spearman(blend, r)
                if bic is not None:
                    blend_ic[f"{int(gw*100)}:{int(cw*100)}"].append(bic)

        h_result = {
            "ic_base": _agg(ic_base),
            "ic_overlay": _agg(ic_overlay),
            "ic_overlay_orthogonal": _agg(ic_resid),
            "ic_overlay_in_top_base": _agg(ic_cond_top),
            "ic_overlay_in_bottom_base": _agg(ic_cond_bottom),
            "blend_ic_curve": {k: _agg(v) for k, v in blend_ic.items()},
        }
        result["by_horizon"][f"h{h}"] = h_result

        print(f"\n==== horizon = {h}d ====", flush=True)
        print(f"  IC {g_col:<16} mean={_f(h_result['ic_base']['mean'])} ICIR={_f(h_result['ic_base']['icir'])} pos%={_f(h_result['ic_base']['pos_ratio'])} n={h_result['ic_base']['n']}")
        print(f"  IC {overlay_label:<16} mean={_f(h_result['ic_overlay']['mean'])} ICIR={_f(h_result['ic_overlay']['icir'])} pos%={_f(h_result['ic_overlay']['pos_ratio'])} n={h_result['ic_overlay']['n']}")
        print(f"  IC overlay (⊥base 残差)   mean={_f(h_result['ic_overlay_orthogonal']['mean'])} ICIR={_f(h_result['ic_overlay_orthogonal']['icir'])} pos%={_f(h_result['ic_overlay_orthogonal']['pos_ratio'])}")
        print(f"  IC overlay | base 头部     mean={_f(h_result['ic_overlay_in_top_base']['mean'])} pos%={_f(h_result['ic_overlay_in_top_base']['pos_ratio'])}")
        print(f"  IC overlay | base 尾部     mean={_f(h_result['ic_overlay_in_bottom_base']['mean'])} pos%={_f(h_result['ic_overlay_in_bottom_base']['pos_ratio'])}")
        print(f"  混合 IC 曲线 ({g_col}:{overlay_label} -> mean IC):")
        for k in blend_ic:
            print(f"    {k:>7}  {_f(h_result['blend_ic_curve'][k]['mean'])}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\n📄 结果写入 {out_path}")

    return 0


def _f(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
