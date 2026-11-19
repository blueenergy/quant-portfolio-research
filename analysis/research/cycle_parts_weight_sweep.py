#!/usr/bin/env python3
"""离线扫描遗留 cycle 短/中/长三段子分的权重：算一次前向收益，本地任意重组。

输入 ``precompute_cycle_v2_scores.py --scorer legacy_cycle_parts`` 产出的 parts
文件（symbol/score_date/cycle_short/cycle_medium/cycle_long）。本脚本连一次
Mongo，用 ``ScoreBacktester.build_dataset`` 取 universe 的前向收益与 growth_score，
按 (symbol, score_date) 合并 parts，然后：

  1) 报告三段子分各自的 rank IC / IC-IR（看哪段周期真正带 alpha、哪段在拖后腿）；
  2) 扫描权重组合（单纯形网格或显式列表），对每组 (ws, wm, wl)：
       - 合成 cycle = ws*short + wm*medium + wl*long
       - 若 --growth-blend>0：composite = gw*growth + (1-gw)*cycle（与生产 60:40 同构）
       - 报告该信号的 rank IC / IC-IR、Top-N 超额（均值 / IR）；
  3) 高亮基线 40/35/25，并按目标指标排序。

只做纯横截面 IC 与同期 Top-N 收益，不涉及跨期权重择时，无未来函数。
为防过拟合：请在多个时段 / 多个指数上分别跑，只采信跨期稳定的结论。

Example:
    python analysis/cycle_parts_weight_sweep.py \
        --parts reports/cycle_parts_csi1000.parquet \
        --universe-index csi1000 --start-date 20250101 --end-date 20260501 \
        --horizon 20 --top-n 20 --growth-blend 0.6 \
        --out reports/cycle_parts_sweep_csi1000.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep legacy-cycle short/medium/long sub-score weights offline.")
    p.add_argument("--parts", required=True, help="legacy_cycle_parts 产出的 parquet/csv 路径")
    p.add_argument("--universe-index", nargs="+", default=["csi1000"], help="指数池标签，如 csi1000/csi2000/hs300/csi500")
    p.add_argument("--start-date", required=True, help="YYYYMMDD")
    p.add_argument("--end-date", required=True, help="YYYYMMDD")
    p.add_argument("--horizon", type=int, default=20, help="前向收益持有期（交易日）")
    p.add_argument("--top-n", type=int, default=20, help="Top-N 组合规模")
    p.add_argument(
        "--growth-blend",
        type=float,
        default=0.6,
        help="growth 在 composite 里的权重（0.6=生产 60:40）；设 0 则评估 cycle 单独信号",
    )
    p.add_argument(
        "--weights",
        nargs="*",
        help="显式权重组合，形如 0.4:0.35:0.25（短:中:长）。不给则按 --grid-step 生成单纯形网格",
    )
    p.add_argument("--grid-step", type=float, default=0.1, help="单纯形网格步长（默认 0.1）")
    p.add_argument("--sort-by", default="topn_excess_ir", choices=["topn_excess_ir", "topn_excess_mean", "ic_ir", "ic_mean"])
    p.add_argument("--out", help="结果 JSON 输出路径")
    return p.parse_args()


def _spearman_by_date(df: pd.DataFrame, score_col: str, ret_col: str) -> List[float]:
    """逐 score_date 的 rank IC（秩相关 = 排名后的皮尔逊相关）。"""
    ics: List[float] = []
    for _, g in df.groupby("score_date"):
        sub = g[[score_col, ret_col]].dropna()
        if len(sub) < 3:
            continue
        if sub[score_col].nunique() < 2 or sub[ret_col].nunique() < 2:
            continue
        ic = sub[score_col].rank().corr(sub[ret_col].rank())
        if pd.notna(ic):
            ics.append(float(ic))
    return ics


def _topn_excess_by_date(df: pd.DataFrame, score_col: str, ret_col: str, top_n: int) -> List[float]:
    """逐 score_date 的 Top-N 等权前向收益减去全池均值。"""
    out: List[float] = []
    for _, g in df.groupby("score_date"):
        sub = g[[score_col, ret_col]].dropna()
        if len(sub) < top_n:
            continue
        universe = float(sub[ret_col].mean())
        top = sub.sort_values(score_col, ascending=False).head(top_n)
        out.append(float(top[ret_col].mean()) - universe)
    return out


def _summ(vals: Sequence[float]) -> Dict[str, Optional[float]]:
    if not vals:
        return {"n": 0, "mean": None, "ir": None, "pos_ratio": None}
    m = mean(vals)
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    return {
        "n": len(vals),
        "mean": m,
        "ir": (m / sd) if sd else None,
        "pos_ratio": sum(1 for v in vals if v > 0) / len(vals),
    }


def _parse_weight_arg(text: str) -> Tuple[float, float, float]:
    parts = [float(x) for x in text.replace("/", ":").split(":")]
    if len(parts) != 3:
        raise ValueError(f"权重需为 短:中:长 三个值: {text}")
    total = sum(parts)
    if total <= 0:
        raise ValueError(f"权重之和必须为正: {text}")
    return parts[0] / total, parts[1] / total, parts[2] / total


def _simplex_grid(step: float) -> List[Tuple[float, float, float]]:
    n = int(round(1.0 / step))
    combos = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            combos.append((i / n, j / n, k / n))
    return combos


def main() -> int:
    from analysis.core.score_backtester import BacktestConfig, ScoreBacktester
    from shared.data_access import open_research_data_access_for_cli

    args = _parse_args()
    ret_col = f"forward_return_{args.horizon}d"

    parts_path = Path(args.parts)
    if not parts_path.exists():
        print(f"❌ parts 文件不存在: {parts_path}")
        return 1
    parts = pd.read_parquet(parts_path) if parts_path.suffix == ".parquet" else pd.read_csv(parts_path)
    need = {"symbol", "score_date", "cycle_short", "cycle_medium", "cycle_long"}
    missing = need - set(parts.columns)
    if missing:
        print(f"❌ parts 缺少列: {sorted(missing)}")
        return 1
    parts["symbol"] = parts["symbol"].astype(str)
    parts["score_date"] = parts["score_date"].astype(str)

    db = open_research_data_access_for_cli(enable_tushare=False)
    config = BacktestConfig(
        horizons=(args.horizon,),
        start_date=args.start_date,
        end_date=args.end_date,
        universe_index_codes=tuple(args.universe_index),
    )
    print(
        f"构建数据集: universe={','.join(args.universe_index)} range={args.start_date}~{args.end_date} "
        f"horizon={args.horizon} ...",
        flush=True,
    )
    dataset = ScoreBacktester(db, config).build_dataset()
    if dataset.empty:
        print("❌ build_dataset 为空")
        return 1
    keep = ["symbol", "score_date", "growth_score", ret_col]
    keep = [c for c in keep if c in dataset.columns]
    if ret_col not in keep:
        print(f"❌ 数据集缺少 {ret_col}")
        return 1
    dataset = dataset[keep].copy()
    dataset["symbol"] = dataset["symbol"].astype(str)
    dataset["score_date"] = dataset["score_date"].astype(str)

    df = dataset.merge(parts, on=["symbol", "score_date"], how="inner")
    for c in ("cycle_short", "cycle_medium", "cycle_long", "growth_score", ret_col):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["cycle_short", "cycle_medium", "cycle_long", ret_col])
    if df.empty:
        print("❌ parts 与数据集按 (symbol, score_date) 合并后为空")
        return 1

    use_blend = args.growth_blend > 0 and "growth_score" in df.columns
    if args.growth_blend > 0 and "growth_score" not in df.columns:
        print("⚠️ 数据集无 growth_score，改为评估 cycle 单独信号")
    print(
        f"合并后: rows={len(df)} dates={df['score_date'].nunique()} symbols={df['symbol'].nunique()} "
        f"评估口径={'growth%.0f:cycle%.0f 混合' % (args.growth_blend*100, (1-args.growth_blend)*100) if use_blend else 'cycle 单独'}",
        flush=True,
    )

    # 1) 三段子分各自的 rank IC / IR
    print("\n=== 子分单独 rank IC（横截面，越高越有预测力） ===")
    sub_ic = {}
    for name, col in [("short", "cycle_short"), ("medium", "cycle_medium"), ("long", "cycle_long")]:
        s = _summ(_spearman_by_date(df, col, ret_col))
        sub_ic[name] = s
        print(
            f"  {name:<6} IC_mean={s['mean']:+.4f}  IC_IR={s['ir'] if s['ir'] is None else round(s['ir'],3)}  "
            f"pos={s['pos_ratio'] if s['pos_ratio'] is None else round(s['pos_ratio'],3)}  n={s['n']}"
            if s["mean"] is not None else f"  {name:<6} 无有效 IC"
        )

    # 2) 权重扫描
    if args.weights:
        weight_vectors = [_parse_weight_arg(w) for w in args.weights]
    else:
        weight_vectors = _simplex_grid(args.grid_step)
    baseline = (0.4, 0.35, 0.25)
    if baseline not in weight_vectors:
        weight_vectors = [baseline] + weight_vectors

    results = []
    for ws, wm, wl in weight_vectors:
        cycle = ws * df["cycle_short"] + wm * df["cycle_medium"] + wl * df["cycle_long"]
        if use_blend:
            signal = args.growth_blend * df["growth_score"] + (1 - args.growth_blend) * cycle
        else:
            signal = cycle
        tmp = pd.DataFrame({"score_date": df["score_date"], "signal": signal, ret_col: df[ret_col]})
        ic = _summ(_spearman_by_date(tmp, "signal", ret_col))
        ex = _summ(_topn_excess_by_date(tmp, "signal", ret_col, args.top_n))
        results.append(
            {
                "weights": {"short": round(ws, 4), "medium": round(wm, 4), "long": round(wl, 4)},
                "ic_mean": ic["mean"],
                "ic_ir": ic["ir"],
                "topn_excess_mean": ex["mean"],
                "topn_excess_ir": ex["ir"],
                "topn_excess_pos": ex["pos_ratio"],
                "is_baseline": (ws, wm, wl) == baseline,
            }
        )

    def sort_key(r):
        v = r.get(args.sort_by)
        return (v is None, -(v or 0))

    results.sort(key=sort_key)

    print(f"\n=== 权重扫描（按 {args.sort_by} 降序，共 {len(results)} 组；★=基线40/35/25） ===")
    print("%-3s %5s %5s %5s | %9s %7s | %12s %9s %7s" % (
        "", "短", "中", "长", "IC_mean", "IC_IR", "Top%d超额" % args.top_n, "超额IR", "pos"))
    for r in results[:15]:
        w = r["weights"]
        star = "★" if r["is_baseline"] else " "
        def f(x, p=4): return ("%+.*f" % (p, x)) if isinstance(x, (int, float)) else "   -  "
        print("%-3s %5.2f %5.2f %5.2f | %9s %7s | %12s %9s %7s" % (
            star, w["short"], w["medium"], w["long"],
            f(r["ic_mean"]), f(r["ic_ir"], 3),
            f(r["topn_excess_mean"]), f(r["topn_excess_ir"], 3),
            f(r["topn_excess_pos"], 3)))

    base = next((r for r in results if r["is_baseline"]), None)
    if base is not None:
        rank = results.index(base) + 1
        print(f"\n基线 40/35/25 在 {len(results)} 组中按 {args.sort_by} 排名第 {rank}。")

    if args.out:
        out = {
            "universe_index": args.universe_index,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "horizon": args.horizon,
            "top_n": args.top_n,
            "growth_blend": args.growth_blend if use_blend else 0.0,
            "rows": len(df),
            "dates": int(df["score_date"].nunique()),
            "sub_score_ic": sub_ic,
            "sort_by": args.sort_by,
            "results": results,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"📄 结果写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
