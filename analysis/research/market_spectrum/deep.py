#!/usr/bin/env python3
"""Threshold sweep with intervals, event clustering, and the reverse (>80%) side.

The sweep covers 7 thresholds x 3 indices.  That is 21 looks, so a lone p~0.05
is what noise produces — read the dose-response shape and the independent-event
count, not the smallest p.

Example:
    python analysis/research/market_spectrum/deep.py --start 20050101
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from analysis.research.market_spectrum.common import (  # noqa: E402
    BREADTH_MERGED,
    align_to_breadth,
    circular_perm_p,
    episode_groups,
    episodes,
    forward_return,
    load_breadth,
    load_index_closes,
    wilson,
)

THRESHOLDS = (0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30)
CLUSTER_LEVELS = (0.10, 0.15, 0.20, 0.25)
GAP = 60


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--breadth", default=str(BREADTH_MERGED))
    p.add_argument("--ma", type=int, default=30)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--n-perm", type=int, default=3000)
    args = p.parse_args()

    br = load_breadth(args.breadth)
    idx = align_to_breadth(load_index_closes(), br.index)
    if args.start:
        br, idx = br[br.index >= args.start], idx[idx.index >= args.start]
    if args.end:
        br, idx = br[br.index <= args.end], idx[idx.index <= args.end]

    s = br[f"ma{args.ma}"]
    h = args.horizon
    print(f"窗口 {br.index.min()} .. {br.index.max()}   MA{args.ma}   前瞻 {h} 日")

    print("\n" + "=" * 78)
    print("A. episode 时间聚集性")
    print("=" * 78)
    for th in CLUSTER_LEVELS:
        ep = episodes(s < th)
        yrs = pd.Series(ep).str[:4].value_counts().sort_index()
        g = episode_groups(ep, s.index, GAP)
        n_grp = len(np.unique(g)) if len(g) else 0
        print(f" <{th:.0%}: {len(ep):3d} episode 跨 {n_grp:2d} 个独立市场事件(间隔>{GAP}交易日)")
        print(f"     年份: {dict(yrs)}")

    print("\n" + "=" * 78)
    print(f"B. 阈值扫描: 条件 P({h}日后上涨) vs 基准, 带 Wilson 95% CI")
    print("=" * 78)
    for nm in idx.columns:
        ser = idx[nm]
        fwd = forward_return(ser, h)
        base = fwd.dropna()
        n = len(base)
        k = int((base > 0).sum())
        lo, hi = wilson(k, n)
        print(f"\n【{nm}】基准 {h}日 P(涨)={k/n:.1%} [{lo:.1%},{hi:.1%}]  n={n}")
        print(f"   {'阈值':>6} {'episode':>8} {'独立事件':>9} {'条件P(涨)':>10} {'95%CI':>18} "
              f"{'条件均值':>10} {'perm-p':>8}")
        for th in THRESHOLDS:
            ep = episodes(s < th)
            cond = fwd.reindex(ep).dropna()
            if len(cond) < 8:
                print(f"   {th:>5.0%} {len(ep):>8} {'-':>9} (样本不足)")
                continue
            grp = len(np.unique(episode_groups(ep, s.index, GAP)))
            c_up = (cond > 0).mean()
            c_lo, c_hi = wilson(int((cond > 0).sum()), len(cond))
            pv = circular_perm_p(c_up, cond.index, fwd, n_perm=args.n_perm)
            print(f"   {th:>5.0%} {len(ep):>8} {grp:>9} {c_up:>10.1%} "
                  f"{'['+f'{c_lo:.1%},{c_hi:.1%}'+']':>18} {cond.mean():>+10.2%} {pv:>8.3f}")

    print("\n" + "=" * 78)
    print(f"C. 反向: 阳谱 > 80% 之后 (MA{args.ma})")
    print("=" * 78)
    for nm in idx.columns:
        ser = idx[nm]
        print(f"\n【{nm}】")
        for hh in (20, 60):
            fwd = forward_return(ser, hh)
            base = fwd.dropna()
            ep = episodes(s > 0.80)
            cond = fwd.reindex(ep).dropna()
            if len(cond) < 8:
                continue
            print(f"  {hh}d: 基准P(涨)={(base>0).mean():.1%}  条件P(涨)={(cond>0).mean():.1%}  "
                  f"episode={len(ep)}  条件均值={cond.mean():+.2%} vs 基准={base.mean():+.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
