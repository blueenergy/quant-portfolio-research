#!/usr/bin/env python3
"""Robustness of the low-breadth signal: leave-one-event-out, sub-periods, entry timing.

Leave-one-out is done over *independent market events* rather than raw episodes —
dropping one of 20 clustered episodes changes nothing, dropping the one event
that carries the result is the whole story.

Example:
    python analysis/research/market_spectrum/robust.py --start 20050101
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
    episode_groups,
    episodes,
    forward_return,
    load_breadth,
    load_index_ohlc,
)

GAP = 60
DEFAULT_SPLITS = (("2005-2009", "20050101", "20091231"),
                  ("2010-2017", "20100101", "20171231"),
                  ("2018-2026", "20180101", "20261231"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--breadth", default=str(BREADTH_MERGED))
    p.add_argument("--threshold", type=float, default=0.10)
    p.add_argument("--ma", type=int, default=30)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    args = p.parse_args()

    br = load_breadth(args.breadth)
    cl, op = load_index_ohlc()
    cl = align_to_breadth(cl, br.index)
    op = align_to_breadth(op, br.index)
    if args.start:
        br = br[br.index >= args.start]
        cl, op = cl[cl.index >= args.start], op[op.index >= args.start]
    if args.end:
        br = br[br.index <= args.end]
        cl, op = cl[cl.index <= args.end], op[op.index <= args.end]

    s = br[f"ma{args.ma}"]
    th, h = args.threshold, args.horizon
    print(f"窗口 {br.index.min()} .. {br.index.max()}  MA{args.ma} < {th:.0%}  前瞻 {h} 日")

    print("\n" + "=" * 76)
    print("D. 留一事件法 (每次剔除一个独立市场事件)")
    print("=" * 76)
    for nm in cl.columns:
        ser = cl[nm]
        fwd = forward_return(ser, h)
        ep = episodes(s < th)
        if len(ep) < 5:
            continue
        grp = episode_groups(ep, s.index, GAP)
        cond = fwd.reindex(ep).dropna()
        if len(cond) < 8:
            continue
        gs = pd.Series(grp, index=ep).reindex(cond.index)
        base = fwd.dropna()
        loo = [(g, (cond[gs != g] > 0).mean()) for g in np.unique(gs)]
        vals = np.array([x[1] for x in loo])
        print(f"\n【{nm}】n={len(cond)}  全样本 P(涨)={(cond>0).mean():.1%}  均值={cond.mean():+.2%}  "
              f"基准={ (base>0).mean():.1%}")
        print(f"   独立事件 {len(np.unique(gs))} 个；留一后 P(涨) 区间: {vals.min():.1%} ~ {vals.max():.1%}")
        worst = sorted(loo, key=lambda x: x[1])[:3]
        print("   最敏感的剔除: " + " | ".join(f"事件{g}(→{u:.0%})" for g, u in worst))

    print("\n" + "=" * 76)
    print("E. 分期稳健性")
    print("=" * 76)
    for nm in cl.columns:
        ser = cl[nm]
        fwd = forward_return(ser, h)
        print(f"\n【{nm}】")
        for lab, a, b in DEFAULT_SPLITS:
            m = (s.index >= a) & (s.index <= b)
            if m.sum() < 100:
                continue
            base = fwd[m].dropna()
            ep = episodes(s[m] < th)
            cond = fwd.reindex(ep).dropna()
            cu = f"{(cond>0).mean():.1%}" if len(cond) else "n/a"
            cm = f"{cond.mean():+.2%}" if len(cond) else "n/a"
            print(f"   {lab}: 基准P(涨)={(base>0).mean():.1%} (n={len(base)})  |  "
                  f"<{th:.0%} episode={len(ep)} P(涨)={cu} 均值={cm}")

    print("\n" + "=" * 76)
    print(f"F. 执行时点 (MA{args.ma}<{th:.0%}, 次日开盘买入)")
    print("=" * 76)
    for nm in cl.columns:
        ser, o = cl[nm], op[nm]
        ep = episodes(s < th)
        if len(ep) < 5:
            continue
        print(f"\n【{nm}】")
        print(f"   {'入场':>10} {'horizon':>8} {'n':>5} {'P(涨)':>8} {'均值':>9} {'中位':>9} {'最差':>9}")
        for entry, px0, lag in (("当日收盘", ser, 0), ("次日开盘", o, 1)):
            for hh in (5, 10, 20, 60):
                fwd = forward_return(px0, hh, entry_lag=lag)
                cond = fwd.reindex(ep).dropna()
                if len(cond) < 5:
                    continue
                print(f"   {entry:>10} {hh:>7}d {len(cond):>5} {(cond>0).mean():>8.1%} "
                      f"{cond.mean():>+9.2%} {cond.median():>+9.2%} {cond.min():>+9.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
