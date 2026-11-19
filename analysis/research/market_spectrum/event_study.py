#!/usr/bin/env python3
"""Event study: index behaviour after 阳谱 drops below a threshold (MA5 / MA30).

Significance comes from a circular-shift permutation of the forward-return
series rather than a t-test, because overlapping 20/60-day windows make the
observations heavily autocorrelated and a naive test would be wildly optimistic.

Example:
    python analysis/research/market_spectrum/event_study.py --threshold 0.20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from analysis.research.market_spectrum.common import (  # noqa: E402
    BREADTH_MERGED,
    align_to_breadth,
    circular_perm_p,
    episodes,
    forward_return,
    load_breadth,
    load_index_closes,
)

HORIZONS = (5, 10, 20, 60)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--breadth", default=str(BREADTH_MERGED))
    p.add_argument("--threshold", type=float, default=0.20)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--n-perm", type=int, default=2000)
    args = p.parse_args()

    br = load_breadth(args.breadth)
    idx = align_to_breadth(load_index_closes(), br.index)
    if args.start:
        br, idx = br[br.index >= args.start], idx[idx.index >= args.start]
    if args.end:
        br, idx = br[br.index <= args.end], idx[idx.index <= args.end]

    print("=" * 74)
    print(f"事件研究：阳谱 < {args.threshold:.0%} 之后，指数未来 N 日表现")
    print(f"窗口 {br.index.min()} .. {br.index.max()}")
    print("=" * 74)

    for k in ("ma30", "ma5"):
        s = br[k]
        ep = episodes(s < args.threshold)
        print(f"\n### MA{k[2:]} 阳谱 < {args.threshold:.0%} "
              f"（交易日 {(s < args.threshold).sum()} 天，去重后 episode {len(ep)} 个）")
        for nm in idx.columns:
            ser = idx[nm]
            print(f"\n  【{nm}】")
            print(f"    {'horizon':>8} {'基准P(涨)':>10} {'条件P(涨)':>10} {'差':>8} "
                  f"{'基准均值':>9} {'条件均值':>9} {'条件中位':>9} {'perm-p':>8}")
            for h in HORIZONS:
                fwd = forward_return(ser, h)
                base = fwd.dropna()
                cond = fwd.reindex(ep).dropna()
                if len(cond) < 5:
                    continue
                b_up, c_up = (base > 0).mean(), (cond > 0).mean()
                pv = circular_perm_p(c_up, cond.index, fwd, n_perm=args.n_perm)
                print(f"    {h:>7}d {b_up:>10.1%} {c_up:>10.1%} {c_up-b_up:>+8.1%} "
                      f"{base.mean():>+9.2%} {cond.mean():>+9.2%} {cond.median():>+9.2%} "
                      f"{pv:>8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
