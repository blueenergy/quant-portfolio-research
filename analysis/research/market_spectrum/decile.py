#!/usr/bin/env python3
"""Decile dose-response and the low-region slope.

The decile table comes out U-shaped (both tails positive).  That is an ex-post
shape, not a rule: the high tail is trend continuation, and the low tail is the
same "index just fell" effect tested in incremental.py.  Do not read it as
"buy both ends".

Example:
    python analysis/research/market_spectrum/decile.py --start 20050101
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
    forward_return,
    load_breadth,
    load_index_closes,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--breadth", default=str(BREADTH_MERGED))
    p.add_argument("--ma", type=int, default=30)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--low-cut", type=float, default=0.30)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
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

    print("\n" + "=" * 80)
    print(f"G. 十分位剂量-反应")
    print("=" * 80)
    for nm in idx.columns:
        ser = idx[nm]
        fwd = forward_return(ser, h)
        j = pd.concat([s.rename("b"), fwd.rename("f")], axis=1).dropna()
        if len(j) < 200:
            continue
        j["dec"] = pd.qcut(j["b"], 10, labels=False)
        print(f"\n【{nm}】 (基准 P(涨)={(j['f']>0).mean():.1%}, 均值={j['f'].mean():+.2%}, n={len(j)})")
        print(f"   {'十分位':>6} {'阳谱区间':>18} {'n':>5} {'P(涨)':>8} {'均值':>9} {'中位':>9}")
        for d in range(10):
            g = j[j["dec"] == d]
            rng = "[" + f"{g.b.min():.1%},{g.b.max():.1%}" + "]"
            print(f"   D{d+1:<5} {rng:>18} {len(g):>5} {(g.f>0).mean():>8.1%} "
                  f"{g.f.mean():>+9.2%} {g.f.median():>+9.2%}")
        r = j.groupby("dec")["f"].mean()
        print(f"   → 十分位 vs 均值收益 相关: {pd.Series(r.values).corr(pd.Series(r.index.astype(float))):+.3f}")

    print("\n" + "=" * 80)
    print(f"H. 低位区间回归: 阳谱<{args.low_cut:.0%} 时, 阳谱水平 vs 未来{h}日收益")
    print("=" * 80)
    for nm in idx.columns:
        ser = idx[nm]
        fwd = forward_return(ser, h)
        j = pd.concat([s.rename("b"), fwd.rename("f")], axis=1).dropna()
        low = j[j.b < args.low_cut]
        if len(low) < 50:
            continue
        x, y = low.b.to_numpy(), low.f.to_numpy()
        b1, b0 = np.polyfit(x, y, 1)
        r2 = 1 - ((y - (b0 + b1 * x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rng = np.random.default_rng(11)
        n = len(x)
        slopes = []
        for _ in range(2000):
            i = rng.integers(0, n, n)
            if np.std(x[i]) > 0:
                slopes.append(np.polyfit(x[i], y[i], 1)[0])
        lo, hi = np.percentile(slopes, [2.5, 97.5])
        print(f" 【{nm}】 n={n}  斜率={b1:+.2f}  R²={r2:.3f}  bootstrap95%CI=[{lo:+.2f},{hi:+.2f}]")
        print(f"      含义: 阳谱每低 10pp, 未来{h}日均值收益 {-b1*0.10*100:+.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
