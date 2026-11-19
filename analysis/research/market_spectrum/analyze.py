#!/usr/bin/env python3
"""Descriptive shape of the breadth series and its raw correlation with indices.

Step 1 of the study: before any threshold or event framing, how is 阴阳谱
distributed, and does its *level* line up with index returns at all — same day,
and forward?

Example:
    python analysis/research/market_spectrum/analyze.py --start 20050101
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
    load_breadth,
    load_index_closes,
)

HORIZONS = (1, 5, 10, 20, 60)
THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30)


def spear(a, b) -> float:
    a, b = pd.Series(a).rank(), pd.Series(b).rank()
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def pear(a, b) -> float:
    a, b = pd.Series(a), pd.Series(b)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--breadth", default=str(BREADTH_MERGED))
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    args = p.parse_args()

    br = load_breadth(args.breadth)
    if args.start:
        br = br[br.index >= args.start]
    if args.end:
        br = br[br.index <= args.end]

    idx = align_to_breadth(load_index_closes(), br.index)
    print("index coverage:")
    for nm in idx.columns:
        print(f"  {nm}: {idx[nm].first_valid_index()} .. {idx[nm].last_valid_index()}  "
              f"n={idx[nm].notna().sum()}")
    print(f"\nbreadth window: {br.index.min()} .. {br.index.max()}  ({len(br)} days)")

    print(f"\n=== 1. 阴阳谱分布 ({br.index.min()}–{br.index.max()}) ===")
    for k in ("ma5", "ma30"):
        s = br[k].dropna()
        qs = " ".join(f"P{int(q*100):02d}={s.quantile(q):.3f}" for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99))
        print(f"\nMA{k[2:]} n={len(s)} mean={s.mean():.3f} std={s.std():.3f}")
        print("   " + qs)
        for th in THRESHOLDS:
            print(f"   <{th:.0%}: {(s < th).sum():4d} 天 ({(s < th).mean():5.1%})")

    print("\n=== 2. 同期相关 (breadth_t vs 指数当日收益) ===")
    for k in ("ma5", "ma30"):
        for nm in idx.columns:
            r = idx[nm].pct_change()
            j = pd.concat([br[k], r], axis=1).dropna()
            print(f" MA{k[2:]} {nm}: pearson={pear(j.iloc[:, 0], j.iloc[:, 1]):+.3f} "
                  f"spearman={spear(j.iloc[:, 0], j.iloc[:, 1]):+.3f} n={len(j)}")

    print("\n=== 3. 前瞻相关 (breadth_t vs 未来N日收益, Spearman) ===")
    for k in ("ma5", "ma30"):
        print(f"\n MA{k[2:]}   " + "".join(f"{h:>9}d" for h in HORIZONS))
        for nm in idx.columns:
            ser = idx[nm]
            cells = []
            for h in HORIZONS:
                fwd = ser.shift(-h) / ser - 1
                j = pd.concat([br[k], fwd], axis=1).dropna()
                cells.append(f"{spear(j.iloc[:, 0], j.iloc[:, 1]):>+10.3f}")
            print("  " + nm.ljust(8) + "".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
