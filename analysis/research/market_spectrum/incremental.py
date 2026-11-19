#!/usr/bin/env python3
"""The falsification test: is 阴阳谱 just a proxy for "the index just fell"?

This is the part that killed the signal.  A raw "low breadth -> bounce" hit rate
looks impressive until you control for the trailing index return, because low
breadth and a recent drawdown are nearly the same observation.

Three views:
  I.  regress forward return on past-20d return, then add breadth — does ΔR² move?
  J.  within a fixed drawdown bucket, does breadth still separate outcomes?
  J2. within a fixed *breadth* bucket, does the drawdown still separate outcomes?

If breadth survives I, J and J2 it is its own variable.  It does not.

Example:
    python analysis/research/market_spectrum/incremental.py --start 20050101
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
    ols,
)


def boot_ci_coef(X: np.ndarray, y: np.ndarray, col: int, n_boot: int = 2000,
                 seed: int = 3) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        Xi, yi = X[i], y[i]
        A = np.column_stack([np.ones(len(Xi)), Xi])
        if np.linalg.matrix_rank(A) == A.shape[1]:
            out.append(np.linalg.lstsq(A, yi, rcond=None)[0][col])
    return tuple(np.percentile(out, [2.5, 97.5]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--breadth", default=str(BREADTH_MERGED))
    p.add_argument("--ma", type=int, default=30)
    p.add_argument("--horizon", type=int, default=20)
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

    print("\n" + "=" * 82)
    print("I. 控制「过去20日指数收益」后, 阴阳谱 是否还有独立信息?")
    print("=" * 82)
    for nm in idx.columns:
        ser = idx[nm]
        past = ser / ser.shift(20) - 1
        fwd = forward_return(ser, h)
        j = pd.concat([s.rename("b"), past.rename("past"), fwd.rename("f")], axis=1).dropna()
        if len(j) < 200:
            continue
        y = j.f.to_numpy()
        b1, r2_1 = ols(j.past.to_numpy(), y)
        X2 = np.column_stack([j.past.to_numpy(), j.b.to_numpy()])
        b2, r2_2 = ols(X2, y)
        lo, hi = boot_ci_coef(X2, y, col=2)
        verdict = "不含0, 有独立信息" if lo * hi > 0 else "含0, 无独立信息"
        print(f"\n【{nm}】 n={len(j)}")
        print(f"  模型1  未来{h}日 ~ 过去20日收益        : 斜率={b1[1]:+.3f}  R²={r2_1:.4f}")
        print(f"  模型2  未来{h}日 ~ 过去20日 + 阳谱     : 过去={b2[1]:+.3f}  阳谱={b2[2]:+.3f}  "
              f"R²={r2_2:.4f}  ΔR²={r2_2-r2_1:+.4f}")
        print(f"     阳谱系数 bootstrap 95%CI = [{lo:+.3f}, {hi:+.3f}]  ← {verdict}")

    print("\n" + "=" * 82)
    print("J. 同一跌幅分组内, 低阳谱 是否更好?")
    print("=" * 82)
    for nm in idx.columns:
        ser = idx[nm]
        past = ser / ser.shift(20) - 1
        fwd = forward_return(ser, h)
        j = pd.concat([s.rename("b"), past.rename("past"), fwd.rename("f")], axis=1).dropna()
        if len(j) < 200:
            continue
        j["bucket"] = pd.cut(j.past, [-1, -0.10, -0.05, 0, 0.05, 1],
                             labels=["跌>10%", "跌5-10%", "跌0-5%", "涨0-5%", "涨>5%"])
        print(f"\n【{nm}】")
        print(f"   {'跌幅分组':>10} {'阳谱<15%':>26} {'阳谱>=15%':>26}")
        for bk, g in j.groupby("bucket", observed=True):
            a, b_ = g[g.b < 0.15]["f"], g[g.b >= 0.15]["f"]
            sa = (f"n={len(a):<4} P涨={(a>0).mean():.0%} 均值={a.mean():+.2%}"
                  if len(a) >= 10 else "样本不足")
            sb = (f"n={len(b_):<4} P涨={(b_>0).mean():.0%} 均值={b_.mean():+.2%}"
                  if len(b_) >= 10 else "样本不足")
            print(f"   {str(bk):>10} {sa:>26} {sb:>26}")

    print("\n" + "=" * 82)
    print("J2. 反向对照: 同一阳谱分组内, 不同跌幅 是否更好?")
    print("=" * 82)
    for nm in idx.columns:
        ser = idx[nm]
        past = ser / ser.shift(20) - 1
        fwd = forward_return(ser, h)
        j = pd.concat([s.rename("b"), past.rename("past"), fwd.rename("f")], axis=1).dropna()
        if len(j) < 200:
            continue
        j["bb"] = pd.cut(j.b, [-1, 0.15, 0.30, 1], labels=["阳谱<15%", "15-30%", ">30%"])
        j["pb"] = pd.cut(j.past, [-1, -0.10, -0.05, 1], labels=["跌>10%", "跌5-10%", "跌<5%"])
        print(f"\n【{nm}】 行=阳谱分组, 列=过去20日跌幅  →  P(涨) / 均值")
        hdr = "".join(f"{str(c):>18}" for c in j.pb.cat.categories)
        print(f"   {'':>10}{hdr}")
        for bb, g in j.groupby("bb", observed=True):
            cells = []
            for pb in j.pb.cat.categories:
                gg = g[g.pb == pb]["f"]
                cells.append(f"{(gg>0).mean():.0%} / {gg.mean():+.2%}" if len(gg) >= 10 else "n/a")
            print(f"   {str(bb):>10}" + "".join(f"{c:>18}" for c in cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
