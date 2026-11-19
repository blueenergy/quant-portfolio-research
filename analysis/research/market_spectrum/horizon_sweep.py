#!/usr/bin/env python3
"""Does the 阳谱 edge live in the first few days, or is it a 20-day slow burn?

The 20-day result is hard to trade: you sit through a month for ~+2-4% mean,
and the tail is brutal (worst 20d -22%).  If the edge is front-loaded, a 3-5
day hold captures most of it with far less exposure; if it is a slow burn,
there is nothing to shorten.

Sections:
  T. 持有期 × 阈值 网格 (条件 P(涨) 与均值)
  U. 超额矩阵 (条件 − 基准), 带 permutation p
  V. 事件时路径: episode 后 t+1..t+20 的平均累计收益 (超额是前置还是后置)
  W. 分期 × 短周期: 超额在三个窗口里是否同向

Example:
    python analysis/research/market_spectrum/horizon_sweep.py --threshold 0.15
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
    ols,
    wilson,
)

GAP = 60
HORIZONS = (1, 2, 3, 5, 10, 20)
PERIODS = (
    ("2005-2009", "20050101", "20091231"),
    ("2010-2017", "20100101", "20171231"),
    ("2018-2026", "20180101", "20261231"),
    ("2005-2026 (全)", "20050101", "20261231"),
)


def stats(s: pd.Series, ser: pd.Series, th: float, h: int, a: str, b: str,
          n_perm: int, side: str = "low") -> dict | None:
    m = (s.index >= a) & (s.index <= b)
    if m.sum() < 120:
        return None
    fwd = forward_return(ser, h)
    base = fwd[m].dropna()
    if len(base) < 80:
        return None
    ep = episodes(s[m] > th if side == "high" else s[m] < th)
    cond = fwd.reindex(ep).dropna()
    if len(cond) < 5:
        return None
    cu = (cond > 0).mean()
    lo, hi = wilson(int((cond > 0).sum()), len(cond))
    return {
        "n": len(cond),
        "groups": len(np.unique(episode_groups(ep, s.index, GAP))),
        "base_p": base.gt(0).mean(),
        "base_m": base.mean(),
        "cond_p": cu,
        "lo": lo, "hi": hi,
        "cond_m": cond.mean(),
        "p": circular_perm_p(cu, cond.index, fwd, n_perm=n_perm),
    }


def grid(s: pd.Series, ser: pd.Series, nm: str, thresholds, n_perm: int,
         a: str, b: str, side: str = "low") -> None:
    """T + U: horizon x threshold, conditional level and excess over baseline."""
    print(f"\n【{nm}】 {a}..{b}")
    print(f"   {'阈值':>5} {'horizon':>7} {'episode':>7} {'独立':>4} "
          f"{'基准P':>7} {'条件P':>7} {'ΔP':>7} {'基准均值':>8} {'条件均值':>8} "
          f"{'Δ均值':>8} {'perm-p':>7}")
    for th in thresholds:
        for h in HORIZONS:
            r = stats(s, ser, th, h, a, b, n_perm, side)
            if r is None:
                continue
            dp = r["cond_p"] - r["base_p"]
            dm = r["cond_m"] - r["base_m"]
            flag = " *" if r["p"] < 0.05 else ""
            print(f"   {th:>4.0%} {h:>6}d {r['n']:>7} {r['groups']:>4} "
                  f"{r['base_p']:>7.1%} {r['cond_p']:>7.1%} {dp:>+7.1%} "
                  f"{r['base_m']:>+8.2%} {r['cond_m']:>+8.2%} {dm:>+8.2%} "
                  f"{r['p']:>7.3f}{flag}")
        print()


def event_path(s: pd.Series, ser: pd.Series, nm: str, th: float,
               a: str, b: str, kmax: int = 20, side: str = "low") -> None:
    """V: average cumulative return from the episode close out to t+k.

    Front-loaded edge -> most of the gain is in t+1..t+5 and holding 20 days
    just adds tail risk.  Back-loaded -> shortening the hold throws it away.
    """
    m = (s.index >= a) & (s.index <= b)
    ep = episodes(s[m] > th if side == "high" else s[m] < th)
    if len(ep) == 0:
        print(f"   【{nm}】该窗口无 episode")
        return
    px = ser.dropna()
    pos = [px.index.get_loc(d) for d in ep if d in px.index]
    if not pos:
        print(f"   【{nm}】episode 全部落在指数样本外")
        return
    vals = px.to_numpy(dtype=float)
    n = len(vals)
    paths = []
    for p0 in pos:
        path = [np.nan] * (kmax + 1)
        for k in range(kmax + 1):
            if p0 + k < n:
                path[k] = vals[p0 + k] / vals[p0] - 1
        paths.append(path)
    P = np.array(paths)
    # baseline: same-length paths started on every non-episode day
    epset = set(pos)
    bpaths = []
    for p0 in range(n):
        if p0 in epset or p0 + kmax >= n:
            continue
        bpaths.append(vals[p0:p0 + kmax + 1] / vals[p0] - 1)
    B = np.array(bpaths)
    print(f"   {nm}  阳谱{'≥' if side == 'high' else '<'}{th:.0%}  n={len(pos)} episode  "
          f"(基准 n={len(bpaths)})")
    print(f"   {'t+k':>5} {'事件均值':>9} {'事件中位':>9} {'事件P(>0)':>10} "
          f"{'基准均值':>9} {'超额':>9}")
    for k in (1, 2, 3, 5, 10, 20):
        if k > kmax:
            break
        e = P[:, k][~np.isnan(P[:, k])]
        bb = B[:, k]
        print(f"   {k:>4}d {e.mean():>+9.2%} {np.median(e):>+9.2%} "
              f"{(e > 0).mean():>10.1%} {bb.mean():>+9.2%} "
              f"{e.mean()-bb.mean():>+9.2%}")


def incremental_short(s: pd.Series, ser: pd.Series, nm: str, horizons,
                      a: str, b: str) -> None:
    """X: is the short-horizon edge breadth, or just the index's own momentum?

    A high 阳谱 reading means the index just went up.  If 1-5 day forward
    returns are positively autocorrelated, "阳谱>=90% -> up tomorrow" is
    momentum wearing a breadth costume.  Control for the trailing return and
    see whether breadth keeps any coefficient.
    """
    past = ser / ser.shift(20) - 1
    print(f"   【{nm}】 控制「过去20日收益」后, 阳谱 是否还有独立信息")
    print(f"   {'h':>3} {'n':>5} {'corr(阳谱,过去)':>15} {'过去斜率':>10} "
          f"{'阳谱系数':>10} {'ΔR²':>8} {'bootstrap 95%CI':>20}")
    rng = np.random.default_rng(7)
    for h in horizons:
        fwd = forward_return(ser, h)
        j = pd.concat([s.rename("b"), past.rename("p"), fwd.rename("f")],
                      axis=1).dropna()
        j = j[(j.index >= a) & (j.index <= b)]
        if len(j) < 200:
            continue
        y = j.f.to_numpy()
        X1 = j.p.to_numpy().reshape(-1, 1)
        _, r2_1 = ols(X1, y)
        X2 = np.column_stack([X1[:, 0], j.b.to_numpy()])
        b2, r2_2 = ols(X2, y)
        n = len(y)
        bs = []
        for _ in range(1500):
            i = rng.integers(0, n, n)
            A = np.column_stack([np.ones(n), X2[i]])
            if np.linalg.matrix_rank(A) == 3:
                bs.append(np.linalg.lstsq(A, y[i], rcond=None)[0][2])
        lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
        mark = "含0" if lo * hi <= 0 else "不含0"
        print(f"   {h:>2}d {n:>5} {j.b.corr(j.p):>15.2f} {b2[1]:>+10.3f} "
              f"{b2[2]:>+10.3f} {r2_2-r2_1:>+8.4f} "
              f"{'['+f'{lo:+.3f},{hi:+.3f}'+']':>20}  {mark}")


def double_sort(s: pd.Series, ser: pd.Series, nm: str, horizons=(1, 5),
                a: str = "20050101", b: str = "20261231") -> None:
    """Y: same trailing-return bucket, high 阳谱 vs low 阳谱.

    A linear control cannot absorb "阳谱>=90% is a persistent state".  Bucket
    on the trailing 20d return instead and compare 阳谱 levels inside each
    bucket: if high 阳谱 still wins there, it is more than momentum.
    """
    past = ser / ser.shift(20) - 1
    edges = [-9, -0.10, -0.05, 0.0, 0.05, 0.10, 9]
    labs = ["跌>10%", "跌5-10%", "跌0-5%", "涨0-5%", "涨5-10%", "涨>10%"]
    pb = pd.cut(past, edges, labels=labs)
    print(f"   【{nm}】 行=过去20日收益分组, 列=同期阳谱分组 → P(涨) / 均值")
    for h in horizons:
        fwd = forward_return(ser, h)
        j = pd.concat([s.rename("b"), pb.rename("pb"), fwd.rename("f")],
                      axis=1).dropna()
        j = j[(j.index >= a) & (j.index <= b)]
        j["bb"] = pd.cut(j.b, [0, 0.30, 0.70, 0.90, 1.0],
                         labels=["<30%", "30-70%", "70-90%", "≥90%"])
        print(f"   ---- 前瞻 {h} 日 (全样本基准 P={j.f.gt(0).mean():.1%}) ----")
        hdr = "".join(f"{c:>18}" for c in j.bb.cat.categories)
        print(f"   {'':>10}{hdr}")
        for lab in labs:
            cells = []
            for c in j.bb.cat.categories:
                g = j[(j.pb == lab) & (j.bb == c)].f
                cells.append(f"{g.gt(0).mean():>8.0%}/{g.mean():>+8.2%}"
                             if len(g) >= 15 else f"{'n/a':>17}")
            print(f"   {lab:>10}" + "".join(cells))
        print()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--breadth", default=str(BREADTH_MERGED))
    p.add_argument("--ma", type=int, default=30)
    p.add_argument("--threshold", type=float, default=None,
                   help="事件时路径用的阈值; 默认 low=0.15 / high=0.80")
    p.add_argument("--index", default="",
                   help="留空=全部指数; 或单个指数名 (见 common.INDEX_NAMES)")
    p.add_argument("--n-perm", type=int, default=2000)
    p.add_argument("--side", choices=("low", "high"), default="low",
                   help="low=阳谱低于阈值 (抄底); high=高于阈值 (动量)")
    args = p.parse_args()
    if args.threshold is None:
        args.threshold = 0.80 if args.side == "high" else 0.15

    br = load_breadth(args.breadth)
    idx = align_to_breadth(load_index_closes(), br.index)
    names = [args.index] if args.index else list(idx.columns)
    for nm in names:
        if nm not in idx.columns:
            raise SystemExit(f"unknown index {nm}; have {list(idx.columns)}")
    s = br[f"ma{args.ma}"]
    thresholds = ((0.70, 0.80, 0.85, 0.90, 0.95) if args.side == "high"
                  else (0.08, 0.10, 0.12, 0.15, 0.20))

    for nm in names:
        ser = idx[nm]
        print("=" * 108)
        print(f"持有期扫描 — {nm}, MA{args.ma}, breadth {br.index.min()}..{br.index.max()}")
        print("=" * 108)
        print("\nT/U. 持有期 × 阈值: 条件 P(涨) 与超额 (全样本)")
        grid(s, ser, nm, thresholds, args.n_perm, "20050101", "20261231", args.side)

        print("=" * 108)
        print(f"V. 事件时路径 — 阳谱{'≥' if args.side == 'high' else '<'}"
              f"{args.threshold:.0%}, 全样本")
        print("=" * 108)
        event_path(s, ser, nm, args.threshold, "20050101", "20261231", 20, args.side)
        print()

        print("=" * 108)
        print(f"W. 分期 × 短周期 — 阳谱{'≥' if args.side == 'high' else '<'}"
              f"{args.threshold:.0%}")
        print("=" * 108)
        print(f"   {'窗口':>16} {'h':>3} {'episode':>7} {'独立':>4} {'基准P':>7} "
              f"{'条件P':>7} {'ΔP':>7} {'Δ均值':>8} {'perm-p':>7}")
        for lab, a, b in PERIODS:
            for h in (1, 3, 5, 20):
                r = stats(s, ser, args.threshold, h, a, b, args.n_perm, args.side)
                if r is None:
                    print(f"   {lab:>16} {h:>2}d {'(样本不足)':>10}")
                    continue
                print(f"   {lab:>16} {h:>2}d {r['n']:>7} {r['groups']:>4} "
                      f"{r['base_p']:>7.1%} {r['cond_p']:>7.1%} "
                      f"{r['cond_p']-r['base_p']:>+7.1%} "
                      f"{(r['cond_m']-r['base_m']):>+8.2%} {r['p']:>7.3f}")
            print()

        print("=" * 108)
        print("X. 短周期增量检验: 高/低阳谱的超额, 是不是指数自己的动量/反转?")
        print("=" * 108)
        incremental_short(s, ser, nm, (1, 2, 3, 5, 20), "20050101", "20261231")
        print()

        print("=" * 108)
        print("Y. 双排序: 同一「过去20日收益」分组内, 高阳谱是否仍然更好?")
        print("=" * 108)
        double_sort(s, ser, nm, (1, 5))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
