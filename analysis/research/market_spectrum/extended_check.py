#!/usr/bin/env python3
"""Does the conclusion survive the pre-2010 extension?

The original study ran 2010-2026 because ``volume_price`` starts there.  The
Tushare extension adds 2005-2009, which contains the one regime the original
sample never saw: the 2008 crash, where the index fell ~70% and buying dips
failed repeatedly for a year.  If "low 阳谱 -> bounce" is real it should show up
there too; if it is an artifact of a sample with only shallow drawdowns, 2008 is
exactly where it breaks.

Sections:
  P.  分期对照总表 (same test, several windows)
  Q.  2005-2009 独立事件逐个清单
  R.  分期增量检验 (control for trailing 20d return, per period)
  S.  结论对照: 2010 起 vs 2005 起

Example:
    python analysis/research/market_spectrum/extended_check.py
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
    INDEX_NAMES,
    INDEX_PUBLISHED,
    align_to_breadth,
    circular_perm_p,
    episode_groups,
    episodes,
    forward_return,
    is_backfilled,
    load_breadth,
    load_index_closes,
    ols,
    wilson,
)

GAP = 60
PERIODS = (
    ("2005-2009", "20050101", "20091231"),
    ("2010-2017", "20100101", "20171231"),
    ("2018-2026", "20180101", "20261231"),
    ("2010-2026 (原)", "20100101", "20261231"),
    ("2005-2026 (全)", "20050101", "20261231"),
)
THRESHOLDS = (0.08, 0.10, 0.12, 0.15, 0.20)


def periods_table(s: pd.Series, ser: pd.Series, horizon: int, thresholds, n_perm: int) -> None:
    fwd = forward_return(ser, horizon)
    print(f"   {'阈值':>6} {'窗口':>16} {'episode':>8} {'独立':>5} {'基准P':>7} "
          f"{'条件P':>7} {'95%CI':>16} {'条件均值':>9} {'perm-p':>7} {'n':>5}")
    for th in thresholds:
        for lab, a, b in PERIODS:
            m = (s.index >= a) & (s.index <= b)
            if m.sum() < 120:
                continue
            base = fwd[m].dropna()
            if len(base) < 80:
                continue
            ep = episodes(s[m] < th)
            cond = fwd.reindex(ep).dropna()
            if len(cond) < 5:
                print(f"   {th:>5.0%} {lab:>16} {len(ep):>8} {'-':>5} {base.gt(0).mean():>7.1%} "
                      f"{'(样本不足)':>7}")
                continue
            grp = len(np.unique(episode_groups(ep, s.index, GAP)))
            cu = (cond > 0).mean()
            lo, hi = wilson(int((cond > 0).sum()), len(cond))
            pv = circular_perm_p(cu, cond.index, fwd, n_perm=n_perm)
            print(f"   {th:>5.0%} {lab:>16} {len(ep):>8} {grp:>5} {base.gt(0).mean():>7.1%} "
                  f"{cu:>7.1%} {'['+f'{lo:.1%},{hi:.1%}'+']':>16} {cond.mean():>+9.2%} "
                  f"{pv:>7.3f} {len(cond):>5}")
        print()


def event_listing(s: pd.Series, ser: pd.Series, th: float, horizon: int,
                  a: str, b: str) -> None:
    m = (s.index >= a) & (s.index <= b)
    fwd = forward_return(ser, horizon)
    ep = episodes(s[m] < th)
    if len(ep) == 0:
        print("   （该窗口无事件）")
        return
    grp = episode_groups(ep, s.index, GAP)
    print(f"   {'事件':>4} {'首日':>10} {'阳谱':>7} {'过去20日':>9} "
          f"{'后'+str(horizon)+'日':>9} {'后60日':>9}")
    past20 = ser / ser.shift(20) - 1
    f60 = forward_return(ser, 60)
    for g in np.unique(grp):
        d = ep[grp == g][0]
        r20 = fwd.get(d, np.nan)
        r60 = f60.get(d, np.nan)
        p20 = past20.get(d, np.nan)
        print(f"   {g:>4} {d:>10} {s.get(d, np.nan):>7.1%} {p20:>+9.2%} "
              f"{r20:>+9.2%} {r60:>+9.2%}")


def incremental_by_period(s: pd.Series, ser: pd.Series, horizon: int) -> None:
    past = ser / ser.shift(20) - 1
    fwd = forward_return(ser, horizon)
    j = pd.concat([s.rename("b"), past.rename("past"), fwd.rename("f")], axis=1).dropna()
    print(f"   {'窗口':>16} {'n':>5} {'过去20日斜率':>13} {'R²(仅过去)':>11} "
          f"{'阳谱系数':>10} {'ΔR²':>8} {'95%CI':>20}")
    for lab, a, b in PERIODS:
        g = j[(j.index >= a) & (j.index <= b)]
        if len(g) < 150:
            continue
        y = g.f.to_numpy()
        b1, r2_1 = ols(g.past.to_numpy(), y)
        X2 = np.column_stack([g.past.to_numpy(), g.b.to_numpy()])
        b2, r2_2 = ols(X2, y)
        rng = np.random.default_rng(3)
        n = len(y)
        bs = []
        for _ in range(1500):
            i = rng.integers(0, n, n)
            A = np.column_stack([np.ones(n), X2[i]])
            if np.linalg.matrix_rank(A) == 3:
                bs.append(np.linalg.lstsq(A, y[i], rcond=None)[0][2])
        lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
        mark = "含0" if lo * hi <= 0 else "不含0"
        print(f"   {lab:>16} {n:>5} {b1[1]:>+13.3f} {r2_1:>11.4f} {b2[2]:>+10.3f} "
              f"{r2_2-r2_1:>+8.4f} {'['+f'{lo:+.3f},{hi:+.3f}'+']':>20}  {mark}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--breadth", default=str(BREADTH_MERGED))
    p.add_argument("--ma", type=int, default=30)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--index", default="中证1000", help="指数名 (见 common.INDEX_NAMES)")
    p.add_argument("--n-perm", type=int, default=2000)
    args = p.parse_args()

    br = load_breadth(args.breadth)
    idx = align_to_breadth(load_index_closes(), br.index)
    if args.index not in idx.columns:
        raise SystemExit(f"unknown index {args.index}; have {list(idx.columns)}")

    s = br[f"ma{args.ma}"]
    ser = idx[args.index]
    h = args.horizon

    print("=" * 96)
    print(f"扩大时间范围检查 — {args.index}, MA{args.ma}, 前瞻 {h} 日")
    print(f"breadth 覆盖 {br.index.min()} .. {br.index.max()} ({len(br)} 交易日)")
    if is_backfilled(args.index, br.index.min()):
        pub = INDEX_PUBLISHED[[k for k, v in INDEX_NAMES.items() if v == args.index][0]]
        span = ser.loc[:pub].notna().sum()
        print(f"⚠️  {args.index} 实际发布于 {pub}；再往前 {span} 个交易日 "
              f"({span/len(ser):.0%} 的样本) 是基日回溯数据，成分股带事后选择偏差，"
              f"该段的胜率天然偏乐观")
    print("=" * 96)

    print("\nP. 分期对照总表\n")
    periods_table(s, ser, h, THRESHOLDS, args.n_perm)

    print("=" * 96)
    print(f"Q. 2005-2009 独立事件清单 (MA{args.ma} < 10%)")
    print("=" * 96)
    event_listing(s, ser, 0.10, h, "20050101", "20091231")

    print()
    print("=" * 96)
    print("R. 分期增量检验: 控制「过去20日指数收益」后, 阳谱 是否还有独立信息?")
    print("=" * 96)
    incremental_by_period(s, ser, h)

    print()
    print("=" * 96)
    print("S. 结论对照")
    print("=" * 96)
    for th in (0.10, 0.20):
        row = []
        for lab, a, b in (("2010-2026", "20100101", "20261231"),
                          ("2005-2026", "20050101", "20261231")):
            m = (s.index >= a) & (s.index <= b)
            fwd = forward_return(ser, h)
            base = fwd[m].dropna()
            ep = episodes(s[m] < th)
            cond = fwd.reindex(ep).dropna()
            row.append(f"{lab}: P涨={ (cond>0).mean():.1%} (基准{ (base>0).mean():.1%}, "
                       f"n={len(cond)})")
        print(f"  阈值<{th:.0%}  " + "   |   ".join(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
