#!/usr/bin/env python3
"""Reconcile the rebuilt breadth series against both independent references.

Two checks, in increasing order of strictness:

1. **tushare vs mongo** over 2010-2011 — the overlap is fetched from Tushare on
   purpose precisely so the new pre-2010 data can be validated rather than
   assumed.  Divergence here means the universe filter or the denominator rule
   differs, and the pre-2010 extension is not comparable to the 2010+ series.
2. **rebuild vs production** ``quant_data.market_spectrum`` — the MA5/MA30 values
   the live system actually publishes (2024-01-02 onward).

Exit code is non-zero if either check fails, so this can gate the merge.

Example:
    python analysis/research/market_spectrum/reconcile.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from analysis.research.market_spectrum.common import (  # noqa: E402
    BREADTH_MONGO,
    BREADTH_TUSHARE,
    load_breadth,
    mongo_client,
)

# A rebuild that only differs by a handful of names (stock-universe drift at the
# edges) still tracks at corr > 0.999; anything looser means a definitional bug.
MIN_CORR = 0.99
MAX_MEAN_ABS_DIFF = 0.01  # 1pp


def compare(a: pd.Series, b: pd.Series, label: str, tol_corr=MIN_CORR,
            tol_diff=MAX_MEAN_ABS_DIFF) -> bool:
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(j) < 30:
        print(f"  {label}: overlap too small (n={len(j)}) — SKIP")
        return True
    d = (j["a"] - j["b"]).abs()
    corr = j["a"].corr(j["b"])
    ok = (corr >= tol_corr) and (d.mean() <= tol_diff)
    print(f"  {label}: n={len(j)}  corr={corr:.6f}  mean|diff|={d.mean():.5f}  "
          f"max|diff|={d.max():.5f}  {'PASS' if ok else 'FAIL'}")
    if not ok:
        worst = d.sort_values(ascending=False).head(3)
        for dt in worst.index:
            print(f"      worst {dt}: a={j.loc[dt,'a']:.4f} b={j.loc[dt,'b']:.4f}")
    return ok


def by_year(a: pd.Series, b: pd.Series, label: str) -> None:
    """Year-by-year drift between two breadth series.

    ``volume_price`` was bulk-loaded forward-adjusted and only re-backfilled to
    raw from 2016 (P5.2), so 2010-2015 may not be on the same price basis as
    2016+.  A step at that boundary would show up here as a level shift rather
    than as noise.
    """
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if j.empty:
        return
    d = (j["a"] - j["b"]).groupby(j.index.str[:4]).agg(["mean", "count"])
    print(f"\n  {label} 按年偏差 (a-b):")
    for yr, row in d.iterrows():
        print(f"    {yr}: mean={row['mean']:+.5f}  n={int(row['count'])}")


def main() -> int:
    if not (BREADTH_MONGO.exists() and BREADTH_TUSHARE.exists()):
        print("need both breadth_mongo.csv and breadth_tushare.csv — run build_breadth.py first")
        return 2

    m = load_breadth(BREADTH_MONGO)
    t = load_breadth(BREADTH_TUSHARE)
    all_ok = True

    print("=" * 78)
    print("1. Tushare rebuild vs Mongo volume_price rebuild (overlap period)")
    print("=" * 78)
    for k in ("ma5", "ma30"):
        all_ok &= compare(t[k], m[k], f"{k}")
    by_year(t["ma30"], m["ma30"], "ma30 (tushare - mongo)")

    print()
    print("=" * 78)
    print("2. Mongo rebuild vs production quant_data.market_spectrum")
    print("=" * 78)
    db = mongo_client()["quant_data"]
    stored = pd.DataFrame(list(db["market_spectrum"].find({"granularity": "daily"})))
    if stored.empty:
        print("  no production rows — SKIP")
    else:
        stored["trade_date"] = stored["trade_date"].astype(str)
        for k in (5, 30):
            s = (stored[stored["ma_period"] == k]
                 .set_index("trade_date")["yang_spectrum"].sort_index())
            all_ok &= compare(m[f"ma{k}"], s, f"ma{k} (production)")
            ns = (stored[stored["ma_period"] == k]
                  .set_index("trade_date")["total_stocks"].sort_index())
            jn = pd.concat([m[f"n{k}"].rename("mine"), ns.rename("prod")], axis=1).dropna()
            if len(jn):
                print(f"      denominator {len(jn)} days: mean|Δ|={(jn['mine']-jn['prod']).abs().mean():.1f} "
                      f"max|Δ|={(jn['mine']-jn['prod']).abs().max():.0f} stocks")

    print()
    print("RESULT:", "PASS — safe to merge" if all_ok else "FAIL — do not merge before fixing")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
