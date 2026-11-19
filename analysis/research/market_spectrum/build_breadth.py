#!/usr/bin/env python3
"""Build the 阴阳谱 (breadth) series from raw daily bars.

Breadth = share of tradable A-shares whose close is above their own N-day MA.
Denominator rule matches production: a stock counts on a date only if it has
both a close and a volume.  Keys are ``(trade_date, ma_period)``.

Three sources:

``mongo``    quant_data.volume_price — production-aligned, but starts 2010-01-04.
``tushare``  local cache from fetch_daily_range.py — raw Tushare, covers 2005+.
``merged``   mongo where it exists, tushare before that.  Run reconcile.py first;
             the join is only safe if the overlap agrees.

Example:
    python analysis/research/market_spectrum/build_breadth.py --source merged
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from analysis.research.market_spectrum.common import (  # noqa: E402
    BREADTH_MONGO,
    BREADTH_MERGED,
    BREADTH_TUSHARE,
    DAILY_RANGE,
    mongo_client,
)

MA_PERIODS = (5, 30)


def breadth_from_panel(px: pd.DataFrame, ok: pd.DataFrame, periods=MA_PERIODS) -> pd.DataFrame:
    """Breadth from a (date x symbol) close panel and a tradable mask."""
    out = {}
    for k in periods:
        ma = px.rolling(k, min_periods=k).mean()
        valid = ma.notna() & ok
        n_valid = valid.sum(axis=1)
        n_above = (valid & (px > ma)).sum(axis=1)
        out[f"ma{k}"] = n_above / n_valid.replace(0, np.nan)
        out[f"n{k}"] = n_valid
    res = pd.DataFrame(out)
    res.index.name = "trade_date"
    return res


def panel_from_records(df: pd.DataFrame, close_col: str, vol_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.drop_duplicates(["symbol", "trade_date"])
    df["close"] = pd.to_numeric(df[close_col], errors="coerce")
    df["vol"] = pd.to_numeric(df[vol_col], errors="coerce")
    ok = (df["close"].notna() & df["vol"].notna())
    px = df.pivot(index="trade_date", columns="symbol", values="close").sort_index()
    mask = df.assign(ok=ok).pivot(index="trade_date", columns="symbol", values="ok").sort_index().fillna(False)
    return px, mask


def build_mongo(verbose: bool = True) -> pd.DataFrame:
    db = mongo_client()["quant_data"]
    cur = db["volume_price"].find({}, {"symbol": 1, "trade_date": 1, "close": 1, "volume": 1, "_id": 0})
    df = pd.DataFrame(list(cur))
    if verbose:
        print(f"[mongo] loaded {len(df):,} rows from volume_price", flush=True)
    px, mask = panel_from_records(df, "close", "volume")
    if verbose:
        print(f"[mongo] panel {px.shape}", flush=True)
    return breadth_from_panel(px, mask)


def build_tushare(verbose: bool = True) -> pd.DataFrame:
    if not DAILY_RANGE.exists():
        raise SystemExit(f"{DAILY_RANGE} missing — run fetch_daily_range.py first")
    df = pd.read_csv(DAILY_RANGE, dtype={"trade_date": str, "ts_code": str})
    df = df.rename(columns={"ts_code": "symbol"})
    if verbose:
        print(f"[tushare] loaded {len(df):,} rows from {DAILY_RANGE.name}", flush=True)
    px, mask = panel_from_records(df, "close", "vol")
    if verbose:
        print(f"[tushare] panel {px.shape}", flush=True)
    return breadth_from_panel(px, mask)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source", choices=("mongo", "tushare", "merged"), default="merged")
    p.add_argument("--out", default=None)
    p.add_argument("--start", default=None, help="Trim output to >= YYYYMMDD")
    p.add_argument("--end", default=None, help="Trim output to <= YYYYMMDD")
    args = p.parse_args()

    if args.source == "mongo":
        res, out = build_mongo(), Path(args.out or BREADTH_MONGO)
    elif args.source == "tushare":
        res, out = build_tushare(), Path(args.out or BREADTH_TUSHARE)
    else:
        m = build_mongo()
        t = build_tushare()
        first_mongo = m.index.min()
        # Mongo wins from its first day onward so the headline series is the
        # production-aligned one; Tushare supplies the pre-history only.
        merged = pd.concat([t[t.index < first_mongo], m]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        overlap = t.index.intersection(m.index)
        res = merged
        out = Path(args.out or BREADTH_MERGED)
        print(f"[merged] tushare < {first_mongo}, mongo >= {first_mongo}; "
              f"overlap {len(overlap)} days kept from mongo")

    res = res[(res["ma30"].notna())]
    if args.start:
        res = res[res.index >= args.start]
    if args.end:
        res = res[res.index <= args.end]

    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out)
    print(f"[done] wrote {out}  rows={len(res):,}  "
          f"range {res.index.min()} .. {res.index.max()}")
    print(res.tail(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
