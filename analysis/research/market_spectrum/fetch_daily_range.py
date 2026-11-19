#!/usr/bin/env python3
"""Fetch raw daily OHLCV for a date range from Tushare into a local research cache.

Why this exists: ``quant_data.volume_price`` starts 2010-01-04, so any breadth
(阴阳谱) study is silently truncated at 16 years and never sees 2008.  This pulls
the missing history straight from Tushare **into a local file** — it does not
write to any production collection.

The default range deliberately overlaps the Mongo period (2010-2011) so the new
data can be reconciled against the stored series before it is trusted.

Universe filter mirrors ``volume_price``: A-shares only, no B shares (900xxx.SH /
200xxx.SZ).  Suspended names simply have no row on a given date, which matches
the production "close and volume both present" denominator rule.

Example:
    python analysis/research/market_spectrum/fetch_daily_range.py \
        --start 20041101 --end 20111231
"""
from __future__ import annotations

import argparse
import gzip
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
PARTIAL = DATA_DIR / "daily_range_partial.csv.gz"
FINAL = DATA_DIR / "daily_range.csv.gz"

# A-share ts_code shapes seen in quant_data.volume_price.  B shares are excluded
# there, so exclude them here too or the denominator silently changes.
A_SHARE_RE = re.compile(r"^(000|001|002|003|300|301)\d{3}\.SZ$|^(600|601|603|605|688)\d{3}\.SH$")

KEEP_COLS = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"]


def read_token(env_file: str) -> str:
    """Pull TUSHARE_TOKEN out of an env file without evaluating the whole file."""
    tok = None
    for line in open(env_file, encoding="utf-8", errors="ignore"):
        m = re.match(r"\s*TUSHARE_TOKEN\s*=\s*(\S+)", line)
        if m:
            tok = m.group(1).strip().strip('"').strip("'")
    if not tok or tok.startswith("your_"):
        raise SystemExit(f"no usable TUSHARE_TOKEN in {env_file}")
    return tok


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--start", default="20041101", help="Start date YYYYMMDD (default 20041101, gives MA30 warm-up before 2005-01-04)")
    p.add_argument("--end", default="20111231", help="End date YYYYMMDD")
    p.add_argument("--env-file", default="/home/shuyolin/trading/quant-infrastructure/apps/env/quant-data-engine.env")
    p.add_argument("--rate", type=float, default=20.0, help="Max Tushare calls per second")
    p.add_argument("--flush-every", type=int, default=100, help="Write partial cache every N trading days")
    p.add_argument("--out", default=str(FINAL))
    p.add_argument("--fresh", action="store_true", help="Ignore existing partial cache and refetch everything")
    return p.parse_args()


def load_partial(path: Path, fresh: bool) -> pd.DataFrame:
    if fresh or not path.exists():
        return pd.DataFrame(columns=KEEP_COLS)
    df = pd.read_csv(path, dtype={"trade_date": str, "ts_code": str})
    print(f"[cache] resumed from {path.name}: {len(df):,} rows, {df['trade_date'].nunique()} days", flush=True)
    return df


def main() -> int:
    args = parse_args()
    import tushare as ts

    pro = ts.pro_api(read_token(args.env_file))

    cal = pro.trade_cal(exchange="SSE", start_date=args.start, end_date=args.end,
                        fields="cal_date,is_open")
    days = sorted(cal[cal["is_open"] == 1]["cal_date"].tolist())
    if not days:
        raise SystemExit("no trading days in range")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    acc = load_partial(Path(args.out), args.fresh)
    done = set(acc["trade_date"].unique()) if len(acc) else set()
    todo = [d for d in days if d not in done]
    print(f"[plan] {len(days)} trading days in {args.start}..{args.end}; "
          f"{len(done)} cached; {len(todo)} to fetch", flush=True)

    min_interval = 1.0 / args.rate
    t0 = time.time()
    frames = [acc] if len(acc) else []
    for i, d in enumerate(todo, 1):
        tick = time.time()
        for attempt in range(5):
            try:
                df = pro.daily(trade_date=d)
                break
            except Exception as e:  # noqa: BLE001 - tushare raises bare Exception on quota
                wait = 2 ** attempt
                print(f"[warn] {d} attempt {attempt+1} failed: {e}; sleep {wait}s", flush=True)
                time.sleep(wait)
        else:
            print(f"[error] {d} gave up after 5 attempts", flush=True)
            return 1

        if df is not None and len(df):
            df = df[[c for c in KEEP_COLS if c in df.columns]].copy()
            df = df[df["ts_code"].str.match(A_SHARE_RE)]
            frames.append(df)

        if i % args.flush_every == 0 or i == len(todo):
            merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=KEEP_COLS)
            merged = merged.drop_duplicates(["trade_date", "ts_code"]).sort_values(["trade_date", "ts_code"])
            merged.to_csv(args.out, index=False, compression="gzip")
            rate = i / max(time.time() - t0, 1e-9)
            eta = (len(todo) - i) / max(rate, 1e-9)
            print(f"[{i:>5}/{len(todo)}] up to {d}  rows={len(merged):,}  "
                  f"{rate:.1f} day/s  eta {eta/60:.1f} min", flush=True)

        elapsed = time.time() - tick
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    out = Path(args.out)
    final = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=KEEP_COLS)
    final = final.drop_duplicates(["trade_date", "ts_code"]).sort_values(["trade_date", "ts_code"])
    final.to_csv(out, index=False, compression="gzip")
    print(f"[done] wrote {out}  rows={len(final):,}  days={final['trade_date'].nunique()}  "
          f"symbols={final['ts_code'].nunique()}", flush=True)
    print(f"       range {final['trade_date'].min()} .. {final['trade_date'].max()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
