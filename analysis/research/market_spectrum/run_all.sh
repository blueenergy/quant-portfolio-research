#!/usr/bin/env bash
# Reproduce the whole 阴阳谱 study from scratch.
#
# Needs a python with pymongo/pandas/tushare.  The data-engine venv has all three;
# the host interpreter does not.
#
# Step 1 (fetch) is the long pole (~10 min) and is skipped if the cache exists.
# Reconcile gates the merge: if the Tushare overlap does not agree with the Mongo
# rebuild, the pre-2010 extension is not comparable and the run stops.
set -euo pipefail

PY="${PY:-/home/shuyolin/trading/quant-data-engine/.venv/bin/python}"
cd "$(dirname "$0")"

echo "== 1/7 fetch pre-2010 daily (skipped if cached) =="
if [ ! -f data/daily_range.csv.gz ]; then
  "$PY" fetch_daily_range.py --start 20041101 --end 20111231
else
  echo "   data/daily_range.csv.gz exists — skip (delete to refetch)"
fi

echo "== 2/7 build breadth from mongo =="
"$PY" build_breadth.py --source mongo

echo "== 3/7 build breadth from tushare =="
"$PY" build_breadth.py --source tushare

echo "== 4/7 reconcile (gates the merge) =="
"$PY" reconcile.py

echo "== 5/7 merged series =="
"$PY" build_breadth.py --source merged

echo "== 6/7 analysis =="
"$PY" analyze.py       --start 20050101
"$PY" event_study.py   --threshold 0.20
"$PY" deep.py          --start 20050101
"$PY" robust.py        --start 20050101
"$PY" decile.py        --start 20050101
"$PY" incremental.py   --start 20050101

echo "== 7/7 extended-range check =="
"$PY" extended_check.py
