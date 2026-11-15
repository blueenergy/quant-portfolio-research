"""Index benchmark price loading shared across backtests and reports.

Sweeps, reports and research scripts all need the benchmark series to
compare a portfolio against, so this lives in ``core`` rather than
inside any one of them.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


def load_index_benchmark_prices(db_manager, symbol: str, start_date: Optional[str]) -> List[Dict[str, Any]]:
    """Return the benchmark price rows for ``symbol`` from ``start_date`` on.

    Falls back through three sources because research data access, the
    scoring database and a bare Mongo handle each expose the series
    differently; an empty list means none of them had it.
    """
    start = start_date or "00000000"
    symbol_text = str(symbol or "").strip()
    if symbol_text.upper().endswith(".SI"):
        sw_getter = getattr(db_manager, "get_sw_index_prices", None)
        if sw_getter is not None:
            rows = sw_getter(symbol_text, start_date=start, end_date="99991231", freq="daily")
            if rows:
                return rows
        data_db = getattr(db_manager, "data_db", None)
        if data_db is not None:
            try:
                rows = list(
                    data_db["sw_daily"].find(
                        {"ts_code": symbol_text, "trade_date": {"$gte": start}},
                        {"_id": 0},
                    ).sort("trade_date", 1)
                )
                if rows:
                    return rows
            except Exception:
                log.warning("sw_daily lookup failed for benchmark %s", symbol_text, exc_info=True)

    getter = getattr(db_manager, "get_index_benchmark_prices", None)
    if getter is not None:
        return getter(symbol, start_date)
    try:
        rows = db_manager.get_volume_price_data(symbol, start_date=start, end_date="99991231")
        if rows:
            return rows
    except Exception:
        log.warning("volume_price lookup failed for benchmark %s; trying index_prices", symbol, exc_info=True)

    data_db = getattr(db_manager, "data_db", None)
    if data_db is None:
        log.warning("no benchmark series for %s: caller has no data_db to fall back to", symbol)
        return []

    try:
        coll = data_db["index_prices"]
        rows = list(coll.find({"ts_code": symbol, "trade_date": {"$gte": start}}, {"_id": 0}).sort("trade_date", 1))
    except Exception:
        log.warning("index_prices lookup failed for benchmark %s", symbol, exc_info=True)
        return []
    if not rows:
        # an empty benchmark silently removes the comparison a backtest is judged against
        log.warning("benchmark %s resolved to an empty series from %s", symbol, start)
    return rows
