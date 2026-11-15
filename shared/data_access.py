"""Read-only data access facade for portfolio research.

Built on ``data-access-lib`` (market/SW/index constituents) plus direct Mongo
reads for the legacy research tables. This replaces the old facade that
constructed ``DatabaseManager`` from ``stock-scoring-system``: research no
longer imports scorer code, Tushare, or the scoring write paths.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

from stock_data_access.index_constituents import (
    latest_snapshot_symbols,
    resolve_snapshot_keys,
    snapshot_symbols_for_keys,
)
from stock_data_access.market import MarketDataAccess
from stock_data_access.mongo_context import get_db
from stock_data_access.sw import SwDataAccess

log = logging.getLogger(__name__)

_APP_DB_ENV = "MONGO_DB"
_APP_DB_DEFAULT = "finance"


class ResearchDataAccess:
    """Thin read facade over Mongo + ``data-access-lib`` for research paths."""

    def __init__(self, data_db=None, app_db=None) -> None:
        self.data_db = data_db if data_db is not None else get_db()
        self._app_db = app_db if app_db is not None else self._derive_app_db()

        self.stock_scores = self._app_db["stock_scores"]
        self.entropy_scores = self._app_db["entropy_scores"]
        self.money_flow = self.data_db["money_flow"]
        self.index_constituents = self.data_db["index_constituents"]
        self.sw_index_member = self.data_db["sw_index_member"]
        self.sw_index_classify = self.data_db["sw_index_classify"]
        self.sw_daily = self.data_db["sw_daily"]

        self._market = MarketDataAccess(self.data_db)
        self._sw = SwDataAccess(self.data_db)
        # Snapshot lookup is a find_one; member reads scan a whole snapshot.
        # Backtests ask per day, so cache the member list per snapshot key.
        self._index_snapshot_symbol_cache: Dict[tuple, tuple] = {}

    def _derive_app_db(self):
        client = getattr(self.data_db, "client", None)
        if client is not None:
            return client[os.getenv(_APP_DB_ENV, _APP_DB_DEFAULT)]
        log.warning(
            "data_db has no .client; using it as the app DB too (stock_scores may be empty)"
        )
        return self.data_db

    # ------------------------------------------------------------- market data
    def get_volume_price_data(
        self,
        symbol: str,
        days: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: str = "hfq",
        projection: Optional[dict] = None,
    ) -> List[Dict[str, Any]]:
        return self._market.get_volume_price_data(
            symbol,
            days=days,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
            projection=projection,
        )

    def get_money_flow_data(
        self,
        symbol: str,
        days: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        try:
            coll = self.money_flow
            if start_date and end_date:
                return list(
                    coll.find(
                        {"symbol": symbol, "trade_date": {"$gte": start_date, "$lte": end_date}}
                    ).sort("trade_date", 1)
                )
            if days is not None:
                return list(coll.find({"symbol": symbol}).sort("trade_date", 1).limit(int(days)))
            return list(coll.find({"symbol": symbol}).sort("trade_date", 1).limit(30))
        except Exception as exc:
            log.warning("money_flow read failed for %s: %s", symbol, exc, exc_info=True)
            return []

    def get_sw_index_prices(
        self,
        sw_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        freq: str = "daily",
    ) -> List[Dict[str, Any]]:
        return self._sw.get_sw_index_prices(
            sw_code, start_date=start_date, end_date=end_date, freq=freq
        )

    # --------------------------------------------------------- index membership
    def get_latest_index_constituent_symbols(self, index_code: str) -> List[str]:
        try:
            return latest_snapshot_symbols(self.index_constituents, index_code)
        except Exception as exc:
            log.warning("latest constituents failed for %s: %s", index_code, exc, exc_info=True)
            return []

    def get_index_constituent_symbols_asof(self, index_code: str, as_of_date: str) -> List[str]:
        try:
            weight_trade_date, update_date = resolve_snapshot_keys(
                self.index_constituents, index_code, as_of_date=as_of_date
            )
            if weight_trade_date is None and update_date is None:
                return []
            cache_key = (index_code, weight_trade_date, update_date)
            cached = self._index_snapshot_symbol_cache.get(cache_key)
            if cached is None:
                cached = tuple(
                    snapshot_symbols_for_keys(
                        self.index_constituents,
                        index_code,
                        weight_trade_date=weight_trade_date,
                        update_date=update_date,
                    )
                )
                self._index_snapshot_symbol_cache[cache_key] = cached
            return list(cached)
        except Exception as exc:
            log.warning(
                "as-of constituents failed for %s @ %s: %s",
                index_code,
                as_of_date,
                exc,
                exc_info=True,
            )
            return []

    # -------------------------------------------------------------- guardrails
    _WRITE_METHODS = frozenset({"save_stock_score", "create_indexes"})

    def __getattr__(self, name: str) -> Any:
        if name in type(self)._WRITE_METHODS:
            raise AttributeError(f"write path {name!r} is not available on ResearchDataAccess")
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")


def open_research_data_access(data_db=None, app_db=None) -> ResearchDataAccess:
    """Factory for the shared research read facade."""
    return ResearchDataAccess(data_db, app_db=app_db)


def open_research_data_access_for_cli(
    *,
    enable_tushare: bool = False,
    data_db=None,
    app_db=None,
) -> ResearchDataAccess:
    """Open research read access for CLI tools.

    ``enable_tushare`` is kept for signature compatibility; financial reads are
    DB-only now, so no Tushare initialization happens either way.
    """
    if not enable_tushare:
        os.environ.pop("TUSHARE_TOKEN", None)
    return open_research_data_access(data_db, app_db=app_db)
