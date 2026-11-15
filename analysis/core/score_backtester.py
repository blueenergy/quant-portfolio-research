#!/usr/bin/env python3
"""Backtest utilities for validating stock score predictive power.

This module intentionally does not change scoring logic. It reads historical
score records, joins future prices, and reports factor diagnostics such as
forward returns, rank IC, quantile spreads, and daily long top-K equal-weight
stats (``topk_summary``). Use ``--export-dataset PATH`` to persist the full
cross-sectional panel (Parquet if pyarrow is installed, otherwise CSV).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from array import array
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from itertools import chain
from math import nan
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from stock_data_access.index_constituents import asof_snapshot_symbols, latest_snapshot_symbols

from analysis.core.trailing_stop import normalize_trailing_stop_ratio

log = logging.getLogger(__name__)

LIVE_CASH_SUFFIX = "live_cash"

# Calendar-day lookback for volatility features (~90 trading days + ATR buffer).
VOLATILITY_LOOKBACK_CALENDAR_DAYS = 140


def live_cash_field(name: str, horizon: int) -> str:
    """Column used when research cash overlay flattens mid-holding like live."""
    return f"{name}_{int(horizon)}d_{LIVE_CASH_SUFFIX}"


# Minimal projection for bulk price preload: only the raw OHLC(+volume) fields the
# adjustment/volatility pipeline consumes. Dropping derived/display columns (MA*,
# Price_Volatility_20D, stock_name, update_time, ...) roughly halves the per-symbol
# network transfer + BSON decode cost, which dominates build_dataset time against a
# remote Mongo. symbol/trade_date are required by dedupe and the adj-factor merge.
_VOLUME_PRICE_PROJECTION = {
    "_id": 0,
    "symbol": 1,
    "trade_date": 1,
    "date": 1,
    "open": 1,
    "open_price": 1,
    "adj_open": 1,
    "high": 1,
    "high_price": 1,
    "adj_high": 1,
    "low": 1,
    "low_price": 1,
    "adj_low": 1,
    "close": 1,
    "close_price": 1,
    "adj_close": 1,
    "pre_close": 1,
    "volume": 1,
    "vol": 1,
}

# Default worker count for the parallel price preload. Each symbol is an
# independent remote read whose cost is dominated by network transfer + BSON
# decode, so concurrent fetches (pymongo releases the GIL during socket I/O)
# cut build_dataset time substantially. Set to 1 to fall back to sequential.
_DEFAULT_PRELOAD_WORKERS = 8


def _preload_worker_count() -> int:
    raw = os.getenv("PORTFOLIO_RESEARCH_PRELOAD_WORKERS")
    if raw is None or raw == "":
        return _DEFAULT_PRELOAD_WORKERS
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_PRELOAD_WORKERS

DEFAULT_SCORE_COLUMNS = (
    "composite_score",
    "composite_balanced_score",
    "composite_aggressive_score",
    "composite_conservative_score",
    "composite_defensive_score",
    "composite_value_oriented_score",
    "composite_trading_oriented_score",
    "composite_growth_oriented_score",
    "composite_cycle_oriented_score",
    "composite_growth_cycle_score",
    "technical_score",
    "fundamental_score",
    "value_score",
    "growth_score",
    "money_flow_score",
    "cycle_score",
)

# Columnar form of the score panel. ``prepare_score_panel`` used to return the raw
# score documents as a list of dicts; for a csi1000 panel over ~11 years that is
# 2.8M dicts and measured ~9 GiB resident, which is what pushed the portfolio
# research worker past its container memory limit. Flattening each document once
# into these columns costs a few hundred MB instead, and the flattening is then
# paid once rather than once per trailing-stop variant.
#
# ``symbol`` and ``score_date`` are the key columns; ``score_date`` stays a plain
# string because it is both a group key and compared with ``<``. ``name`` and
# ``industry`` are carried through to the dataset, so they are part of the
# contract too rather than something a consumer patches in on arrival. The score
# columns that follow are discovered from ``_score_row_from_doc``, which is a
# pure function of the document, so subclasses projecting their own fields
# (entropy, alpha factors) need no registry here.
_PANEL_KEY_COLUMNS = ("symbol", "score_date")
_PANEL_DISPLAY_COLUMNS = ("name", "industry")
_PANEL_REQUIRED_COLUMNS = _PANEL_KEY_COLUMNS + _PANEL_DISPLAY_COLUMNS


def _is_numeric(value: Any) -> bool:
    """Whether a row value belongs in a C double buffer rather than a list.

    Only genuine floats qualify. An ``int`` stays on the list path so that a
    column present on every row keeps the int64 dtype ``pd.DataFrame(rows)``
    gave it; the ragged numeric columns that dominate the frame are floats, or
    become float through the NaN filled in for rows that lack the key.

    The kind is decided from the value on the row that first carries the key, so
    it is only meaningful because every writer omits the keys it has no value
    for rather than writing ``None`` -- `_fill_horizon_exit` included. A writer
    that emitted an explicit ``None`` would send a float column down the boxing
    path for the whole build.
    """
    if isinstance(value, bool) or isinstance(value, np.bool_):
        return False
    return isinstance(value, (float, np.floating))


def _ensure_score_panel(panel: Any) -> pd.DataFrame:
    """Fail loudly on the pre-refactor list-of-dicts form.

    Accepting it silently would rebuild the very allocation this refactor
    removed: whoever passed it would keep a multi-GB list alive for no reason.
    """
    if not isinstance(panel, pd.DataFrame):
        raise TypeError(
            f"score panel must be a DataFrame, got {type(panel).__name__}. "
            "Build it with ScoreBacktester.prepare_score_panel() or _panel_from_records()."
        )
    missing = [column for column in _PANEL_REQUIRED_COLUMNS if column not in panel.columns]
    if missing:
        raise ValueError(f"score panel is missing required columns: {', '.join(missing)}")
    return panel


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for score validation."""

    horizons: Sequence[int] = (1, 5, 10, 20)
    quantiles: int = 5
    score_columns: Sequence[str] = DEFAULT_SCORE_COLUMNS
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    index_codes: Sequence[str] = ()
    universe_index_codes: Sequence[str] = ()
    min_names_per_date: int = 5
    execution_price: str = "next_open"
    exit_limit_delay_days: int = 5
    # Close <= peak_high * (1 - pct) within the horizon window triggers exit next open.
    trailing_stop_pct: Optional[float] = None
    # Daily long top-K equal-weight vs universe; empty tuple skips top-K stats.
    top_k_sizes: Sequence[int] = (10, 20, 50)


@dataclass
class ScoreBacktestResult:
    """Container for all score validation outputs."""

    dataset: pd.DataFrame
    rank_ic: pd.DataFrame = field(default_factory=pd.DataFrame)
    quantile_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    topk_summary: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self, include_dataset: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "rows": int(len(self.dataset)),
            "rank_ic": self.rank_ic.to_dict(orient="records"),
            "quantile_returns": self.quantile_returns.to_dict(orient="records"),
            "summary": self.summary.to_dict(orient="records"),
            "topk_summary": self.topk_summary.to_dict(orient="records"),
        }
        if include_dataset and not self.dataset.empty:
            out["dataset"] = self.dataset.to_dict(orient="records")
        return out


class ScoreBacktester:
    """Validate stock score records against future price returns."""

    def __init__(self, db_manager, config: Optional[BacktestConfig] = None):
        self.db_manager = db_manager
        self.config = config or BacktestConfig()
        self._price_cache: Dict[str, List[Dict[str, Any]]] = {}
        # symbol -> (history object, its ascending trade dates). Identity-keyed so a
        # replaced price cache rebuilds instead of answering from the stale index.
        self._price_date_index: Dict[
            str, Tuple[Sequence[Dict[str, Any]], Optional[List[str]]]
        ] = {}
        # Cache as-of universe symbol sets so a multi-year daily backtest resolves
        # each (index_code, score_date) snapshot at most once.
        self._universe_asof_cache: Dict[Tuple[str, str], Set[str]] = {}
        # One warning per index is enough; a daily backtest would otherwise emit
        # the same degradation notice several hundred times.
        self._survivorship_warned: Set[str] = set()

    @property
    def price_cache(self) -> Dict[str, List[Dict[str, Any]]]:
        """Symbol -> full price history loaded during ``build_dataset()``."""
        return self._price_cache

    def run(self) -> ScoreBacktestResult:
        dataset = self.build_dataset()
        rank_ic = self.calculate_rank_ic(dataset)
        quantile_returns = self.calculate_quantile_returns(dataset)
        summary = self.summarize(rank_ic, quantile_returns)
        topk_summary = self.calculate_topk_summary(dataset)
        return ScoreBacktestResult(
            dataset=dataset,
            rank_ic=rank_ic,
            quantile_returns=quantile_returns,
            summary=summary,
            topk_summary=topk_summary,
        )

    def prepare_score_panel(
        self, *, on_step: Optional[Callable[[str, float], None]] = None
    ) -> pd.DataFrame:
        """Load scores, apply the universe filter, and preload the price cache once.

        Returns the columnar panel (see ``_panel_from_records``) rather than the
        raw score documents: for a csi1000 panel over ~11 years the documents are
        2.8M dicts, which measured ~9 GiB resident and pushed the portfolio
        research worker past its container memory limit.

        ``on_step`` receives ``(step_name, elapsed_seconds)``. The three steps have
        very different cost profiles — one bulk read, one per-date snapshot lookup,
        one fan-out of per-symbol reads — and a long run reports as a single
        multi-minute block without a way to tell which of them is responsible.
        """

        def _step(name: str, fn: Callable[[], Any]) -> Any:
            if on_step is None:
                return fn()
            started = time.perf_counter()
            try:
                return fn()
            finally:
                on_step(name, time.perf_counter() - started)

        panel = _step("load_score_records", lambda: self._panel_from_records(self._load_score_records()))
        panel = _step("filter_records_by_universe", lambda: self._filter_panel_by_universe(panel))
        _step("preload_price_cache", lambda: self._preload_price_cache(panel))
        return panel

    def _panel_from_records(self, records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
        """Flatten score documents into the columnar score panel.

        Each document is flattened exactly once, straight into column lists.
        ``records`` may be a live cursor -- it is consumed in a single pass and
        never indexed -- so the transient cost is one document rather than all of
        them, provided the caller streams (`_load_score_records` does). Handing
        this a materialised list still works but holds every document alongside
        the columns, which is what an earlier version of this refactor did and
        what measured ~9.9 GB peak on a csi1000 panel over ~11 years.

        The score columns are taken from ``_score_row_from_doc``'s own key set,
        which keeps the subclasses that project different fields (entropy, alpha
        factors) working without any change here.

        Documents without a usable symbol or score_date are dropped: every
        consumer that walks the panel skips them anyway.
        """
        columns: Dict[str, List[Any]] = {
            name: [] for name in _PANEL_KEY_COLUMNS + _PANEL_DISPLAY_COLUMNS
        }
        score_columns: List[str] = []
        dropped = 0
        for doc in records:
            symbol = doc.get("symbol")
            score_date = self._normalize_date(doc.get("score_date"))
            if not symbol or not score_date:
                dropped += 1
                continue
            row = self._score_row_from_doc(doc)
            if not score_columns:
                score_columns = list(row)
                columns.update({name: [] for name in score_columns})
            columns["symbol"].append(symbol)
            columns["score_date"].append(score_date)
            columns["name"].append(doc.get("name", ""))
            columns["industry"].append(doc.get("industry", ""))
            for name in score_columns:
                columns[name].append(row.get(name))
        if dropped:
            log.info("score panel: dropped %d document(s) without symbol/score_date", dropped)
        # No explicit dtype: a float column with missing values becomes float64
        # (NaN), which is what `pd.DataFrame(list_of_dicts)` produced before.
        return pd.DataFrame(columns)

    def build_dataset(self) -> pd.DataFrame:
        return self.build_dataset_from_panel(self.prepare_score_panel())

    def build_dataset_from_panel(
        self,
        panel: pd.DataFrame,
        *,
        trailing_stop_pct: Optional[float] = None,
    ) -> pd.DataFrame:
        """Build dataset from a preloaded score panel and price cache (no DB reads)."""
        panel = _ensure_score_panel(panel)
        config = self.config
        if trailing_stop_pct is not None and trailing_stop_pct != self.config.trailing_stop_pct:
            config = replace(self.config, trailing_stop_pct=trailing_stop_pct)

        usable = panel["symbol"].notna() & panel["score_date"].notna()
        if not usable.all():
            panel = panel[usable]

        symbols = panel["symbol"].to_numpy()
        score_dates = panel["score_date"].to_numpy()
        names = panel["name"].to_numpy()
        industries = panel["industry"].to_numpy()
        # Only the columns the config actually reads, as one contiguous block
        # instead of a per-row dict lookup. Absent names are tolerated the same
        # way `_score_row_from_doc` tolerated them: ignored, not an error.
        selected = [column for column in config.score_columns if column in panel.columns]
        if selected:
            # to_numeric first: a score column that is missing for every row
            # comes back as object dtype full of None, which to_numpy(float64)
            # refuses. Coercing gives the float64 NaN those rows mean.
            score_values = np.column_stack(
                [pd.to_numeric(panel[column], errors="coerce").to_numpy(dtype="float64") for column in selected]
            )
            has_score = (~np.isnan(score_values)).any(axis=1)
        else:
            score_values = np.empty((len(panel), 0), dtype="float64")
            has_score = np.zeros(len(panel), dtype=bool)

        # Per-column accumulation, with float columns going into C double buffers
        # instead of Python lists. A list holds boxed floats — 24 bytes each on
        # top of the 8-byte pointer — so on a 2.8M-row frame the boxing alone was
        # several GB, which is exactly the transient that had to go.
        out: Dict[str, Any] = {}
        emitted = 0
        # A key first seen with an explicit None is the one case that can box a
        # whole column without saying so: it takes the list path, and if real
        # floats arrive afterwards every one of them is boxed for the rest of the
        # build. Nothing does this today -- the score columns are pulled from a
        # float64 matrix and `_fill_horizon_exit` writes its keys conditionally --
        # so this set stays empty and costs nothing. It is here to catch the next
        # writer that emits a None instead of omitting the key.
        none_first: Set[str] = set()

        def _emit(row: Dict[str, Any]) -> None:
            nonlocal emitted
            # `_fill_horizon_exit` only writes the keys it has values for, so the
            # key set really does vary by row. A column is appended to exactly
            # once per emitted row, with None where this row lacks the key, the
            # same NaN-filling `pd.DataFrame(list_of_dicts)` did.
            for key, value in row.items():
                if key not in out:
                    if value is None:
                        none_first.add(key)
                    out[key] = array("d", [nan] * emitted) if _is_numeric(value) else [None] * emitted
            for key, column in out.items():
                if isinstance(column, list):
                    column.append(row.get(key))
                else:
                    value = row.get(key)
                    column.append(nan if value is None else value)
            emitted += 1

        max_future_bars = self._max_future_bars_needed(config.trailing_stop_pct)
        for position in range(len(panel)):
            if not has_score[position]:
                continue
            symbol = symbols[position]
            score_date = score_dates[position]
            price_data = self._load_future_prices(symbol, score_date, max_future_bars)
            if not price_data:
                continue
            forward_returns = self._calculate_forward_returns(
                price_data,
                config.horizons,
                score_date,
                trailing_stop_pct=config.trailing_stop_pct,
            )
            if not forward_returns:
                continue
            row: Dict[str, Any] = {
                "symbol": symbol,
                "score_date": score_date,
                "name": names[position],
                "industry": industries[position],
            }
            if selected:
                row.update(zip(selected, score_values[position]))
            row.update(forward_returns)
            _emit(row)

        if not out:
            return pd.DataFrame()

        df = pd.DataFrame(out)
        df["score_date"] = df["score_date"].astype(str)
        # Checked once against the finished dtypes rather than per value: a column
        # that went to the list path and still came out float64 is holding boxed
        # floats, which is the silent cost `_is_numeric`'s first-value rule can
        # incur. An int-first column reaches float64 the same way and is expected
        # to, so only the None-first ones are reported.
        boxed = sorted(key for key in none_first if str(df[key].dtype) == "float64")
        if boxed:
            log.warning(
                "score panel: %s took the boxed-list path because the first row carrying them "
                "had None, then filled with floats. _is_numeric decides a column's kind once, on "
                "first appearance, so every value in these columns is a boxed Python float.",
                ", ".join(boxed),
            )
        return df.sort_values(["score_date", "symbol"]).reset_index(drop=True)

    def add_live_cash_return_columns(
        self,
        dataset: pd.DataFrame,
        cash_dates: Sequence[str],
        *,
        trailing_stop_pct: Optional[float] = None,
    ) -> pd.DataFrame:
        """Add mid-holding cash-exit returns without rebuilding the panel.

        Live plans can flatten on a monitor day when the index label leaves bull.
        Research otherwise keeps the whole horizon return. These extra columns
        take the earlier of trailing-stop and first cash close, then sell next open.
        """
        if dataset is None or dataset.empty:
            return dataset
        cash = frozenset(str(date) for date in (cash_dates or ()) if str(date))
        trailing = (
            trailing_stop_pct if trailing_stop_pct is not None else self.config.trailing_stop_pct
        )
        max_bars = self._max_future_bars_needed(trailing, need_cash_exit=True)
        horizons = tuple(int(value) for value in self.config.horizons)
        extra_rows: List[Dict[str, Any]] = []
        for record in dataset[["symbol", "score_date"]].to_dict(orient="records"):
            symbol = record.get("symbol")
            score_date = self._normalize_date(record.get("score_date"))
            row: Dict[str, Any] = {}
            price_data = (
                self._load_future_prices(symbol, score_date, max_bars)
                if symbol and score_date
                else []
            )
            forward = (
                self._calculate_forward_returns(
                    price_data,
                    horizons,
                    score_date,
                    trailing_stop_pct=trailing,
                    cash_exit_dates=cash,
                )
                if price_data
                else {}
            )
            for horizon in horizons:
                for name in (
                    "forward_return",
                    "exit_price",
                    "exit_date",
                    "exit_price_type",
                    "exit_delayed_days",
                    "exit_reason",
                ):
                    field = live_cash_field(name, horizon)
                    row[field] = forward.get(field)
            extra_rows.append(row)
        extra = pd.DataFrame(extra_rows, index=dataset.index)
        out = dataset.copy()
        for column in extra.columns:
            out[column] = extra[column]
        # Rows that already have a completed horizon return but no cash overlay
        # (missing bars, limit-up skip on rebuild) keep the trailing/horizon path.
        for horizon in horizons:
            live_col = live_cash_field("forward_return", horizon)
            base_col = f"forward_return_{horizon}d"
            if live_col in out.columns and base_col in out.columns:
                out[live_col] = out[live_col].where(out[live_col].notna(), out[base_col])
            for name in ("exit_price", "exit_date", "exit_price_type", "exit_delayed_days", "exit_reason"):
                live_field = live_cash_field(name, horizon)
                base_field = f"{name}_{horizon}d"
                if live_field in out.columns and base_field in out.columns:
                    out[live_field] = out[live_field].where(out[live_field].notna(), out[base_field])
        return out

    def calculate_rank_ic(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if dataset.empty:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        for horizon in self.config.horizons:
            ret_col = self._return_col(horizon)
            if ret_col not in dataset.columns:
                continue
            for score_col in self.config.score_columns:
                if score_col not in dataset.columns:
                    continue
                for score_date, group in dataset.groupby("score_date"):
                    valid = group[[score_col, ret_col]].dropna()
                    if len(valid) < self.config.min_names_per_date or valid[score_col].nunique() < 2 or valid[ret_col].nunique() < 2:
                        continue
                    rows.append(
                        {
                            "score_date": score_date,
                            "score_column": score_col,
                            "horizon": horizon,
                            # Ranks + Pearson is Spearman without pulling in
                            # scipy, which the worker image does not install.
                            "rank_ic": float(valid[score_col].rank().corr(valid[ret_col].rank())),
                            "count": int(len(valid)),
                        }
                    )

        return pd.DataFrame(rows)

    def calculate_quantile_returns(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if dataset.empty:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        for horizon in self.config.horizons:
            ret_col = self._return_col(horizon)
            if ret_col not in dataset.columns:
                continue
            for score_col in self.config.score_columns:
                if score_col not in dataset.columns:
                    continue
                for score_date, group in dataset.groupby("score_date"):
                    valid = group[[score_col, ret_col]].dropna().copy()
                    if len(valid) < max(self.config.quantiles, self.config.min_names_per_date) or valid[score_col].nunique() < 2:
                        continue
                    try:
                        valid["quantile"] = pd.qcut(valid[score_col].rank(method="first"), self.config.quantiles, labels=False) + 1
                    except ValueError:
                        continue
                    grouped = valid.groupby("quantile", as_index=False)[ret_col].agg(["mean", "count"]).reset_index()
                    for _, row in grouped.iterrows():
                        rows.append(
                            {
                                "score_date": score_date,
                                "score_column": score_col,
                                "horizon": horizon,
                                "quantile": int(row["quantile"]),
                                "mean_return": float(row["mean"]),
                                "count": int(row["count"]),
                            }
                        )

        return pd.DataFrame(rows)

    def calculate_topk_summary(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Per trading day: equal-weight mean forward return of top-K names by score.

        If a cross-section has fewer than K valid rows, uses all of them (effective K).
        Also records top-K minus bottom-K spread (same K) and excess vs equal-weight universe.
        """
        ks = tuple(int(k) for k in (self.config.top_k_sizes or ()) if int(k) >= 1)
        if dataset.empty or not ks:
            return pd.DataFrame()

        rows_daily: List[Dict[str, Any]] = []

        for horizon in self.config.horizons:
            ret_col = self._return_col(horizon)
            if ret_col not in dataset.columns:
                continue
            for score_col in self.config.score_columns:
                if score_col not in dataset.columns:
                    continue
                for score_date, group in dataset.groupby("score_date", sort=True):
                    valid = group[[score_col, ret_col]].dropna()
                    if len(valid) < self.config.min_names_per_date:
                        continue
                    if valid[score_col].nunique() < 2 or valid[ret_col].nunique() < 2:
                        continue
                    univ_mean = float(valid[ret_col].mean())
                    sorted_desc = valid.sort_values(score_col, ascending=False)
                    for k in ks:
                        kk = min(int(k), len(sorted_desc))
                        top_mean = float(sorted_desc.head(kk)[ret_col].mean())
                        bottom_mean = float(sorted_desc.tail(kk)[ret_col].mean())
                        rows_daily.append(
                            {
                                "score_date": score_date,
                                "score_column": score_col,
                                "horizon": horizon,
                                "top_k": int(k),
                                "effective_k": int(kk),
                                "top_mean_return": top_mean,
                                "bottom_mean_return": bottom_mean,
                                "universe_mean_return": univ_mean,
                                "top_vs_universe": top_mean - univ_mean,
                                "top_minus_bottom": top_mean - bottom_mean,
                            }
                        )

        daily = pd.DataFrame(rows_daily)
        if daily.empty:
            return pd.DataFrame()

        agg_rows: List[Dict[str, Any]] = []
        for (score_col, horizon, top_k), grp in daily.groupby(["score_column", "horizon", "top_k"]):
            ex = grp["top_vs_universe"]
            tb = grp["top_minus_bottom"]
            ex_std = float(ex.std(ddof=1)) if len(ex) > 1 else None
            tb_std = float(tb.std(ddof=1)) if len(tb) > 1 else None
            agg_rows.append(
                {
                    "score_column": score_col,
                    "horizon": int(horizon),
                    "top_k": int(top_k),
                    "mean_effective_k": float(grp["effective_k"].mean()),
                    "mean_top_return": float(grp["top_mean_return"].mean()),
                    "mean_universe_return": float(grp["universe_mean_return"].mean()),
                    "mean_top_vs_universe": float(ex.mean()),
                    "top_vs_universe_std": ex_std,
                    "top_vs_universe_ir": float(ex.mean() / ex_std) if ex_std and ex_std > 0 else None,
                    "mean_top_minus_bottom": float(tb.mean()),
                    "top_minus_bottom_std": tb_std,
                    "top_minus_bottom_ir": float(tb.mean() / tb_std) if tb_std and tb_std > 0 else None,
                    "observations": int(len(grp)),
                }
            )

        return pd.DataFrame(agg_rows)

    def summarize(self, rank_ic: pd.DataFrame, quantile_returns: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        if not rank_ic.empty:
            for (score_col, horizon), group in rank_ic.groupby(["score_column", "horizon"]):
                ic_mean = group["rank_ic"].mean()
                ic_std = group["rank_ic"].std(ddof=1)
                ic_values = group["rank_ic"].dropna()
                # 胜率：截面 IC 与 ic_mean 同号的交易日占比（>0 看多因子、<0 看反转因子都成立）。
                ic_win_rate = float((ic_values * ic_mean > 0).mean()) if len(ic_values) else None
                rows.append(
                    {
                        "score_column": score_col,
                        "horizon": int(horizon),
                        "rank_ic_mean": float(ic_mean),
                        "rank_ic_std": None if pd.isna(ic_std) else float(ic_std),
                        "rank_ic_ir": None if pd.isna(ic_std) or ic_std == 0 else float(ic_mean / ic_std),
                        "ic_win_rate": ic_win_rate,
                        "ic_observations": int(len(group)),
                        **self._quantile_summary(quantile_returns, score_col, int(horizon)),
                    }
                )

        if not rows and not quantile_returns.empty:
            for (score_col, horizon), _ in quantile_returns.groupby(["score_column", "horizon"]):
                rows.append(
                    {
                        "score_column": score_col,
                        "horizon": int(horizon),
                        "rank_ic_mean": None,
                        "rank_ic_std": None,
                        "rank_ic_ir": None,
                        "ic_win_rate": None,
                        "ic_observations": 0,
                        **self._quantile_summary(quantile_returns, score_col, int(horizon)),
                    }
                )

        return pd.DataFrame(rows)

    def _quantile_summary(self, quantile_returns: pd.DataFrame, score_col: str, horizon: int) -> Dict[str, Any]:
        if quantile_returns.empty:
            return {"top_quantile_return": None, "bottom_quantile_return": None, "top_bottom_spread": None}

        subset = quantile_returns[(quantile_returns["score_column"] == score_col) & (quantile_returns["horizon"] == horizon)]
        if subset.empty:
            return {"top_quantile_return": None, "bottom_quantile_return": None, "top_bottom_spread": None}

        by_q = subset.groupby("quantile")["mean_return"].mean()
        bottom_q = int(by_q.index.min())
        top_q = int(by_q.index.max())
        top_ret = float(by_q.loc[top_q])
        bottom_ret = float(by_q.loc[bottom_q])
        return {
            "top_quantile_return": top_ret,
            "bottom_quantile_return": bottom_ret,
            "top_bottom_spread": top_ret - bottom_ret,
        }

    def _load_score_records(self) -> Iterable[Dict[str, Any]]:
        query: Dict[str, Any] = {}
        if self.config.start_date or self.config.end_date:
            date_query: Dict[str, Any] = {}
            if self.config.start_date:
                date_query["$gte"] = self.config.start_date
            if self.config.end_date:
                date_query["$lte"] = self.config.end_date
            query["score_date"] = date_query

        index_codes = list(self.config.index_codes)
        universe_pushdown = False
        if not index_codes and self.config.universe_index_codes:
            index_codes = list(self.config.universe_index_codes)
            universe_pushdown = True
        if index_codes:
            query["index_codes"] = {"$in": index_codes}

        coll = self.db_manager.stock_scores
        projection = {
            "_id": 0,
            "symbol": 1,
            "score_date": 1,
            "name": 1,
            "industry": 1,
            "index_codes": 1,
            "scores": 1,
            "composite_score": 1,
            "technical_score": 1,
            "fundamental_score": 1,
            "value_score": 1,
            "growth_score": 1,
            "money_flow_score": 1,
            "cycle_score": 1,
        }

        def _fetch_records(fetch_query: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
            """Return the cursor itself rather than a list of its documents.

            Materialising it costs ~8 GB resident on a csi1000 panel over ~11
            years, and `_panel_from_records` would then hold that list for the
            whole flattening -- the persistent cost this refactor removed, but
            not the transient one. Documents are yielded as they arrive instead,
            so the flattening holds one document at a time.

            The cursor is built eagerly so that a collection which rejects the
            projection or the sort still raises here, where the fallback below
            can catch it, rather than mid-iteration.
            """
            try:
                cursor = coll.find(fetch_query, projection).sort("score_date", 1)
                try:
                    cursor = cursor.batch_size(1000)
                except (TypeError, AttributeError):
                    # batch_size is a throughput hint; fakes and older drivers lack it
                    pass
                return iter(cursor)
            except (TypeError, AttributeError):
                return iter(coll.find(fetch_query))

        records = _fetch_records(query)
        # Peek one document: the pushdown fallback needs to know whether the
        # query returned anything, and a cursor has no truthiness of its own.
        first = next(records, None)
        if first is None and universe_pushdown:
            fallback_query = {key: value for key, value in query.items() if key != "index_codes"}
            print(
                "⚠️ stock_scores pushdown by index_codes returned no rows; falling back to date-range query",
                flush=True,
            )
            records = _fetch_records(fallback_query)
            first = next(records, None)
        if first is None:
            return iter(())
        return chain((first,), records)

    def _filter_panel_by_universe(self, panel: pd.DataFrame) -> pd.DataFrame:
        if not self.config.universe_index_codes or panel.empty:
            return panel

        # Positions, not labels: `panel.index` happens to be a RangeIndex here, but
        # a mask built from labels would silently misplace rows if it ever were not.
        grouped = panel.groupby("score_date", sort=True).indices
        if not grouped:
            # `groupby` drops null keys, so a panel whose score_date is null
            # throughout yields no group at all while staying non-empty. Not
            # filtering is the call the missing-snapshot branch below already
            # makes: with no universe to test against, keep everything. An
            # all-False mask would instead return an empty panel.
            return panel

        symbols = panel["symbol"].to_numpy()
        keep = np.zeros(len(panel), dtype=bool)
        _warned_empty_universe: Set[str] = set()
        for score_date, positions in grouped.items():
            universe_symbols = self._load_universe_symbols_asof(self.config.universe_index_codes, score_date)
            if not universe_symbols:
                if score_date not in _warned_empty_universe:
                    _warned_empty_universe.add(score_date)
                    log.warning(
                        "index_constituents snapshot missing for score_date=%s; "
                        "falling back to stock_scores index_codes field",
                        score_date,
                    )
                # Fallback: trust the stock_scores record's own index_codes field,
                # which already passed the MongoDB query filter.
                keep[positions] = True
                continue
            # Assigning rather than OR-ing is safe because a position belongs to
            # exactly one group. `np.isin` wants the set as a sequence: handed the
            # set itself it builds a 0-d object array and matches nothing, quietly.
            keep[positions] = np.isin(symbols[positions], list(universe_symbols))

        # Boolean indexing keeps the input order, so the positions no longer have
        # to be sorted back into it.
        return panel[keep]

    def _load_universe_symbols_asof(self, index_codes: Sequence[str], as_of_date: str) -> Set[str]:
        symbols: Set[str] = set()
        for index_code in index_codes:
            symbols.update(self._load_index_symbols_asof(index_code, as_of_date))
        return symbols

    def _load_index_symbols_asof(self, index_code: str, as_of_date: str) -> Set[str]:
        cache_key = (index_code, as_of_date)
        cached = self._universe_asof_cache.get(cache_key)
        if cached is not None:
            return cached
        symbols = self._resolve_index_symbols_asof(index_code, as_of_date)
        self._universe_asof_cache[cache_key] = symbols
        return symbols

    def _resolve_index_symbols_asof(self, index_code: str, as_of_date: str) -> Set[str]:
        """Index membership as it stood on ``as_of_date``.

        The last resort answers with *today's* membership, which silently turns
        a backtest into a study of the stocks that survived. Every step down
        this ladder is logged so that never happens unnoticed again.
        """
        if hasattr(self.db_manager, "get_index_constituent_symbols_asof"):
            try:
                return set(self.db_manager.get_index_constituent_symbols_asof(index_code, as_of_date) or [])
            except Exception:
                log.warning(
                    "as-of constituent lookup failed for %s at %s; trying snapshot collection",
                    index_code,
                    as_of_date,
                    exc_info=True,
                )

        coll = getattr(self.db_manager, "index_constituents", None)
        if coll is not None:
            try:
                return set(asof_snapshot_symbols(coll, index_code, as_of_date))
            except Exception:
                log.warning(
                    "as-of snapshot lookup failed for %s at %s",
                    index_code,
                    as_of_date,
                    exc_info=True,
                )

        if hasattr(self.db_manager, "get_latest_index_constituent_symbols"):
            try:
                symbols = set(self.db_manager.get_latest_index_constituent_symbols(index_code) or [])
            except Exception:
                log.warning("latest constituent lookup failed for %s", index_code, exc_info=True)
            else:
                if index_code not in self._survivorship_warned:
                    self._survivorship_warned.add(index_code)
                    log.warning(
                        "%s membership at %s fell back to TODAY's constituents: results carry "
                        "survivorship bias and overstate returns",
                        index_code,
                        as_of_date,
                    )
                return symbols

        log.warning("no constituents resolved for %s at %s; universe is empty", index_code, as_of_date)
        return set()

    def _load_universe_symbols(self, index_codes: Sequence[str]) -> set:
        symbols = set()
        for index_code in index_codes:
            symbols.update(self._load_index_symbols(index_code))
        return symbols

    def _load_index_symbols(self, index_code: str) -> set:
        if hasattr(self.db_manager, "get_latest_index_constituent_symbols"):
            try:
                return set(self.db_manager.get_latest_index_constituent_symbols(index_code) or [])
            except Exception:
                log.warning("latest constituent lookup failed for %s; trying snapshot", index_code, exc_info=True)

        coll = getattr(self.db_manager, "index_constituents", None)
        if coll is not None:
            try:
                return set(latest_snapshot_symbols(coll, index_code))
            except Exception:
                log.warning("latest snapshot lookup failed for %s", index_code, exc_info=True)

        log.warning("no constituents resolved for %s; universe is empty", index_code)
        return set()

    def _preload_price_cache(self, panel: pd.DataFrame) -> None:
        if panel.empty:
            self._price_cache = {}
            return

        # score_date is a normalized YYYYMMDD string, so min() is chronological.
        earliest_score_date_by_symbol = panel.groupby("symbol", sort=False)["score_date"].min()
        tasks = {
            symbol: self._shift_start_date_back(score_date, VOLATILITY_LOOKBACK_CALENDAR_DAYS)
            for symbol, score_date in earliest_score_date_by_symbol.items()
        }

        workers = _preload_worker_count()
        if workers <= 1 or len(tasks) <= 1:
            self._price_cache = {
                symbol: self._load_symbol_price_history(symbol, shifted)
                for symbol, shifted in tasks.items()
            }
            return

        cache: Dict[str, List[Dict[str, Any]]] = {}
        with ThreadPoolExecutor(
            max_workers=min(workers, len(tasks)), thread_name_prefix="pr-price"
        ) as pool:
            future_to_symbol = {
                pool.submit(self._load_symbol_price_history, symbol, shifted): symbol
                for symbol, shifted in tasks.items()
            }
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    cache[symbol] = future.result()
                except Exception as exc:  # pragma: no cover - defensive; per-symbol reads already guard
                    print(f"price preload failed for {symbol}: {exc}", flush=True)
                    cache[symbol] = []
        self._price_cache = cache

    def _load_future_prices(self, symbol: str, score_date: str, max_horizon: int) -> List[Dict[str, Any]]:
        cached = self._price_cache.get(symbol)
        if cached is None:
            # Fallback for direct unit calls that bypass build_dataset().
            cached = self._load_symbol_price_history(symbol, score_date)
            self._price_cache[symbol] = cached
        return self._slice_future_prices(cached, score_date, max_horizon, symbol=symbol)

    def _load_symbol_price_history(self, symbol: str, start_date: str) -> List[Dict[str, Any]]:
        end_date = "99991231"
        try:
            price_data = self.db_manager.get_volume_price_data(
                symbol,
                start_date=start_date,
                end_date=end_date,
                adjust="hfq",
                projection=_VOLUME_PRICE_PROJECTION,
            )
        except TypeError:
            price_data = self.db_manager.get_volume_price_data(symbol)

        if not price_data:
            return []

        cleaned = []
        for row in price_data:
            trade_date = self._normalize_date(row.get("trade_date") or row.get("date"))
            open_price = self._to_float(row.get("open") or row.get("open_price") or row.get("adj_open"))
            close = self._to_float(row.get("close") or row.get("close_price") or row.get("adj_close"))
            high = self._to_float(row.get("high") or row.get("high_price") or row.get("adj_high"))
            low = self._to_float(row.get("low") or row.get("low_price") or row.get("adj_low"))
            if trade_date and trade_date >= start_date and close is not None and close > 0:
                cleaned.append(
                    {
                        "trade_date": trade_date,
                        "symbol": symbol,
                        "open": open_price if open_price and open_price > 0 else close,
                        "close": close,
                        "high": high,
                        "low": low,
                        # Needed to recover the executable raw price; dropping it
                        # here silently turns lot sizing back into the hfq bug.
                        "adj_factor": self._to_float(row.get("adj_factor")),
                    }
                )

        cleaned.sort(key=lambda x: x["trade_date"])
        return cleaned

    @staticmethod
    def _shift_start_date_back(start_date: str, calendar_days: int) -> str:
        dt = datetime.strptime(start_date, "%Y%m%d")
        return (dt - timedelta(days=int(calendar_days))).strftime("%Y%m%d")

    def _slice_future_prices(
        self,
        price_history: Sequence[Dict[str, Any]],
        score_date: str,
        max_horizon: int,
        *,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        trade_dates = self._ascending_trade_dates(price_history, symbol)
        if trade_dates is None:
            future_prices = [row for row in price_history if row.get("trade_date") and row["trade_date"] >= score_date]
            return future_prices[: max_horizon + 1]
        start = bisect_left(trade_dates, score_date)
        return list(price_history[start : start + max_horizon + 1])

    def _ascending_trade_dates(
        self, price_history: Sequence[Dict[str, Any]], symbol: Optional[str]
    ) -> Optional[List[str]]:
        """Trade dates of ``price_history``, or ``None`` if it resists a binary search.

        A daily multi-year backtest slices the same history once per score date, so
        scanning it each time makes panel construction quadratic in the backtest
        span. Histories built by ``_load_symbol_price_history`` are sorted and fully
        dated; anything else falls back to the linear scan.

        The index is keyed on object identity, so a history must be replaced rather
        than mutated in place; appending to one already handed out here would leave
        this answering from a stale index.
        """
        if symbol is not None:
            cached = self._price_date_index.get(symbol)
            if cached is not None and cached[0] is price_history:
                return cached[1]

        trade_dates: Optional[List[str]] = []
        previous = ""
        for row in price_history:
            trade_date = row.get("trade_date")
            if not trade_date or trade_date < previous:
                trade_dates = None
                break
            trade_dates.append(trade_date)
            previous = trade_date

        if symbol is not None:
            self._price_date_index[symbol] = (price_history, trade_dates)
        return trade_dates

    def _calculate_forward_returns(
        self,
        price_data: List[Dict[str, Any]],
        horizons: Sequence[int],
        score_date: Optional[str] = None,
        *,
        trailing_stop_pct: Optional[float] = None,
        cash_exit_dates: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        if not price_data:
            return {}

        if self.config.execution_price != "next_open":
            raise ValueError(f"Unsupported execution_price: {self.config.execution_price}")

        entry_idx = self._find_entry_index(price_data, score_date)
        if entry_idx is None:
            return {}
        entry_row = price_data[entry_idx]
        previous_row = price_data[entry_idx - 1] if entry_idx > 0 else None
        if self._is_one_price_limit_up(entry_row, previous_row):
            return {}

        entry_price = self._execution_price(entry_row)
        if entry_price is None or entry_price <= 0:
            return {}

        cash_dates = (
            frozenset(str(date) for date in cash_exit_dates if str(date))
            if cash_exit_dates is not None
            else None
        )
        returns: Dict[str, Any] = {
            "entry_price": entry_price,
            # Lot sizing and cash amounts are executable quantities, so they must
            # use the raw (unadjusted) price; returns stay on the adjusted series.
            "entry_raw_price": self._raw_execution_price(entry_row),
            "entry_date": entry_row.get("trade_date"),
            "entry_price_type": self.config.execution_price,
            "entry_blocked_by_limit": False,
        }
        for horizon in horizons:
            target_idx = entry_idx + int(horizon)
            exit_reason = "horizon"
            peak_high: Optional[float] = None
            stop_trigger_date: Optional[str] = None
            trailing_pct = self._normalize_trailing_stop_pct(
                trailing_stop_pct if trailing_stop_pct is not None else self.config.trailing_stop_pct
            )
            if trailing_pct is not None:
                peak_high, stop_trigger_date, stop_target_idx = self._find_trailing_stop_target(
                    price_data,
                    entry_idx=entry_idx,
                    horizon=int(horizon),
                    trailing_pct=trailing_pct,
                )
                if stop_target_idx is not None:
                    target_idx = stop_target_idx
                    exit_reason = "trailing_stop"
            self._fill_horizon_exit(
                returns,
                price_data,
                entry_price=entry_price,
                horizon=int(horizon),
                target_idx=target_idx,
                exit_reason=exit_reason,
                peak_high=peak_high,
                stop_trigger_date=stop_trigger_date,
                suffix="",
            )
            if cash_dates is not None:
                live_target = target_idx
                live_reason = exit_reason
                cash_target = self._find_regime_cash_target(
                    price_data,
                    entry_idx=entry_idx,
                    horizon=int(horizon),
                    cash_dates=cash_dates,
                )
                if cash_target is not None and cash_target < live_target:
                    live_target = cash_target
                    live_reason = "regime_cash"
                self._fill_horizon_exit(
                    returns,
                    price_data,
                    entry_price=entry_price,
                    horizon=int(horizon),
                    target_idx=live_target,
                    exit_reason=live_reason,
                    peak_high=peak_high,
                    stop_trigger_date=stop_trigger_date,
                    suffix=f"_{LIVE_CASH_SUFFIX}",
                )
        return returns

    def _fill_horizon_exit(
        self,
        returns: Dict[str, Any],
        price_data: Sequence[Dict[str, Any]],
        *,
        entry_price: float,
        horizon: int,
        target_idx: int,
        exit_reason: str,
        peak_high: Optional[float],
        stop_trigger_date: Optional[str],
        suffix: str,
    ) -> None:
        exit_idx = None
        future_row = None
        future_price = None
        if len(price_data) > target_idx:
            exit_idx = self._find_exit_index(price_data, target_idx)
            if exit_idx is not None:
                future_row = price_data[exit_idx]
                future_price = self._execution_price(future_row)

        if future_price is not None and future_price > 0:
            returns[f"forward_return_{horizon}d{suffix}"] = future_price / entry_price - 1
            returns[f"exit_price_{horizon}d{suffix}"] = future_price
            returns[f"exit_raw_price_{horizon}d{suffix}"] = self._raw_execution_price(
                future_row
            )
            returns[f"exit_date_{horizon}d{suffix}"] = future_row.get("trade_date")
            returns[f"exit_price_type_{horizon}d{suffix}"] = self.config.execution_price
            returns[f"exit_delayed_days_{horizon}d{suffix}"] = int(exit_idx - target_idx)
            returns[f"exit_reason_{horizon}d{suffix}"] = exit_reason
            if suffix == "":
                if peak_high is not None:
                    returns[f"peak_high_{horizon}d"] = peak_high
                if stop_trigger_date:
                    returns[f"stop_trigger_date_{horizon}d"] = stop_trigger_date
            return
        if suffix:
            return
        mark_row = price_data[-1]
        mark_price = self._execution_price(mark_row)
        if mark_price is not None and mark_price > 0:
            returns[f"unrealized_return_{horizon}d"] = mark_price / entry_price - 1
            returns[f"mark_price_{horizon}d"] = mark_price
            returns[f"mark_raw_price_{horizon}d"] = self._raw_execution_price(mark_row)
            returns[f"mark_date_{horizon}d"] = mark_row.get("trade_date")
            returns[f"is_open_{horizon}d"] = True

    def _max_future_bars_needed(
        self,
        trailing_stop_pct: Optional[float] = None,
        *,
        need_cash_exit: bool = False,
    ) -> int:
        max_horizon = max(self.config.horizons) + max(0, int(self.config.exit_limit_delay_days)) + 1
        extra_session = self._normalize_trailing_stop_pct(
            trailing_stop_pct if trailing_stop_pct is not None else self.config.trailing_stop_pct
        ) is not None or need_cash_exit
        if extra_session:
            # Trailing stop / regime cash may trigger on the last horizon day
            # and exit the next session.
            max_horizon += 1
        return max_horizon

    def _trailing_stop_pct(self) -> Optional[float]:
        return self._normalize_trailing_stop_pct(self.config.trailing_stop_pct)

    @staticmethod
    def _normalize_trailing_stop_pct(pct: Any) -> Optional[float]:
        """Accept a ratio in (0, 1] or legacy percent points (e.g. 8 → 0.08)."""
        return normalize_trailing_stop_ratio(pct)

    def _find_trailing_stop_target(
        self,
        price_data: Sequence[Dict[str, Any]],
        *,
        entry_idx: int,
        horizon: int,
        trailing_pct: float,
    ) -> Tuple[Optional[float], Optional[str], Optional[int]]:
        """Return (peak_high, trigger_date, exit_target_idx) when stop fires."""
        last_idx = entry_idx + int(horizon)
        if last_idx >= len(price_data):
            return None, None, None
        peak_high = self._bar_high(price_data[entry_idx])
        if peak_high is None or peak_high <= 0:
            return None, None, None
        for idx in range(entry_idx, last_idx + 1):
            bar_high = self._bar_high(price_data[idx])
            if bar_high is not None and bar_high > peak_high:
                peak_high = bar_high
            close_price = self._bar_close(price_data[idx])
            if close_price is None:
                continue
            stop_floor = peak_high * (1.0 - trailing_pct)
            if close_price <= stop_floor:
                trigger_date = price_data[idx].get("trade_date")
                return peak_high, str(trigger_date) if trigger_date else None, idx + 1
        return peak_high, None, None

    def _find_regime_cash_target(
        self,
        price_data: Sequence[Dict[str, Any]],
        *,
        entry_idx: int,
        horizon: int,
        cash_dates: Sequence[str],
    ) -> Optional[int]:
        """Exit next open after the first cash close in the holding window.

        Matches live plan generation: monitor day sees non-bull, execute_date is
        the next session. Cash on the score_date itself is a rebalance skip, not
        an intra-period flatten, and entry bars are already after score_date.
        """
        if not cash_dates:
            return None
        last_idx = entry_idx + int(horizon)
        if last_idx >= len(price_data):
            last_idx = len(price_data) - 1
        cash = cash_dates if isinstance(cash_dates, (set, frozenset)) else frozenset(str(d) for d in cash_dates)
        for idx in range(entry_idx, last_idx + 1):
            trade_date = price_data[idx].get("trade_date")
            if trade_date and str(trade_date) in cash:
                return idx + 1
        return None

    def _bar_high(self, row: Dict[str, Any]) -> Optional[float]:
        return self._to_float(row.get("high") or row.get("high_price") or row.get("adj_high") or row.get("close"))

    def _bar_close(self, row: Dict[str, Any]) -> Optional[float]:
        return self._to_float(row.get("close") or row.get("close_price") or row.get("adj_close"))

    def _find_entry_index(self, price_data: Sequence[Dict[str, Any]], score_date: Optional[str]) -> Optional[int]:
        if not score_date:
            return 1 if len(price_data) > 1 else None
        for idx, row in enumerate(price_data):
            trade_date = row.get("trade_date")
            if trade_date and trade_date > score_date:
                return idx
        return None

    def _find_exit_index(self, price_data: Sequence[Dict[str, Any]], target_idx: int) -> Optional[int]:
        max_delay = max(0, int(self.config.exit_limit_delay_days))
        last_idx = min(len(price_data) - 1, target_idx + max_delay)
        for idx in range(target_idx, last_idx + 1):
            previous_row = price_data[idx - 1] if idx > 0 else None
            if not self._is_one_price_limit_down(price_data[idx], previous_row):
                return idx
        return None

    def _execution_price(self, row: Dict[str, Any]) -> Optional[float]:
        return self._to_float(row.get("open") or row.get("close"))

    def _raw_execution_price(self, row: Dict[str, Any]) -> Optional[float]:
        """Execution price in raw (unadjusted) terms for lot sizing and cash.

        The research price cache is hfq-adjusted and carries ``adj_factor``; raw
        = adjusted / adj_factor. Rows without a usable factor (e.g. synthetic
        datasets or pre-2015 history) stay on the adjusted price.
        """
        price = self._execution_price(row)
        if price is None:
            return None
        factor = self._to_float(row.get("adj_factor"))
        if factor is None or factor <= 0:
            return price
        return price / factor

    def _is_one_price_limit_up(self, row: Dict[str, Any], previous_row: Optional[Dict[str, Any]]) -> bool:
        return self._is_one_price_limit(row, previous_row, direction="up")

    def _is_one_price_limit_down(self, row: Dict[str, Any], previous_row: Optional[Dict[str, Any]]) -> bool:
        return self._is_one_price_limit(row, previous_row, direction="down")

    def _is_one_price_limit(self, row: Dict[str, Any], previous_row: Optional[Dict[str, Any]], direction: str) -> bool:
        if not previous_row:
            return False
        open_price = self._to_float(row.get("open"))
        high = self._to_float(row.get("high"))
        low = self._to_float(row.get("low"))
        close = self._to_float(row.get("close"))
        previous_close = self._to_float(previous_row.get("close"))
        if None in (open_price, high, low, close, previous_close) or not previous_close:
            return False
        one_price = max(open_price, high, low, close) - min(open_price, high, low, close) <= max(0.01, close * 0.001)
        pct_change = close / previous_close - 1
        # Board-aware threshold: 10% main board, 20% STAR/ChiNext, 30% BSE,
        # 5% ST. The old fixed 9.5% blocked tradable STAR +10% one-price days
        # and let +20% STAR limit-ups through as fillable.
        limit_pct = self._price_limit_pct(row)
        tolerance = 0.005
        if direction == "up":
            return bool(one_price and pct_change >= limit_pct - tolerance)
        return bool(one_price and pct_change <= -limit_pct + tolerance)

    @staticmethod
    def _price_limit_pct(row: Dict[str, Any]) -> float:
        name = str(row.get("name") or "")
        if "ST" in name.upper():
            return 0.05
        code = str(row.get("symbol") or "").split(".")[0]
        if code.startswith(("688", "689")):
            return 0.20
        if code.startswith(("300", "301")):
            return 0.20
        if code.startswith(("8", "4")):
            return 0.30
        return 0.10

    def _score_row_from_doc(self, doc: Dict[str, Any]) -> Dict[str, Optional[float]]:
        scores = doc.get("scores") if isinstance(doc.get("scores"), dict) else {}
        composite = doc.get("composite_score")
        growth_score = self._extract_score(doc, scores, "growth")
        cycle_score = self._extract_score(doc, scores, "cycle")
        row = {
            "composite_score": self._coerce_composite_score(composite),
            "composite_balanced_score": self._extract_composite_strategy_score(composite, "balanced"),
            "composite_aggressive_score": self._extract_composite_strategy_score(composite, "aggressive"),
            "composite_conservative_score": self._extract_composite_strategy_score(composite, "conservative"),
            "composite_defensive_score": self._extract_composite_strategy_score(composite, "defensive"),
            "composite_value_oriented_score": self._extract_composite_strategy_score(composite, "value_oriented"),
            "composite_trading_oriented_score": self._extract_composite_strategy_score(composite, "trading_oriented"),
            "composite_growth_oriented_score": self._extract_composite_strategy_score(composite, "growth_oriented"),
            "composite_cycle_oriented_score": self._extract_composite_strategy_score(composite, "cycle_oriented"),
            "composite_growth_cycle_score": self._weighted_score(
                (growth_score, 0.6),
                (cycle_score, 0.4),
            ),
            "technical_score": self._extract_score(doc, scores, "technical"),
            "fundamental_score": self._extract_score(doc, scores, "fundamental"),
            "value_score": self._extract_score(doc, scores, "value"),
            "growth_score": growth_score,
            "money_flow_score": self._extract_score(doc, scores, "money_flow"),
            "cycle_score": cycle_score,
        }
        return row

    def _extract_composite_strategy_score(self, composite_score: Any, strategy: str) -> Optional[float]:
        if isinstance(composite_score, dict):
            return self._to_float(composite_score.get(strategy))
        if strategy == "balanced":
            return self._to_float(composite_score)
        return None

    def _extract_score(self, doc: Dict[str, Any], scores: Dict[str, Any], name: str) -> Optional[float]:
        direct = self._to_float(doc.get(f"{name}_score"))
        if direct is not None:
            return direct
        return self._to_float(scores.get(name))

    def _coerce_composite_score(self, value: Any) -> Optional[float]:
        if isinstance(value, dict):
            balanced = self._to_float(value.get("balanced"))
            if balanced is not None:
                return balanced
            numeric = [self._to_float(v) for v in value.values()]
            numeric = [v for v in numeric if v is not None]
            return max(numeric) if numeric else None
        return self._to_float(value)

    def _weighted_score(self, *items: tuple) -> Optional[float]:
        weighted_sum = 0.0
        total_weight = 0.0
        for value, weight in items:
            numeric = self._to_float(value)
            if numeric is None:
                continue
            weighted_sum += numeric * float(weight)
            total_weight += float(weight)
        if total_weight <= 0:
            return None
        return weighted_sum / total_weight

    def _normalize_date(self, value: Any) -> Optional[str]:
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            return value.strftime("%Y%m%d")
        text = str(value).replace("-", "").replace("/", "")[:8]
        return text if len(text) == 8 and text.isdigit() else None

    def _to_float(self, value: Any) -> Optional[float]:
        # No ``value == ""`` guard: float("") already yields None via ValueError,
        # while comparing pd.NA to a string raises instead of answering.
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if numeric != numeric:  # NaN; cheaper than pd.isna on a plain float
            return None
        return numeric

    def _return_col(self, horizon: int) -> str:
        return f"forward_return_{horizon}d"

    def _simulate_open_holding(
        self,
        price_data: Sequence[Dict[str, Any]],
        *,
        entry_idx: int,
        horizon: int,
        trailing_stop_pct: Optional[float] = None,
        cash_exit_dates: Optional[Sequence[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Hold from ``entry_idx`` open for ``horizon`` sessions (stop / cash may cut short)."""
        if entry_idx < 0 or entry_idx >= len(price_data) or int(horizon) < 1:
            return None
        entry_row = price_data[entry_idx]
        previous_row = price_data[entry_idx - 1] if entry_idx > 0 else None
        if self._is_one_price_limit_up(entry_row, previous_row):
            return None
        entry_price = self._execution_price(entry_row)
        if entry_price is None or entry_price <= 0:
            return None
        target_idx = entry_idx + int(horizon)
        exit_reason = "horizon"
        trailing_pct = self._normalize_trailing_stop_pct(trailing_stop_pct)
        if trailing_pct is not None:
            _peak, _trigger, stop_target_idx = self._find_trailing_stop_target(
                price_data,
                entry_idx=entry_idx,
                horizon=int(horizon),
                trailing_pct=trailing_pct,
            )
            if stop_target_idx is not None:
                target_idx = stop_target_idx
                exit_reason = "trailing_stop"
        if cash_exit_dates:
            cash_dates = frozenset(str(date) for date in cash_exit_dates if str(date))
            cash_target = self._find_regime_cash_target(
                price_data,
                entry_idx=entry_idx,
                horizon=int(horizon),
                cash_dates=cash_dates,
            )
            if cash_target is not None and cash_target < target_idx:
                target_idx = cash_target
                exit_reason = "regime_cash"
        if len(price_data) <= target_idx:
            return None
        exit_idx = self._find_exit_index(price_data, target_idx)
        if exit_idx is None:
            return None
        exit_row = price_data[exit_idx]
        exit_price = self._execution_price(exit_row)
        if exit_price is None or exit_price <= 0:
            return None
        return {
            "holding_return": float(exit_price / entry_price - 1.0),
            "entry_date": entry_row.get("trade_date"),
            "entry_price": float(entry_price),
            "entry_raw_price": self._raw_execution_price(entry_row),
            "exit_date": exit_row.get("trade_date"),
            "exit_price": float(exit_price),
            "exit_raw_price": self._raw_execution_price(exit_row),
            "exit_reason": exit_reason,
            "entry_idx": int(entry_idx),
            "exit_idx": int(exit_idx),
        }


def find_bar_index(price_data: Sequence[Dict[str, Any]], trade_date: Optional[str]) -> Optional[int]:
    """Return the index of ``trade_date`` in a price history, or None."""
    if not trade_date:
        return None
    wanted = str(trade_date)
    for idx, row in enumerate(price_data):
        if str(row.get("trade_date") or "") == wanted:
            return idx
    return None


def simulate_open_holding(
    price_data: Sequence[Dict[str, Any]],
    *,
    entry_idx: int,
    horizon: int,
    trailing_stop_pct: Optional[float] = None,
    cash_exit_dates: Optional[Sequence[str]] = None,
    exit_limit_delay_days: int = 5,
) -> Optional[Dict[str, Any]]:
    """Session-open holding used by intra-period refill (no score-date shift)."""
    helper = ScoreBacktester.__new__(ScoreBacktester)
    helper.config = BacktestConfig(
        horizons=(max(1, int(horizon)),),
        trailing_stop_pct=trailing_stop_pct,
        exit_limit_delay_days=int(exit_limit_delay_days),
    )
    return helper._simulate_open_holding(
        price_data,
        entry_idx=entry_idx,
        horizon=horizon,
        trailing_stop_pct=trailing_stop_pct,
        cash_exit_dates=cash_exit_dates,
    )


def export_dataset_to_path(df: pd.DataFrame, path: str | Path) -> Path:
    """Persist the row-level backtest panel.

    ``.parquet`` uses pandas/pyarrow when available; otherwise writes
    ``<name>.csv.gz`` next to the requested path and returns that path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not df.empty and "score_date" in df.columns:
        df = df.sort_values(["score_date", "symbol"]).reset_index(drop=True)

    p_lower = str(target).lower()
    if p_lower.endswith(".parquet"):
        try:
            df.to_parquet(target, index=False)
            return target
        except ImportError:
            # Silence here cost production roughly 50s a run: the image shipped
            # without pyarrow for months and gzip CSV is ~10x slower on a real panel.
            log.warning("pyarrow unavailable; writing %s as gzip CSV instead", target.name)
            fallback = target.with_suffix(".csv.gz")
            df.to_csv(fallback, index=False, compression="gzip")
            return fallback
    if p_lower.endswith(".csv.gz"):
        df.to_csv(target, index=False, compression="gzip")
        return target
    df.to_csv(target, index=False)
    return target


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate stock scores against future returns.")
    parser.add_argument("--start-date", help="Inclusive score_date lower bound, YYYYMMDD")
    parser.add_argument("--end-date", help="Inclusive score_date upper bound, YYYYMMDD")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10, 20], help="Forward return horizons in trading days")
    parser.add_argument("--quantiles", type=int, default=5, help="Number of score buckets")
    parser.add_argument("--index-codes", nargs="+", default=[], help="Optional stock_scores.index_codes filter, e.g. csi1000 or csi2000")
    parser.add_argument(
        "--universe-index",
        "--universe-index-codes",
        nargs="+",
        default=[],
        help="Optional index universe filter loaded from index_constituents, e.g. csi1000 or csi2000",
    )
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument(
        "--export-dataset",
        metavar="PATH",
        help="Write full row-level dataset (.parquet needs pyarrow; otherwise .csv or .csv.gz)",
    )
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=[10, 20, 50],
        help="Top-K sizes for daily long top vs universe / vs bottom stats (default: 10 20 50)",
    )
    parser.add_argument("--enable-tushare", action="store_true", help="Keep TUSHARE_TOKEN enabled when creating ResearchDataAccess")
    return parser.parse_args()


def main() -> int:
    from shared.data_access import open_research_data_access_for_cli

    args = _parse_args()
    config = BacktestConfig(
        horizons=tuple(args.horizons),
        quantiles=args.quantiles,
        start_date=args.start_date,
        end_date=args.end_date,
        index_codes=tuple(args.index_codes),
        universe_index_codes=tuple(args.universe_index),
        top_k_sizes=tuple(int(k) for k in args.top_k if int(k) >= 1),
    )
    result = ScoreBacktester(open_research_data_access_for_cli(enable_tushare=args.enable_tushare), config).run()
    payload = result.to_dict()

    if args.export_dataset:
        written = export_dataset_to_path(result.dataset, args.export_dataset)
        print(f"Wrote row-level dataset to {written}", file=sys.stderr)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
