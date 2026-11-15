#!/usr/bin/env python3
"""Reusable builder for per-combination portfolio trade detail.

Single source of truth for the "click a combo -> open its simulated holdings /
trades" feature. The heavy lifting (per-rebalance selection + per-stock round-lot
buy/sell simulation with fees) already lives in ``PortfolioBacktester`` behind the
``include_trades`` flag; this module just wires the same dataset/overlay/weight
plumbing the sweep uses and returns a stable, JSON-safe detail dict.

Design contract (so static Option A and a future on-demand Option B share code):

* ``build_dataset_bundle(...)`` builds the (expensive) score/return dataset ONCE
  for a (universe, date range, overlay) and can be reused across every
  construction-mode variant.
* ``build_combo_detail(bundle, variant_id, top_n)`` runs one combination with
  ``include_trades=True`` and returns ``{meta, summary, periods, trades}``.
* The JSON schema returned here is the stable interface. Option A writes it to a
  file; Option B would return the very same shape from an HTTP endpoint. Only the
  transport differs, never the schema or the builder.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from analysis.core.benchmark_prices import load_index_benchmark_prices
from analysis.core.portfolio_backtester import PortfolioBacktestConfig, PortfolioBacktester
from analysis.sweep.score_columns import (
    add_growth_cycle_weight_columns,
    build_score_sweeps,
    overlay_cycle_score,
)

# Canonical construction-mode variants: id -> (construction_mode, max_industry_weight)
# ``None`` weight means the mode does not use an active industry cap.
VARIANT_SPECS: Dict[str, Tuple[str, Optional[float]]] = {
    "topn": ("top_n", None),
    "cap20": ("industry_capped", 0.20),
    "cap25": ("industry_capped", 0.25),
    "cap30": ("industry_capped", 0.30),
    "cap40": ("industry_capped", 0.40),
    "neutral": ("industry_neutral", None),
}

ALL_VARIANTS: Tuple[str, ...] = tuple(VARIANT_SPECS.keys())


def combo_id(universe_label: str, variant_id: str) -> str:
    """Stable id used both as the JSON filename and the summary-page link key."""
    return f"{universe_label}_{variant_id}"


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy scalars and NaN/Inf into plain JSON-safe values."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes, bool)):
        try:
            obj = obj.item()
        except Exception:
            # not a scalar numpy value after all; the checks below still handle it
            pass
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


@dataclass
class DatasetBundle:
    """Prebuilt, reusable dataset + benchmark prices for a (universe, range, overlay)."""

    dataset: Any  # pandas.DataFrame
    index_prices: List[Dict[str, Any]]
    trading_calendar: List[Any]
    score_column: str
    meta: Dict[str, Any]


def build_dataset_bundle(
    db_manager,
    universe_index: Sequence[str],
    start_date: str,
    end_date: str,
    horizon: int = 20,
    growth_cycle_weight: str = "60:40",
    cycle_score_source: Optional[str] = None,
    index_benchmark_symbol: Optional[str] = None,
    verbose: bool = True,
) -> DatasetBundle:
    """Build the score/return dataset once; reuse across all construction variants.

    Mirrors ``portfolio_sweep_report.main`` exactly (same BacktestConfig, optional
    cycle overlay, same growth/cycle weighted score column) so detail numbers stay
    consistent with the sweep summary.
    """
    from analysis.core.score_backtester import BacktestConfig, ScoreBacktester

    score_sweeps = build_score_sweeps("composite_growth_cycle_score", [growth_cycle_weight])
    score_column = score_sweeps[0].score_column

    dataset = ScoreBacktester(
        db_manager,
        BacktestConfig(
            horizons=(horizon,),
            start_date=start_date,
            end_date=end_date,
            universe_index_codes=tuple(universe_index),
        ),
    ).build_dataset()
    if dataset.empty:
        raise ValueError("build_dataset returned empty; check universe/date range")
    if cycle_score_source:
        dataset = overlay_cycle_score(dataset, cycle_score_source, verbose)
    dataset = add_growth_cycle_weight_columns(dataset, score_sweeps)

    index_prices: List[Dict[str, Any]] = []
    trading_calendar: List[Any] = []
    if index_benchmark_symbol:
        index_prices = list(load_index_benchmark_prices(db_manager, index_benchmark_symbol, start_date))
        trading_calendar = [row.get("trade_date") for row in index_prices]

    meta = {
        "universe_index": list(universe_index),
        "start_date": start_date,
        "end_date": end_date,
        "horizon": int(horizon),
        "growth_cycle_weight": growth_cycle_weight,
        "growth_weight": score_sweeps[0].growth_weight,
        "cycle_weight": score_sweeps[0].cycle_weight,
        "cycle_score_source": cycle_score_source,
        "index_benchmark_symbol": index_benchmark_symbol,
        "score_column": score_column,
    }
    return DatasetBundle(dataset, index_prices, trading_calendar, score_column, meta)


def research_combo_key(
    score_variant: str,
    variant: str,
    top_n: int,
    rebalance_interval_days: Optional[int] = None,
) -> str:
    """Stable id for a single sweep row's drill-down detail.

    Built from the fields every ``run_parameter_sweep`` row already carries, so the
    research worker (writer) and the API/UI (reader) agree without extra mapping. The
    rebalance interval is included so rows that differ only by interval do not collide.
    """
    safe = lambda s: re.sub(r"[^0-9A-Za-z]+", "-", str(s)).strip("-")
    key = f"{safe(score_variant)}__{safe(variant)}__tn{int(top_n)}"
    if rebalance_interval_days is not None:
        key += f"__rb{int(rebalance_interval_days)}"
    return key


def run_single_combo(
    dataset: Any,
    score_column: str,
    construction_mode: str,
    top_n: int,
    horizon: int,
    *,
    rebalance_interval_days: Optional[int] = None,
    max_industry_weight: Optional[float] = None,
    index_prices: Optional[List[Dict[str, Any]]] = None,
    trading_calendar: Optional[List[Any]] = None,
    index_benchmark_symbol: Optional[str] = None,
    transaction_cost: float = 0.001,
    buy_commission_rate: Optional[float] = None,
    sell_commission_rate: Optional[float] = None,
    min_commission: Optional[float] = None,
    stamp_tax_rate: Optional[float] = None,
    transfer_fee_rate: Optional[float] = None,
    initial_capital: float = 1_000_000.0,
    meta_extra: Optional[Dict[str, Any]] = None,
    volatility_mode: str = "off",
    target_vol_annual: Optional[float] = None,
    volatility_column: str = "vol_20d_annual",
    selection_mode: str = "fixed_top_n",
    threshold_lookback_days: int = 10,
    max_positions: int = 20,
    cash_on_dates: Optional[Sequence[str]] = None,
    intra_period_cash: bool = False,
    bull_exposure: float = 1.0,
    intra_period_refill: bool = False,
    crowd_exit_k: int = 0,
    trailing_stop_pct: Optional[float] = None,
    non_bull_top_n_scale: float = 0.0,
    non_bull_keep_parent_slots: bool = False,
    industry_regime_scale: float = 1.0,
    industry_l1_codes: Sequence[str] = (),
    price_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run ONE combination with trade tracking against an already-built dataset.

    This is the reusable core: it never (re)builds the dataset, so callers that
    already hold a dataset (e.g. the research worker, which builds it once for the
    whole sweep) can generate per-combo detail cheaply. Returns a JSON-safe
    ``{meta, summary, periods, trades}`` dict.
    """
    config = PortfolioBacktestConfig(
        score_column=score_column,
        horizon=int(horizon),
        top_n_values=(int(top_n),),
        selection_mode=selection_mode,
        threshold_lookback_days=int(threshold_lookback_days),
        max_positions=int(max_positions),
        rebalance_interval_days=int(rebalance_interval_days) if rebalance_interval_days is not None else int(horizon),
        transaction_cost=transaction_cost,
        buy_commission_rate=buy_commission_rate,
        sell_commission_rate=sell_commission_rate,
        min_commission=min_commission,
        stamp_tax_rate=stamp_tax_rate,
        transfer_fee_rate=transfer_fee_rate,
        index_benchmark_symbol=index_benchmark_symbol,
        initial_capital=initial_capital,
        include_trades=True,
        construction_mode=construction_mode,
        max_industry_weight=max_industry_weight if max_industry_weight is not None else 0.3,
        volatility_mode=volatility_mode,
        target_vol_annual=target_vol_annual,
        volatility_column=volatility_column,
        cash_on_dates=tuple(str(date) for date in (cash_on_dates or []) if str(date)),
        intra_period_cash=bool(intra_period_cash),
        bull_exposure=float(bull_exposure),
        intra_period_refill=bool(intra_period_refill),
        crowd_exit_k=int(crowd_exit_k or 0),
        trailing_stop_pct=trailing_stop_pct,
        non_bull_top_n_scale=float(non_bull_top_n_scale or 0.0),
        non_bull_keep_parent_slots=bool(non_bull_keep_parent_slots),
        industry_regime_scale=float(industry_regime_scale),
        industry_l1_codes=tuple(str(code) for code in (industry_l1_codes or ()) if str(code).strip()),
    )
    result = PortfolioBacktester(
        dataset,
        config,
        index_price_history=index_prices or [],
        trading_calendar=trading_calendar or [],
        price_cache=price_cache,
    ).run()
    detail = result.to_dict()
    summary_rows = detail.get("summary") or []

    meta: Dict[str, Any] = {
        "score_column": score_column,
        "construction_mode": construction_mode,
        "max_industry_weight": max_industry_weight,
        "top_n": int(top_n),
        "selection_mode": selection_mode,
        "threshold_lookback_days": int(threshold_lookback_days),
        "max_positions": int(max_positions),
        "horizon": int(horizon),
        "rebalance_interval_days": int(rebalance_interval_days) if rebalance_interval_days is not None else int(horizon),
        "index_benchmark_symbol": index_benchmark_symbol,
        "industry_l1_codes": list(industry_l1_codes or ()),
        "fees": {
            "transaction_cost": transaction_cost,
            "buy_commission_rate": buy_commission_rate,
            "sell_commission_rate": sell_commission_rate,
            "min_commission": min_commission,
            "stamp_tax_rate": stamp_tax_rate,
            "transfer_fee_rate": transfer_fee_rate,
            "initial_capital": initial_capital,
        },
    }
    if meta_extra:
        meta.update(meta_extra)
    payload = {
        "meta": meta,
        "summary": summary_rows[0] if summary_rows else {},
        "periods": detail.get("periods", []),
        "trades": detail.get("trades", []),
    }
    return sanitize_for_json(payload)


def compact_trade_date(value: Any) -> str:
    return "".join(ch for ch in str(value or "") if ch.isdigit())[:8]


def annotate_combo_regime(
    detail: Dict[str, Any],
    schedule: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Copy as-of index regime labels onto combo periods and trades.

    ``schedule`` entries use ``score_date`` / ``label`` / ``action`` from
    ``build_regime_cash_schedule``. Missing or unmatched dates are left unchanged.
    """
    by_date: Dict[str, Dict[str, Any]] = {}
    for entry in schedule or []:
        key = compact_trade_date(entry.get("score_date"))
        if not key:
            continue
        by_date[key] = {
            "regime_label": entry.get("label"),
            "regime_action": entry.get("action"),
        }

    def apply(row: Any) -> None:
        if not isinstance(row, dict):
            return
        info = by_date.get(compact_trade_date(row.get("score_date")))
        if not info:
            return
        row["regime_label"] = info["regime_label"]
        row["regime_action"] = info["regime_action"]

    for period in detail.get("periods") or []:
        apply(period)
    for trade in detail.get("trades") or []:
        apply(trade)
    return detail


def build_combo_detail(
    bundle: DatasetBundle,
    variant_id: str,
    top_n: int = 20,
    *,
    transaction_cost: float = 0.001,
    buy_commission_rate: Optional[float] = None,
    sell_commission_rate: Optional[float] = None,
    min_commission: Optional[float] = None,
    stamp_tax_rate: Optional[float] = None,
    transfer_fee_rate: Optional[float] = None,
    initial_capital: float = 1_000_000.0,
) -> Dict[str, Any]:
    """Run a single combination (by canonical variant id) against a prebuilt bundle.

    Thin wrapper over ``run_single_combo`` that resolves a ``VARIANT_SPECS`` id and
    merges the bundle metadata. Returns ``{meta, summary, periods, trades}``.
    """
    if variant_id not in VARIANT_SPECS:
        raise ValueError(f"unknown variant_id {variant_id!r}; expected one of {sorted(VARIANT_SPECS)}")
    construction_mode, max_weight = VARIANT_SPECS[variant_id]
    meta_extra = dict(bundle.meta)
    meta_extra["variant_id"] = variant_id
    return run_single_combo(
        bundle.dataset,
        bundle.score_column,
        construction_mode,
        int(top_n),
        int(bundle.meta["horizon"]),
        max_industry_weight=max_weight,
        index_prices=bundle.index_prices,
        trading_calendar=bundle.trading_calendar,
        index_benchmark_symbol=bundle.meta.get("index_benchmark_symbol"),
        transaction_cost=transaction_cost,
        buy_commission_rate=buy_commission_rate,
        sell_commission_rate=sell_commission_rate,
        min_commission=min_commission,
        stamp_tax_rate=stamp_tax_rate,
        transfer_fee_rate=transfer_fee_rate,
        initial_capital=initial_capital,
        meta_extra=meta_extra,
    )
