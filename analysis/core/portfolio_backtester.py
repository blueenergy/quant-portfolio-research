#!/usr/bin/env python3
"""Portfolio-level backtests for score-ranked stock selections."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from analysis.core.benchmark_prices import load_index_benchmark_prices
from shared.industry_universe import (
    SW_L1_CODE_COL,
    SW_L1_REGIME_COL,
    attach_sw_l1,
    resolve_industry_l1_codes,
    sw_l1_catalog,
)
from analysis.core.path_features import PATH_FEATURE_COLUMNS, apply_path_gate
from shared.industry_universe import apply_industry_filter
from analysis.core.score_backtester import (
    BacktestConfig,
    ScoreBacktester,
    find_bar_index,
    live_cash_field,
    simulate_open_holding,
)
from analysis.core.trade_fee_model import TradeFeeModel
from analysis.core.trailing_stop import normalize_trailing_stop_ratio

_MIN_SPAN_YEARS_FOR_CAGR = 1 / 12
_MAX_REFILLS_PER_SLOT = 5


def _span_years_from_score_dates(score_dates: Sequence[str]) -> Optional[float]:
    dates = sorted(str(date) for date in score_dates if date)
    if len(dates) < 2:
        return None
    try:
        first = pd.Timestamp(dates[0])
        last = pd.Timestamp(dates[-1])
    except (ValueError, TypeError):
        return None
    days = (last - first).days
    if days <= 0:
        return None
    return days / 365.25


def _cagr(
    cumulative_return: Optional[float],
    span_years: Optional[float],
    *,
    min_span_years: float = _MIN_SPAN_YEARS_FOR_CAGR,
) -> Optional[float]:
    if cumulative_return is None or span_years is None or span_years < min_span_years:
        return None
    base = 1 + float(cumulative_return)
    if base <= 0:
        return None
    return float(base ** (1 / span_years) - 1)


def _yearly_compound_returns(periods: pd.DataFrame, return_col: str) -> Dict[str, float]:
    if periods.empty or return_col not in periods.columns or "score_date" not in periods.columns:
        return {}
    yearly: Dict[str, float] = {}
    year_groups = periods.groupby(periods["score_date"].astype(str).str[:4], sort=True)
    for year, group in year_groups:
        returns = group[return_col].astype(float)
        yearly[str(year)] = float((1 + returns).prod() - 1)
    return yearly


def _yearly_index_excess(periods: pd.DataFrame) -> Dict[str, float]:
    if periods.empty or "score_date" not in periods.columns:
        return {}
    yearly: Dict[str, float] = {}
    year_groups = periods.groupby(periods["score_date"].astype(str).str[:4], sort=True)
    for year, group in year_groups:
        portfolio_equity = float((1 + group["portfolio_return_net"].astype(float)).prod())
        if "index_benchmark_return" not in group.columns:
            continue
        index_returns = group["index_benchmark_return"].dropna().astype(float)
        if index_returns.empty:
            continue
        index_equity = float((1 + index_returns).prod())
        yearly[str(year)] = portfolio_equity - index_equity
    return yearly


@dataclass(frozen=True)
class PortfolioBacktestConfig:
    score_column: str = "composite_cycle_oriented_score"
    horizon: int = 20
    top_n_values: Sequence[int] = (50, 100)
    selection_mode: str = "fixed_top_n"
    threshold_lookback_days: int = 10
    max_positions: int = 20
    rebalance_interval_days: Optional[int] = 20
    rebalance_step: Optional[int] = None
    transaction_cost: float = 0.001
    buy_commission_rate: Optional[float] = None
    sell_commission_rate: Optional[float] = None
    min_commission: Optional[float] = None
    stamp_tax_rate: Optional[float] = None
    transfer_fee_rate: Optional[float] = None
    index_benchmark_symbol: Optional[str] = None
    initial_capital: float = 1_000_000.0
    include_trades: bool = False
    lot_size: int = 100
    construction_mode: str = "top_n"
    max_industry_weight: float = 0.3
    volatility_mode: str = "off"
    target_vol_annual: Optional[float] = None
    volatility_column: str = "vol_20d_annual"
    path_gate: str = "off"
    industry_l1_codes: Sequence[str] = ()
    cash_on_dates: Sequence[str] = ()
    intra_period_cash: bool = False
    bull_exposure: float = 1.0
    intra_period_refill: bool = False
    crowd_exit_k: int = 0
    trailing_stop_pct: Optional[float] = None
    non_bull_top_n_scale: float = 0.0
    non_bull_keep_parent_slots: bool = False
    # Per-name overlay: a name whose own SW L1 industry index is not bull has its
    # target weight multiplied by this factor. 1.0 disables the overlay entirely.
    industry_regime_scale: float = 1.0
    industry_regime_column: str = SW_L1_REGIME_COL
    # Taxonomy used by industry_capped / industry_neutral. Defaults to SW L1:
    # the research UI and industry filter use SW L1, while the legacy
    # ``industry`` column is a finer sub-industry that silently under-caps.
    industry_cap_column: str = SW_L1_CODE_COL

    def __post_init__(self) -> None:
        exposure = float(self.bull_exposure)
        if not (0.0 < exposure <= 1.0):
            raise ValueError(f"bull_exposure must be in (0, 1], got {self.bull_exposure}")
        crowd_k = int(self.crowd_exit_k or 0)
        if crowd_k < 0:
            raise ValueError(f"crowd_exit_k must be >= 0, got {self.crowd_exit_k}")
        object.__setattr__(self, "crowd_exit_k", crowd_k)
        scale = float(self.non_bull_top_n_scale or 0.0)
        if scale < 0.0 or scale > 1.0:
            raise ValueError(f"non_bull_top_n_scale must be in [0, 1], got {self.non_bull_top_n_scale}")
        object.__setattr__(self, "non_bull_top_n_scale", scale)
        object.__setattr__(self, "non_bull_keep_parent_slots", bool(self.non_bull_keep_parent_slots))
        industry_scale = float(self.industry_regime_scale)
        if industry_scale < 0.0 or industry_scale > 1.0:
            raise ValueError(
                f"industry_regime_scale must be in [0, 1], got {self.industry_regime_scale}"
            )
        object.__setattr__(self, "industry_regime_scale", industry_scale)
        object.__setattr__(self, "industry_regime_column", str(self.industry_regime_column))
        object.__setattr__(
            self,
            "industry_cap_column",
            str(self.industry_cap_column or SW_L1_CODE_COL),
        )
        selection_mode = (self.selection_mode or "fixed_top_n").strip().lower()
        if selection_mode not in {"fixed_top_n", "dynamic_score_threshold"}:
            raise ValueError(f"Unsupported selection_mode: {self.selection_mode}")
        if selection_mode != "dynamic_score_threshold":
            return
        top_n_values = tuple(int(value) for value in self.top_n_values)
        if len(top_n_values) != 1:
            raise ValueError(
                "dynamic_score_threshold requires top_n_values to contain exactly one value"
            )
        top_n = top_n_values[0]
        if top_n < 1:
            raise ValueError("dynamic_score_threshold requires top_n >= 1")
        if int(self.threshold_lookback_days) < 1:
            raise ValueError(
                "dynamic_score_threshold requires threshold_lookback_days >= 1"
            )
        if int(self.max_positions) < top_n:
            raise ValueError(
                "dynamic_score_threshold requires max_positions >= top_n"
            )


@dataclass
class PortfolioBacktestResult:
    periods: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "periods": self.periods.to_dict(orient="records"),
            "summary": self.summary.to_dict(orient="records"),
            "trades": self.trades.to_dict(orient="records"),
        }


class PortfolioBacktester:
    """Build simple equal-weight top-N portfolios from score validation datasets."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        config: Optional[PortfolioBacktestConfig] = None,
        index_price_history: Optional[Sequence[Dict[str, Any]]] = None,
        trading_calendar: Optional[Sequence[str]] = None,
        *,
        copy_dataset: bool = True,
        sorted_groups: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
        selection_cache: Optional[Dict[Any, List[Dict[str, Any]]]] = None,
        selection_cache_maxsize: int = 256,
        price_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ):
        self.dataset = dataset.copy() if copy_dataset else dataset
        self.config = config or PortfolioBacktestConfig()
        # Per-score_column daily sorted groups. When ``sorted_groups`` is injected it is
        # shared across every backtester in a sweep so the groupby+sort runs once per
        # score column instead of once per parameter combination.
        self._sorted_groups = sorted_groups if sorted_groups is not None else {}
        # Optional selection-plan cache shared across volatility variants (which do not
        # change stock selection, only weighting). ``None`` disables cross-combo reuse.
        # Default 256 covers typical research grids (~100 unique selection keys);
        # <=0 means unbounded (no eviction).
        self._selection_cache = selection_cache
        self._selection_cache_maxsize = int(selection_cache_maxsize)
        self.fee_model = TradeFeeModel.from_params(
            transaction_cost=self.config.transaction_cost,
            buy_commission_rate=self.config.buy_commission_rate,
            sell_commission_rate=self.config.sell_commission_rate,
            min_commission=self.config.min_commission,
            stamp_tax_rate=self.config.stamp_tax_rate,
            transfer_fee_rate=self.config.transfer_fee_rate,
        )
        self.index_price_history = self._clean_price_history(index_price_history or [])
        self.trading_calendar = self._clean_trading_calendar(trading_calendar)
        self._cash_on_dates = frozenset(str(date) for date in (self.config.cash_on_dates or ()) if str(date))
        # Shared read-only bars (sweep fork CoW). Do not copy: CSI1000 cache is huge.
        self._price_cache = price_cache if price_cache is not None else {}
        if self.config.intra_period_refill and not self._price_cache:
            raise ValueError("intra_period_refill requires a price_cache of daily bars")
        if int(self.config.crowd_exit_k or 0) > 0 and not self._price_cache:
            raise ValueError("crowd_exit_k requires a price_cache of daily bars")
        if self.config.intra_period_cash and not self.dataset.empty:
            live_col = live_cash_field("forward_return", self.config.horizon)
            if live_col not in self.dataset.columns:
                raise ValueError(
                    "intra_period_cash requires live-cash return columns; "
                    "call ScoreBacktester.add_live_cash_return_columns first"
                )
        # Fail loudly rather than silently weighting at 1.0: the column is dropped
        # by the column whitelist and by ``row.get`` without any trace otherwise.
        if float(self.config.industry_regime_scale) < 1.0 and not self.dataset.empty:
            if self.config.industry_regime_column not in self.dataset.columns:
                raise ValueError(
                    f"industry_regime_scale={self.config.industry_regime_scale} requires the "
                    f"{self.config.industry_regime_column!r} column on the dataset; the research "
                    "worker attaches it next to attach_sw_l1"
                )

    def _selection_mode(self) -> str:
        return (self.config.selection_mode or "fixed_top_n").strip().lower()

    def _is_dynamic_selection(self) -> bool:
        return self._selection_mode() == "dynamic_score_threshold"

    def _is_cash_date(self, score_date: str) -> bool:
        return str(score_date) in self._cash_on_dates

    def _non_bull_top_n_scale(self) -> float:
        return float(self.config.non_bull_top_n_scale or 0.0)

    def _is_flat_cash_date(self, score_date: str) -> bool:
        """True when the overlay flats the book (scale 0). Half-TopN stays invested."""
        return self._is_cash_date(score_date) and self._non_bull_top_n_scale() <= 0.0

    def _non_bull_action(self) -> str:
        scale = self._non_bull_top_n_scale()
        if scale <= 0.0:
            return "cash"
        if self.config.non_bull_keep_parent_slots:
            return "half_topn_slot"
        return "half_topn"

    def _effective_select_n(self, score_date: str, top_n: int) -> int:
        scale = self._non_bull_top_n_scale()
        if scale <= 0.0 or not self._is_cash_date(score_date) or self._is_dynamic_selection():
            return int(top_n)
        return max(1, int(round(int(top_n) * scale)))

    def _weight_denominator(self, score_date: str, top_n: int, selected_n: int) -> Optional[int]:
        """Override equal-weight denominator only on the half-TopN overlay.

        ``None`` means callers keep the historical default (dynamic: parent TopN,
        equal-weight: 1/n).
        """
        if (
            self._non_bull_top_n_scale() <= 0.0
            or not self._is_cash_date(score_date)
            or self._is_dynamic_selection()
        ):
            return None
        held = max(int(selected_n), 1)
        if bool(self.config.non_bull_keep_parent_slots):
            return max(int(top_n), held)
        return held

    @property
    def price_cache(self) -> Dict[str, List[Dict[str, Any]]]:
        return self._price_cache

    @staticmethod
    def _next_rebalance_date(score_date: str, selected_dates: Sequence[str]) -> Optional[str]:
        current = str(score_date)
        for date in selected_dates:
            if str(date) > current:
                return str(date)
        return None

    def _first_intra_cash_date(self, score_date: str, next_rebalance: Optional[str]) -> Optional[str]:
        """First non-bull session strictly after this rebalance and before the next.

        Live monitor days can flatten in that window; the next rebalance date itself
        is a new period (cash or invest), not an intra-period exit.
        """
        if not self.config.intra_period_cash or self._non_bull_top_n_scale() > 0.0:
            return None
        first: Optional[str] = None
        start = str(score_date)
        end = str(next_rebalance) if next_rebalance else None
        for date in self._cash_on_dates:
            if date <= start:
                continue
            if end is not None and date >= end:
                continue
            if first is None or date < first:
                first = date
        return first

    @staticmethod
    def _empty_selected() -> pd.DataFrame:
        return pd.DataFrame(columns=["symbol"])

    @staticmethod
    def _cash_turnover(previous_selection: set) -> float:
        return 1.0 if previous_selection else 0.0

    def run(self) -> PortfolioBacktestResult:
        periods_df, trades_df = self._run_top_n_values()
        all_summary = []
        if not periods_df.empty:
            for top_n in self.config.top_n_values:
                top_n_periods = periods_df[periods_df["top_n"] == int(top_n)]
                if not top_n_periods.empty:
                    all_summary.append(self._summarize_periods(top_n_periods, int(top_n)))
        summary_df = pd.DataFrame(all_summary) if all_summary else pd.DataFrame()
        return PortfolioBacktestResult(periods=periods_df, summary=summary_df, trades=trades_df)

    def _run_top_n_values(self) -> tuple:
        ret_col = self._return_col(self.config.horizon)
        mark_col = f"mark_price_{self.config.horizon}d"
        if self.dataset.empty or self.config.score_column not in self.dataset.columns:
            return pd.DataFrame(), pd.DataFrame()
        # Selection is independent of volatility mode and fees, so when a shared
        # selection cache is available (sweep path, no per-trade detail) reuse the
        # cached per-date selection across volatility variants and only recompute
        # the volatility-weighted return here.
        if (
            self._selection_cache is not None
            and not self.config.include_trades
            and ret_col in self.dataset.columns
        ):
            return self._run_top_n_values_via_plan(ret_col), pd.DataFrame()
        has_returns = ret_col in self.dataset.columns
        has_open_marks = mark_col in self.dataset.columns
        # Allow open-only datasets (no completed forward returns) to still emit MTM trades.
        if not has_returns and not has_open_marks:
            return pd.DataFrame(), pd.DataFrame()

        top_n_values = [int(value) for value in self.config.top_n_values]
        rows = []
        trade_rows = []
        previous_selection_by_top_n = {top_n: set() for top_n in top_n_values}
        previous_dynamic_weights_by_top_n = {top_n: {"__cash__": 1.0} for top_n in top_n_values}
        # Trade detail is sized against the capital actually available at that
        # rebalance, so it compounds alongside the `(1 + portfolio_return_net).cumprod()`
        # that produces `cumulative_return`. Advancing here rather than from the finished
        # summary keeps both on the same row sequence, cash-only periods included.
        equity_by_top_n = {top_n: float(self.config.initial_capital) for top_n in top_n_values}
        score_dates = sorted(self.dataset["score_date"].dropna().unique())
        selected_dates = self._select_rebalance_dates(score_dates)
        valid_columns = self._base_valid_columns(ret_col) if has_returns else []

        for score_date in selected_dates:
            next_rebalance = self._next_rebalance_date(score_date, selected_dates)
            intra_cash_date = self._first_intra_cash_date(score_date, next_rebalance)
            if has_returns and self._is_flat_cash_date(score_date):
                rows.extend(
                    self._cash_period_rows(
                        score_date,
                        top_n_values,
                        ret_col,
                        previous_selection_by_top_n,
                        previous_dynamic_weights_by_top_n,
                        equity_by_top_n=equity_by_top_n,
                    )
                )
                continue
            group = self._sorted_group_for_date(score_date, self.config.score_column)
            if group is None:
                continue
            group = self._apply_industry_l1_filter(group)
            if group.empty:
                continue
            dynamic_selection = None
            if self._is_dynamic_selection():
                dynamic_selection = self._select_for_date(
                    score_date,
                    group,
                    score_dates,
                    top_n_values[0],
                )
                # A complete lookback is required before the first dynamic period.
                if dynamic_selection is None:
                    continue
            if not has_returns:
                if self.config.include_trades:
                    for top_n in top_n_values:
                        selected = (
                            dynamic_selection["selected"]
                            if dynamic_selection is not None
                            else None
                        )
                        trade_rows.extend(
                            self._build_open_trade_rows(
                                score_date,
                                top_n,
                                group,
                                selected=selected,
                            )
                        )
                continue
            valid = group[valid_columns].dropna(subset=["symbol", self.config.score_column, ret_col])
            if valid.empty:
                if self.config.include_trades:
                    for top_n in top_n_values:
                        selected = (
                            dynamic_selection["selected"]
                            if dynamic_selection is not None
                            else None
                        )
                        trade_rows.extend(
                            self._build_open_trade_rows(
                                score_date,
                                top_n,
                                group,
                                selected=selected,
                            )
                        )
                continue

            benchmark_return = float(valid[ret_col].mean())
            index_benchmark_return = self._calculate_index_benchmark_return(
                score_date,
                until_date=intra_cash_date,
            )
            path_gate = (self.config.path_gate or "off").strip().lower()
            gated_valid = apply_path_gate(valid, path_gate)
            path_gate_pool_size = (
                int(dynamic_selection["path_gate_pool_size"])
                if dynamic_selection is not None
                else int(len(gated_valid))
            )

            for top_n in top_n_values:
                select_n = self._effective_select_n(str(score_date), top_n)
                selected = (
                    dynamic_selection["selected"]
                    if dynamic_selection is not None
                    else self._select_portfolio(gated_valid, select_n)
                )
                selected_symbols = (
                    set(selected["symbol"])
                    if "symbol" in selected.columns
                    else set()
                )
                if not selected_symbols and not self._is_dynamic_selection():
                    continue

                if self._is_dynamic_selection():
                    weight_n = self._weight_denominator(str(score_date), top_n, len(selected))
                    current_weights = self._base_target_weights(
                        selected,
                        int(weight_n) if weight_n is not None else top_n,
                        ret_col=ret_col,
                    )
                    turnover = self._calculate_weight_turnover(
                        previous_dynamic_weights_by_top_n[top_n],
                        current_weights,
                    )
                else:
                    previous_selection = previous_selection_by_top_n[top_n]
                    turnover = self._calculate_turnover(previous_selection, selected_symbols)
                holdings, fee_cost_rate, refill_count, crowd_flat, crowd_triggered, adjusted = (
                    self._period_performance(
                        selected,
                        ret_col,
                        turnover,
                        score_date=str(score_date),
                        top_n=top_n,
                    )
                )
                gross_return = holdings["gross_return"]
                portfolio_exposure = holdings["exposure"]
                net_return = gross_return - fee_cost_rate

                row = {
                    "score_date": score_date,
                    "top_n": top_n,
                    "score_column": self.config.score_column,
                    "horizon": self.config.horizon,
                    "rebalance_interval_days": self.config.rebalance_interval_days,
                    "rebalance_step": self.config.rebalance_step,
                    "selection_mode": self._selection_mode(),
                    "threshold_lookback_days": int(self.config.threshold_lookback_days),
                    "max_positions": int(self.config.max_positions),
                    "construction_mode": self.config.construction_mode,
                    "max_industry_weight": self.config.max_industry_weight,
                    "volatility_mode": self.config.volatility_mode,
                    "target_vol_annual": self.config.target_vol_annual,
                    "path_gate": path_gate,
                    "path_gate_pool_size": path_gate_pool_size,
                    "selected_count": int(len(selected)),
                    "portfolio_exposure": portfolio_exposure,
                    "portfolio_return": gross_return,
                    "portfolio_return_net": net_return,
                    "benchmark_return": benchmark_return,
                    "excess_return": net_return - benchmark_return,
                    "turnover": turnover,
                    "estimated_fee_cost": fee_cost_rate,
                    "intra_period_cash_exit": bool(intra_cash_date),
                    "cash_exit_date": intra_cash_date,
                    "intra_period_refills": int(refill_count),
                    "bull_exposure": float(self.config.bull_exposure),
                    "crowd_exit_k": int(self.config.crowd_exit_k or 0),
                    "crowd_exit": bool(crowd_triggered),
                    "crowd_flattened": int(crowd_flat),
                    "non_bull_action": self._non_bull_action(),
                    "non_bull_top_n_scale": self._non_bull_top_n_scale(),
                    "non_bull_keep_parent_slots": bool(self.config.non_bull_keep_parent_slots),
                }
                if dynamic_selection is not None:
                    row.update(
                        {
                            "score_threshold": dynamic_selection["score_threshold"],
                            "threshold_sample_size": dynamic_selection[
                                "threshold_sample_size"
                            ],
                            "threshold_rank": dynamic_selection["threshold_rank"],
                            "cash_weight": max(0.0, 1.0 - portfolio_exposure),
                            "unfilled_weight": holdings["unfilled_weight"],
                            "unfilled_count": int(holdings["unfilled_count"]),
                        }
                    )
                if index_benchmark_return is not None:
                    row["index_benchmark_symbol"] = self.config.index_benchmark_symbol
                    row["index_benchmark_return"] = index_benchmark_return
                    row["index_excess_return"] = net_return - index_benchmark_return
                rows.append(row)
                # Size this period's book against the capital it starts with, then let
                # the period's own return carry the running equity into the next one.
                period_equity = equity_by_top_n[top_n]
                equity_by_top_n[top_n] = period_equity * (1.0 + net_return)
                if self.config.include_trades:
                    trade_rows.extend(
                        self._build_trade_rows(
                            score_date,
                            top_n,
                            adjusted,
                            ret_col,
                            turnover,
                            equity=period_equity,
                        )
                    )
                    # Always pass the period book. selected=None would re-run TopN
                    # on names that only have marks (halt / delist, no 20d exit)
                    # and emit phantom lots that were never in the portfolio.
                    trade_rows.extend(
                        self._build_open_trade_rows(
                            score_date,
                            top_n,
                            group,
                            selected=adjusted,
                            equity=period_equity,
                        )
                    )
                previous_selection_by_top_n[top_n] = (
                    set() if intra_cash_date or crowd_triggered else selected_symbols
                )
                if self._is_dynamic_selection():
                    previous_dynamic_weights_by_top_n[top_n] = (
                        {"__cash__": 1.0} if intra_cash_date or crowd_triggered else current_weights
                    )

        return pd.DataFrame(rows), pd.DataFrame(trade_rows)

    def _sorted_group_for_date(self, score_date: Any, score_column: str) -> Optional[pd.DataFrame]:
        if score_column not in self._sorted_groups:
            grouped: Dict[str, pd.DataFrame] = {}
            if score_column in self.dataset.columns and "score_date" in self.dataset.columns:
                for date_value, group in self.dataset.groupby("score_date"):
                    grouped[str(date_value)] = group.sort_values(score_column, ascending=False)
            self._sorted_groups[score_column] = grouped
        return self._sorted_groups[score_column].get(str(score_date))

    def _score_valid_pool(self, group: pd.DataFrame) -> pd.DataFrame:
        """Return the point-in-time selection pool without consulting future returns."""
        if group.empty:
            return group.copy()
        valid = group.dropna(subset=["symbol", self.config.score_column]).copy()
        valid[self.config.score_column] = pd.to_numeric(
            valid[self.config.score_column],
            errors="coerce",
        )
        valid = valid.dropna(subset=[self.config.score_column])
        if self.config.industry_l1_codes:
            valid = apply_industry_filter(valid, self.config.industry_l1_codes)
        path_gate = (self.config.path_gate or "off").strip().lower()
        return apply_path_gate(valid, path_gate).sort_values(
            self.config.score_column,
            ascending=False,
        )

    def _dynamic_threshold_for_date(
        self,
        score_date: Any,
        all_score_dates: Sequence[Any],
        top_n: int,
    ) -> Optional[Dict[str, Any]]:
        cutoff_scores: List[float] = []
        lookback = int(self.config.threshold_lookback_days)
        current_date = str(score_date)
        for historical_date in reversed([str(value) for value in all_score_dates]):
            if historical_date >= current_date:
                continue
            group = self._sorted_group_for_date(
                historical_date,
                self.config.score_column,
            )
            if group is None:
                continue
            historical_pool = self._score_valid_pool(group)
            if len(historical_pool) < top_n:
                continue
            cutoff_scores.append(
                float(historical_pool.iloc[top_n - 1][self.config.score_column])
            )
            if len(cutoff_scores) >= lookback:
                break
        if len(cutoff_scores) < lookback:
            return None
        return {
            "score_threshold": float(sum(cutoff_scores) / len(cutoff_scores)),
            "threshold_sample_size": len(cutoff_scores),
            "threshold_rank": int(top_n),
        }

    def _select_for_date(
        self,
        score_date: Any,
        group: pd.DataFrame,
        all_score_dates: Sequence[Any],
        top_n: int,
    ) -> Optional[Dict[str, Any]]:
        score_pool = self._score_valid_pool(group)
        if not self._is_dynamic_selection():
            return {
                "selected": self._select_portfolio(score_pool, top_n),
                "path_gate_pool_size": int(len(score_pool)),
            }

        threshold_info = self._dynamic_threshold_for_date(
            score_date,
            all_score_dates,
            top_n,
        )
        if threshold_info is None:
            return None
        threshold = float(threshold_info["score_threshold"])
        threshold_pool = score_pool[
            pd.to_numeric(score_pool[self.config.score_column], errors="coerce")
            >= threshold
        ]
        selected = self._select_portfolio(
            threshold_pool,
            int(self.config.max_positions),
        )
        return {
            "selected": selected,
            "path_gate_pool_size": int(len(score_pool)),
            **threshold_info,
        }

    def _plan_valid_columns(self, ret_col: str) -> List[str]:
        """Columns retained in a cached selection plan.

        Superset of ``_base_valid_columns`` that always keeps the volatility column
        and path features (when present) so a plan built by one volatility variant can
        be reused to weight any other variant. Extra columns never change which rows
        survive (dropna targets only symbol/score/return) nor the ranking, so selection
        is identical to the inline path.
        """
        columns = list(self._base_valid_columns(ret_col))
        for extra in [self.config.volatility_column, *PATH_FEATURE_COLUMNS]:
            if extra in self.dataset.columns and extra not in columns:
                columns.append(extra)
        return columns

    def _apply_industry_l1_filter(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.config.industry_l1_codes:
            return apply_industry_filter(frame, self.config.industry_l1_codes)
        return frame

    def _universe_benchmark_return(self, score_date: str, ret_col: str) -> float:
        group = self._sorted_group_for_date(score_date, self.config.score_column)
        if group is None or group.empty:
            return 0.0
        group = self._apply_industry_l1_filter(group)
        if group.empty or ret_col not in group.columns:
            return 0.0
        valid = group.dropna(subset=["symbol", ret_col])
        if valid.empty:
            return 0.0
        return float(valid[ret_col].mean())

    def _cash_period_rows(
        self,
        score_date: str,
        top_n_values: Sequence[int],
        ret_col: str,
        previous_selection_by_top_n: Dict[int, set],
        previous_dynamic_weights_by_top_n: Dict[int, Dict[str, float]],
        *,
        equity_by_top_n: Optional[Dict[int, float]] = None,
    ) -> List[Dict[str, Any]]:
        benchmark_return = self._universe_benchmark_return(score_date, ret_col)
        index_benchmark_return = self._calculate_index_benchmark_return(score_date)
        path_gate = (self.config.path_gate or "off").strip().lower()
        selected = self._empty_selected()
        rows: List[Dict[str, Any]] = []
        for top_n in top_n_values:
            if self._is_dynamic_selection():
                current_weights = {"__cash__": 1.0}
                turnover = self._calculate_weight_turnover(
                    previous_dynamic_weights_by_top_n[top_n],
                    current_weights,
                )
                previous_dynamic_weights_by_top_n[top_n] = current_weights
            else:
                turnover = self._cash_turnover(previous_selection_by_top_n[top_n])
            holdings = self._weighted_portfolio_return(selected, ret_col)
            fee_cost_rate = self.fee_model.period_cost_rate(turnover)
            net_return = holdings["gross_return"] - fee_cost_rate
            row = {
                "score_date": score_date,
                "top_n": top_n,
                "score_column": self.config.score_column,
                "horizon": self.config.horizon,
                "rebalance_interval_days": self.config.rebalance_interval_days,
                "rebalance_step": self.config.rebalance_step,
                "selection_mode": self._selection_mode(),
                "threshold_lookback_days": int(self.config.threshold_lookback_days),
                "max_positions": int(self.config.max_positions),
                "construction_mode": self.config.construction_mode,
                "max_industry_weight": self.config.max_industry_weight,
                "volatility_mode": self.config.volatility_mode,
                "target_vol_annual": self.config.target_vol_annual,
                "path_gate": path_gate,
                "path_gate_pool_size": 0,
                "selected_count": 0,
                "portfolio_exposure": 0.0,
                "portfolio_return": holdings["gross_return"],
                "portfolio_return_net": net_return,
                "benchmark_return": benchmark_return,
                "excess_return": net_return - benchmark_return,
                "turnover": turnover,
                "estimated_fee_cost": fee_cost_rate,
                "cash_weight": 1.0,
            }
            if index_benchmark_return is not None:
                row["index_benchmark_symbol"] = self.config.index_benchmark_symbol
                row["index_benchmark_return"] = index_benchmark_return
                row["index_excess_return"] = net_return - index_benchmark_return
            rows.append(row)
            if equity_by_top_n is not None:
                equity_by_top_n[top_n] *= 1.0 + net_return
            previous_selection_by_top_n[top_n] = set()
        return rows

    def _cash_selection_plan_entry(
        self,
        score_date: str,
        top_n_values: Sequence[int],
        ret_col: str,
        previous_selection_by_top_n: Dict[int, set],
        previous_dynamic_weights_by_top_n: Dict[int, Dict[str, float]],
    ) -> Dict[str, Any]:
        benchmark_return = self._universe_benchmark_return(score_date, ret_col)
        index_benchmark_return = self._calculate_index_benchmark_return(score_date)
        per_top_n: Dict[int, Dict[str, Any]] = {}
        selected = self._empty_selected()
        for top_n in top_n_values:
            if self._is_dynamic_selection():
                current_weights = {"__cash__": 1.0}
                turnover = self._calculate_weight_turnover(
                    previous_dynamic_weights_by_top_n[top_n],
                    current_weights,
                )
                previous_dynamic_weights_by_top_n[top_n] = current_weights
            else:
                turnover = self._cash_turnover(previous_selection_by_top_n[top_n])
            per_top_n[top_n] = {"selected": selected, "turnover": turnover}
            previous_selection_by_top_n[top_n] = set()
        return {
            "score_date": score_date,
            "benchmark_return": benchmark_return,
            "index_benchmark_return": index_benchmark_return,
            "path_gate_pool_size": 0,
            "per_top_n": per_top_n,
        }

    def _selection_plan_key(self, top_n_values: Sequence[int]) -> tuple:
        return (
            self.config.score_column,
            self._selection_mode(),
            int(self.config.threshold_lookback_days),
            int(self.config.max_positions),
            (self.config.construction_mode or "top_n").lower(),
            self.config.max_industry_weight,
            (self.config.path_gate or "off").strip().lower(),
            tuple(str(code) for code in self.config.industry_l1_codes or ()),
            self.config.rebalance_interval_days,
            self.config.rebalance_step,
            self.config.horizon,
            self.config.volatility_column,
            self.config.index_benchmark_symbol,
            tuple(int(v) for v in top_n_values),
            tuple(sorted(self._cash_on_dates)),
            bool(self.config.intra_period_cash),
            self._non_bull_top_n_scale(),
            bool(self.config.non_bull_keep_parent_slots),
        )

    def _build_selection_plan(self, ret_col: str, top_n_values: Sequence[int]) -> List[Dict[str, Any]]:
        """Volatility/fee-independent per-date selection (selected rows, turnover, benchmarks)."""
        plan: List[Dict[str, Any]] = []
        previous_selection_by_top_n = {top_n: set() for top_n in top_n_values}
        previous_dynamic_weights_by_top_n = {
            top_n: {"__cash__": 1.0} for top_n in top_n_values
        }
        score_dates = sorted(self.dataset["score_date"].dropna().unique())
        selected_dates = self._select_rebalance_dates(score_dates)
        valid_columns = self._plan_valid_columns(ret_col)

        for score_date in selected_dates:
            next_rebalance = self._next_rebalance_date(score_date, selected_dates)
            intra_cash_date = self._first_intra_cash_date(score_date, next_rebalance)
            if self._is_flat_cash_date(score_date):
                plan.append(
                    self._cash_selection_plan_entry(
                        score_date,
                        top_n_values,
                        ret_col,
                        previous_selection_by_top_n,
                        previous_dynamic_weights_by_top_n,
                    )
                )
                continue
            group = self._sorted_group_for_date(score_date, self.config.score_column)
            if group is None:
                continue
            group = self._apply_industry_l1_filter(group)
            if group.empty:
                continue
            dynamic_selection = None
            if self._is_dynamic_selection():
                dynamic_selection = self._select_for_date(
                    score_date,
                    group,
                    score_dates,
                    top_n_values[0],
                )
                if dynamic_selection is None:
                    continue
            valid = group[valid_columns].dropna(subset=["symbol", self.config.score_column, ret_col])
            if valid.empty:
                continue

            benchmark_return = float(valid[ret_col].mean())
            index_benchmark_return = self._calculate_index_benchmark_return(
                score_date,
                until_date=intra_cash_date,
            )
            path_gate = (self.config.path_gate or "off").strip().lower()
            gated_valid = apply_path_gate(valid, path_gate)
            path_gate_pool_size = (
                int(dynamic_selection["path_gate_pool_size"])
                if dynamic_selection is not None
                else int(len(gated_valid))
            )

            per_top_n: Dict[int, Dict[str, Any]] = {}
            for top_n in top_n_values:
                select_n = self._effective_select_n(str(score_date), top_n)
                selected = (
                    dynamic_selection["selected"]
                    if dynamic_selection is not None
                    else self._select_portfolio(gated_valid, select_n)
                )
                selected_symbols = (
                    set(selected["symbol"])
                    if "symbol" in selected.columns
                    else set()
                )
                if not selected_symbols and not self._is_dynamic_selection():
                    continue
                if self._is_dynamic_selection():
                    weight_n = self._weight_denominator(str(score_date), top_n, len(selected))
                    current_weights = self._base_target_weights(
                        selected,
                        int(weight_n) if weight_n is not None else top_n,
                        ret_col=ret_col,
                    )
                    turnover = self._calculate_weight_turnover(
                        previous_dynamic_weights_by_top_n[top_n],
                        current_weights,
                    )
                    previous_dynamic_weights_by_top_n[top_n] = (
                        {"__cash__": 1.0} if intra_cash_date else current_weights
                    )
                else:
                    turnover = self._calculate_turnover(
                        previous_selection_by_top_n[top_n],
                        selected_symbols,
                    )
                per_top_n[top_n] = {
                    "selected": selected,
                    "turnover": turnover,
                    **(
                        {
                            "score_threshold": dynamic_selection["score_threshold"],
                            "threshold_sample_size": dynamic_selection[
                                "threshold_sample_size"
                            ],
                            "threshold_rank": dynamic_selection["threshold_rank"],
                        }
                        if dynamic_selection is not None
                        else {}
                    ),
                }
                previous_selection_by_top_n[top_n] = set() if intra_cash_date else selected_symbols

            plan.append(
                {
                    "score_date": score_date,
                    "benchmark_return": benchmark_return,
                    "index_benchmark_return": index_benchmark_return,
                    "path_gate_pool_size": path_gate_pool_size,
                    "intra_period_cash_exit": bool(intra_cash_date),
                    "cash_exit_date": intra_cash_date,
                    "per_top_n": per_top_n,
                }
            )
        return plan

    def _get_selection_plan(self, ret_col: str, top_n_values: Sequence[int]) -> List[Dict[str, Any]]:
        cache = self._selection_cache
        if cache is None:
            return self._build_selection_plan(ret_col, top_n_values)
        key = self._selection_plan_key(top_n_values)
        plan = cache.get(key)
        if plan is None:
            plan = self._build_selection_plan(ret_col, top_n_values)
            cache[key] = plan
            maxsize = self._selection_cache_maxsize
            if maxsize > 0:
                while len(cache) > maxsize:
                    oldest = next(iter(cache))
                    del cache[oldest]
        return plan

    def _run_top_n_values_via_plan(self, ret_col: str) -> pd.DataFrame:
        top_n_values = [int(value) for value in self.config.top_n_values]
        plan = self._get_selection_plan(ret_col, top_n_values)
        path_gate = (self.config.path_gate or "off").strip().lower()
        rows: List[Dict[str, Any]] = []
        previous_crowd_flat_by_top_n = {int(value): False for value in top_n_values}
        for entry in plan:
            score_date = entry["score_date"]
            benchmark_return = entry["benchmark_return"]
            index_benchmark_return = entry["index_benchmark_return"]
            path_gate_pool_size = entry["path_gate_pool_size"]
            intra_cash_date = entry.get("cash_exit_date")
            per_top_n = entry["per_top_n"]
            for top_n in top_n_values:
                sel = per_top_n.get(top_n)
                if sel is None:
                    continue
                selected = sel["selected"]
                turnover = sel["turnover"]
                if previous_crowd_flat_by_top_n[int(top_n)]:
                    if self._is_dynamic_selection():
                        current_weights = self._base_target_weights(
                            selected, int(top_n), ret_col=ret_col
                        )
                        turnover = self._calculate_weight_turnover(
                            {"__cash__": 1.0}, current_weights
                        )
                    else:
                        selected_symbols = (
                            set(selected["symbol"])
                            if selected is not None and "symbol" in selected.columns
                            else set()
                        )
                        turnover = self._calculate_turnover(set(), selected_symbols)
                holdings, fee_cost_rate, refill_count, crowd_flat, crowd_triggered, _adjusted = (
                    self._period_performance(
                        selected,
                        ret_col,
                        turnover,
                        score_date=str(score_date),
                        top_n=top_n,
                    )
                )
                gross_return = holdings["gross_return"]
                portfolio_exposure = holdings["exposure"]
                net_return = gross_return - fee_cost_rate

                row = {
                    "score_date": score_date,
                    "top_n": top_n,
                    "score_column": self.config.score_column,
                    "horizon": self.config.horizon,
                    "rebalance_interval_days": self.config.rebalance_interval_days,
                    "rebalance_step": self.config.rebalance_step,
                    "selection_mode": self._selection_mode(),
                    "threshold_lookback_days": int(self.config.threshold_lookback_days),
                    "max_positions": int(self.config.max_positions),
                    "construction_mode": self.config.construction_mode,
                    "max_industry_weight": self.config.max_industry_weight,
                    "volatility_mode": self.config.volatility_mode,
                    "target_vol_annual": self.config.target_vol_annual,
                    "path_gate": path_gate,
                    "path_gate_pool_size": path_gate_pool_size,
                    "selected_count": int(len(selected)),
                    "portfolio_exposure": portfolio_exposure,
                    "portfolio_return": gross_return,
                    "portfolio_return_net": net_return,
                    "benchmark_return": benchmark_return,
                    "excess_return": net_return - benchmark_return,
                    "turnover": turnover,
                    "estimated_fee_cost": fee_cost_rate,
                    "intra_period_cash_exit": bool(entry.get("intra_period_cash_exit")),
                    "cash_exit_date": intra_cash_date,
                    "intra_period_refills": int(refill_count),
                    "bull_exposure": float(self.config.bull_exposure),
                    "crowd_exit_k": int(self.config.crowd_exit_k or 0),
                    "crowd_exit": bool(crowd_triggered),
                    "crowd_flattened": int(crowd_flat),
                    "non_bull_action": self._non_bull_action(),
                    "non_bull_top_n_scale": self._non_bull_top_n_scale(),
                    "non_bull_keep_parent_slots": bool(self.config.non_bull_keep_parent_slots),
                }
                if self._is_dynamic_selection():
                    row.update(
                        {
                            "score_threshold": sel["score_threshold"],
                            "threshold_sample_size": sel["threshold_sample_size"],
                            "threshold_rank": sel["threshold_rank"],
                            "cash_weight": max(0.0, 1.0 - portfolio_exposure),
                            "unfilled_weight": holdings["unfilled_weight"],
                            "unfilled_count": int(holdings["unfilled_count"]),
                        }
                    )
                if index_benchmark_return is not None:
                    row["index_benchmark_symbol"] = self.config.index_benchmark_symbol
                    row["index_benchmark_return"] = index_benchmark_return
                    row["index_excess_return"] = net_return - index_benchmark_return
                rows.append(row)
                previous_crowd_flat_by_top_n[int(top_n)] = bool(crowd_triggered)
        return pd.DataFrame(rows)

    def _summarize_periods(self, periods: pd.DataFrame, top_n: int) -> Dict[str, Any]:
        returns = periods["portfolio_return_net"].astype(float)
        benchmark_returns = periods["benchmark_return"].astype(float)
        excess_returns = periods["excess_return"].astype(float)
        equity = (1 + returns).cumprod()
        benchmark_equity = (1 + benchmark_returns).cumprod()
        index_returns = periods["index_benchmark_return"].dropna().astype(float) if "index_benchmark_return" in periods else pd.Series(dtype=float)
        index_excess_returns = periods["index_excess_return"].dropna().astype(float) if "index_excess_return" in periods else pd.Series(dtype=float)
        index_equity = (1 + index_returns).cumprod()

        summary = {
            "top_n": top_n,
            "score_column": self.config.score_column,
            "horizon": self.config.horizon,
            "rebalance_interval_days": self.config.rebalance_interval_days,
            "rebalance_step": self.config.rebalance_step,
            "selection_mode": self._selection_mode(),
            "threshold_lookback_days": int(self.config.threshold_lookback_days),
            "max_positions": int(self.config.max_positions),
            "construction_mode": self.config.construction_mode,
            "max_industry_weight": self.config.max_industry_weight,
            "volatility_mode": self.config.volatility_mode,
            "target_vol_annual": self.config.target_vol_annual,
            "path_gate": self.config.path_gate,
            "periods": int(len(periods)),
            "cumulative_return": float(equity.iloc[-1] - 1) if len(equity) else None,
            "benchmark_cumulative_return": float(benchmark_equity.iloc[-1] - 1) if len(benchmark_equity) else None,
            "excess_cumulative_return": float(equity.iloc[-1] - benchmark_equity.iloc[-1]) if len(equity) else None,
            "average_period_return": float(returns.mean()) if len(returns) else None,
            "average_excess_return": float(excess_returns.mean()) if len(excess_returns) else None,
            "hit_rate": float((returns > 0).mean()) if len(returns) else None,
            "excess_hit_rate": float((excess_returns > 0).mean()) if len(excess_returns) else None,
            "volatility": float(returns.std(ddof=1)) if len(returns) > 1 else None,
            "sharpe": self._sharpe(returns),
            "sortino": self._sortino(returns),
            "information_ratio": self._information_ratio(excess_returns),
            "max_drawdown": self._max_drawdown(equity),
            "average_turnover": float(periods["turnover"].mean()) if len(periods) else None,
            "average_selected_count": float(periods["selected_count"].mean()) if len(periods) else None,
            "average_exposure": float(periods["portfolio_exposure"].mean()) if len(periods) and "portfolio_exposure" in periods else None,
            "bull_exposure": float(self.config.bull_exposure),
            "intra_period_refill": bool(self.config.intra_period_refill),
            "crowd_exit_k": int(self.config.crowd_exit_k or 0),
            "non_bull_action": self._non_bull_action(),
            "non_bull_top_n_scale": self._non_bull_top_n_scale(),
            "non_bull_keep_parent_slots": bool(self.config.non_bull_keep_parent_slots),
        }
        if "intra_period_cash_exit" in periods.columns:
            summary["intra_period_cash_exits"] = int(periods["intra_period_cash_exit"].eq(True).sum())
        if "intra_period_refills" in periods.columns:
            summary["intra_period_refills"] = int(
                pd.to_numeric(periods["intra_period_refills"], errors="coerce").fillna(0).sum()
            )
        if "crowd_exit" in periods.columns:
            summary["crowd_exits"] = int(periods["crowd_exit"].eq(True).sum())
        if "crowd_flattened" in periods.columns:
            summary["crowd_flattened"] = int(
                pd.to_numeric(periods["crowd_flattened"], errors="coerce").fillna(0).sum()
            )
        span_years = _span_years_from_score_dates(periods["score_date"].tolist()) if len(periods) else None
        cumulative_return = summary.get("cumulative_return")
        summary["span_years"] = float(span_years) if span_years is not None else None
        summary["annualized_return"] = _cagr(cumulative_return, span_years)
        max_drawdown = summary.get("max_drawdown")
        summary["calmar"] = (
            float(summary["annualized_return"] / abs(max_drawdown))
            if summary.get("annualized_return") is not None and max_drawdown
            else None
        )
        summary["yearly_returns"] = _yearly_compound_returns(periods, "portfolio_return_net")
        summary["yearly_index_excess"] = _yearly_index_excess(periods)

        if len(index_returns):
            index_cumulative_return = float(index_equity.iloc[-1] - 1)
            summary.update(
                {
                    "index_benchmark_symbol": self.config.index_benchmark_symbol,
                    "index_benchmark_periods": int(len(index_returns)),
                    "index_benchmark_cumulative_return": index_cumulative_return,
                    "index_excess_cumulative_return": float(equity.iloc[-1] - index_equity.iloc[-1]) if len(equity) else None,
                    "average_index_excess_return": float(index_excess_returns.mean()) if len(index_excess_returns) else None,
                    "index_excess_hit_rate": float((index_excess_returns > 0).mean()) if len(index_excess_returns) else None,
                    "index_information_ratio": self._information_ratio(index_excess_returns),
                    "annualized_index_benchmark_return": _cagr(index_cumulative_return, span_years),
                }
            )
        return summary

    def _select_portfolio(self, valid: pd.DataFrame, top_n: int) -> pd.DataFrame:
        mode = (self.config.construction_mode or "top_n").lower()
        if mode == "top_n" or "industry" not in valid.columns:
            # Copy so cached selection plans never hold a view into ``valid``.
            return valid.head(top_n).copy()
        if mode == "industry_capped":
            return self._select_industry_capped(valid, top_n)
        if mode == "industry_neutral":
            return self._select_industry_neutral(valid, top_n)
        raise ValueError(f"Unsupported construction_mode: {self.config.construction_mode}")

    def _industry_cap_groups(self, valid: pd.DataFrame) -> pd.Series:
        column = str(self.config.industry_cap_column or SW_L1_CODE_COL)
        if column not in valid.columns:
            column = "industry"
        if column not in valid.columns:
            return pd.Series("Unknown", index=valid.index)
        return valid[column].fillna("Unknown").astype(str).replace("", "Unknown")

    def _select_industry_capped(self, valid: pd.DataFrame, top_n: int) -> pd.DataFrame:
        # Equal-weight selection: count/top_n is the industry weight, so a
        # weight cap is enforced as max_count = floor(top_n * cap).
        max_count = max(1, int(top_n * float(self.config.max_industry_weight)))
        groups = self._industry_cap_groups(valid)
        industry_counts: Dict[str, int] = {}
        selected_rows = []
        for idx, row in valid.iterrows():
            industry = str(groups.loc[idx])
            if industry_counts.get(industry, 0) >= max_count:
                continue
            selected_rows.append(row)
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
            if len(selected_rows) >= top_n:
                break
        return pd.DataFrame(selected_rows)

    def _select_industry_neutral(self, valid: pd.DataFrame, top_n: int) -> pd.DataFrame:
        groups = self._industry_cap_groups(valid)
        industry_counts = groups.value_counts()
        total_count = float(industry_counts.sum())
        if total_count <= 0:
            return valid.head(top_n)

        raw_quotas = {industry: top_n * count / total_count for industry, count in industry_counts.items()}
        quotas = {industry: int(raw) for industry, raw in raw_quotas.items()}
        remaining = top_n - sum(quotas.values())
        remainders = sorted(raw_quotas.items(), key=lambda item: item[1] - int(item[1]), reverse=True)
        for industry, _ in remainders[:remaining]:
            quotas[industry] += 1

        selected = []
        selected_symbols = set()
        for industry, quota in quotas.items():
            if quota <= 0:
                continue
            group = valid[groups == industry]
            for _, row in group.head(quota).iterrows():
                selected.append(row)
                selected_symbols.add(row.get("symbol"))

        if len(selected) < top_n:
            for _, row in valid.iterrows():
                if row.get("symbol") in selected_symbols:
                    continue
                selected.append(row)
                selected_symbols.add(row.get("symbol"))
                if len(selected) >= top_n:
                    break

        if not selected:
            return valid.head(top_n)
        return pd.DataFrame(selected).sort_values(self.config.score_column, ascending=False).head(top_n)

    def _calculate_turnover(self, previous_selection: set, current_selection: set) -> float:
        if not current_selection:
            return 0.0
        if not previous_selection:
            return 1.0
        overlap = len(previous_selection.intersection(current_selection))
        return max(0.0, min(1.0, 1 - overlap / len(current_selection)))

    def _is_unfilled(self, record: Any, columns: Any) -> bool:
        """True when a selected name never became a position at all.

        The dataset omits ``entry_price`` when the name could not be bought at the
        execution price (suspended, or a one-price limit up). A name that has an
        entry price but no period return was bought and is merely still open, so it
        keeps its exposure and is reported as an unrealized trade instead.
        """
        if "entry_price" not in columns:
            return True
        return self._to_float(record.get("entry_price")) is None

    def _base_target_weights(
        self,
        selected: pd.DataFrame,
        top_n: int,
        *,
        ret_col: Optional[str] = None,
    ) -> Dict[str, float]:
        """Volatility-free target weights used for turnover; unfilled names stay cash.

        An order that never filled moves no capital, so charging turnover (hence
        fees) on it would overstate trading.
        """
        n = len(selected)
        if n == 0:
            return {"__cash__": 1.0}
        denominator = max(n, int(top_n))
        weights: Dict[str, float] = {}
        for record in selected.to_dict(orient="records"):
            missing_return = (
                ret_col is not None and self._to_float(record.get(ret_col)) is None
            )
            if missing_return and self._is_unfilled(record, selected.columns):
                continue
            weights[str(record.get("symbol"))] = 1.0 / denominator
        weights["__cash__"] = max(0.0, 1.0 - sum(weights.values()))
        return weights

    @staticmethod
    def _calculate_weight_turnover(
        previous_weights: Dict[str, float],
        current_weights: Dict[str, float],
    ) -> float:
        assets = set(previous_weights).union(current_weights)
        return 0.5 * sum(
            abs(
                float(current_weights.get(asset, 0.0))
                - float(previous_weights.get(asset, 0.0))
            )
            for asset in assets
        )

    def _base_valid_columns(self, ret_col: str) -> List[str]:
        columns = [
            "symbol",
            "name",
            "industry",
            self.config.score_column,
            ret_col,
            "entry_price",
            "entry_raw_price",
            "entry_date",
            "entry_price_type",
            f"exit_price_{self.config.horizon}d",
            f"exit_raw_price_{self.config.horizon}d",
            f"exit_date_{self.config.horizon}d",
            f"exit_price_type_{self.config.horizon}d",
            f"exit_delayed_days_{self.config.horizon}d",
            f"unrealized_return_{self.config.horizon}d",
            f"mark_price_{self.config.horizon}d",
            f"mark_raw_price_{self.config.horizon}d",
            f"mark_date_{self.config.horizon}d",
            f"is_open_{self.config.horizon}d",
            f"exit_reason_{self.config.horizon}d",
            f"stop_trigger_date_{self.config.horizon}d",
        ]
        # Unconditional, unlike the volatility/path columns below. The sweep's
        # selection plan is cached process-wide and its key carries no regime
        # parameters, so a plan built by a scale=1.0 combo must still carry the
        # label -- otherwise a later scale=0 combo silently reuses a plan without it.
        columns.extend([SW_L1_CODE_COL, self.config.industry_regime_column])
        if self.config.intra_period_cash:
            horizon = self.config.horizon
            columns.extend(
                [
                    live_cash_field("exit_price", horizon),
                    live_cash_field("exit_raw_price", horizon),
                    live_cash_field("exit_date", horizon),
                    live_cash_field("exit_price_type", horizon),
                    live_cash_field("exit_delayed_days", horizon),
                    live_cash_field("exit_reason", horizon),
                ]
            )
        if (self.config.volatility_mode or "off").lower() != "off":
            columns.append(self.config.volatility_column)
        if (self.config.path_gate or "off").strip().lower() != "off":
            columns.extend(PATH_FEATURE_COLUMNS)
        return list(dict.fromkeys(col for col in columns if col in self.dataset.columns))

    def _volatility_scale(self, row: Dict[str, Any]) -> float:
        mode = (self.config.volatility_mode or "off").lower()
        target = self.config.target_vol_annual
        if mode == "off" or target is None or target <= 0:
            return 1.0
        vol = self._to_float(row.get(self.config.volatility_column))
        if vol is None or vol <= 0:
            return 1.0
        return min(1.0, float(target) / float(vol))

    def _horizon_field(self, name: str) -> str:
        horizon = self.config.horizon
        if self.config.intra_period_cash:
            return live_cash_field(name, horizon)
        return f"{name}_{horizon}d"

    @staticmethod
    def _cell_text(value: Any) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value)

    def _trailing_stop_pct(self) -> Optional[float]:
        return normalize_trailing_stop_ratio(self.config.trailing_stop_pct)

    def _period_performance(
        self,
        selected: pd.DataFrame,
        ret_col: str,
        turnover: float,
        *,
        score_date: str,
        top_n: int,
    ) -> Tuple[Dict[str, float], float, int, int, bool, pd.DataFrame]:
        crowd_k = int(self.config.crowd_exit_k or 0)
        if crowd_k > 0:
            adjusted, extra_turnover, flattened, triggered = self._with_crowd_exit_returns(
                selected,
                ret_col,
                score_date=score_date,
                crowd_k=crowd_k,
            )
            refill_count = 0
        else:
            refill_n = self._effective_select_n(score_date, top_n)
            adjusted, extra_turnover, refill_count = self._with_refill_returns(
                selected,
                ret_col,
                score_date=score_date,
                top_n=refill_n,
            )
            flattened = 0
            triggered = False
        holdings = self._weighted_portfolio_return(
            adjusted,
            ret_col,
            weight_n=self._weight_denominator(score_date, top_n, len(adjusted)),
        )
        fee_cost_rate = self.fee_model.period_cost_rate(float(turnover) + extra_turnover)
        return holdings, fee_cost_rate, refill_count, flattened, triggered, adjusted

    def _ranked_refill_symbols(self, score_date: str) -> List[str]:
        group = self._sorted_group_for_date(score_date, self.config.score_column)
        if group is None or group.empty:
            return []
        pool = self._score_valid_pool(group)
        if pool.empty or "symbol" not in pool.columns:
            return []
        return [str(symbol) for symbol in pool["symbol"].tolist()]

    @staticmethod
    def _period_end_idx(prices: Sequence[Dict[str, Any]], period_end_date: str) -> Optional[int]:
        direct = find_bar_index(prices, period_end_date)
        if direct is not None:
            return direct
        end = str(period_end_date)
        last: Optional[int] = None
        for idx, row in enumerate(prices):
            trade_date = str(row.get("trade_date") or "")
            if trade_date and trade_date <= end:
                last = idx
            elif trade_date > end:
                break
        return last

    def _simulate_refill_leg(
        self,
        symbol: str,
        entry_date: str,
        period_end_date: str,
    ) -> Optional[Dict[str, Any]]:
        prices = self._price_cache.get(symbol) or []
        entry_idx = find_bar_index(prices, entry_date)
        end_idx = self._period_end_idx(prices, period_end_date)
        if entry_idx is None or end_idx is None:
            return None
        remaining = int(end_idx) - int(entry_idx)
        if remaining < 1:
            return None
        cash_dates = list(self._cash_on_dates) if self.config.intra_period_cash else None
        result = simulate_open_holding(
            prices,
            entry_idx=entry_idx,
            horizon=remaining,
            trailing_stop_pct=self._trailing_stop_pct(),
            cash_exit_dates=cash_dates,
        )
        if result is None:
            return None
        payload = dict(result)
        payload["symbol"] = symbol
        return payload

    def _pick_refill_candidate(
        self,
        ranked_symbols: Sequence[str],
        used: set,
        entry_date: str,
        period_end_date: str,
    ) -> Optional[Dict[str, Any]]:
        for symbol in ranked_symbols:
            if not symbol or symbol in used:
                continue
            result = self._simulate_refill_leg(symbol, entry_date, period_end_date)
            if result is not None:
                return result
        return None

    def _slot_lookup(self, score_date: str, symbol: str, record: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(record)
        group = self._sorted_group_for_date(score_date, self.config.score_column)
        if group is None or group.empty or "symbol" not in group.columns:
            return merged
        match = group[group["symbol"].astype(str) == str(symbol)]
        if match.empty:
            return merged
        extra = match.iloc[0].to_dict()
        for key, value in extra.items():
            if key not in merged or merged.get(key) is None or self._cell_text(merged.get(key)) == "":
                merged[key] = value
        return merged

    def _with_refill_returns(
        self,
        selected: pd.DataFrame,
        ret_col: str,
        *,
        score_date: str,
        top_n: int,
    ) -> Tuple[pd.DataFrame, float, int]:
        """Compound trailing-stop slots with next unused names until the period end."""
        if not self.config.intra_period_refill or selected is None or selected.empty:
            return selected, 0.0, 0
        if self._trailing_stop_pct() is None or not self._price_cache:
            return selected, 0.0, 0
        reason_col = self._horizon_field("exit_reason")
        exit_date_col = self._horizon_field("exit_date")
        ranked = self._ranked_refill_symbols(score_date)
        if not ranked:
            return selected, 0.0, 0
        weights = self._compute_weights(selected)
        out = selected.copy()
        used = {
            str(symbol)
            for symbol in (out["symbol"].tolist() if "symbol" in out.columns else [])
            if str(symbol)
        }
        extra_turnover = 0.0
        refill_count = 0
        max_period = 3 * max(1, int(top_n))
        for idx, raw in out.iterrows():
            if refill_count >= max_period:
                break
            record = self._slot_lookup(score_date, str(raw.get("symbol") or ""), raw.to_dict())
            if self._cell_text(record.get(reason_col)) != "trailing_stop":
                continue
            original_return = self._to_float(record.get(ret_col))
            if original_return is None:
                continue
            symbol = str(record.get("symbol") or "")
            orig_prices = self._price_cache.get(symbol) or []
            entry_date = self._cell_text(record.get("entry_date"))
            exit_date = self._cell_text(record.get(exit_date_col))
            entry_idx = find_bar_index(orig_prices, entry_date)
            if entry_idx is None or not exit_date:
                continue
            period_end_idx = entry_idx + int(self.config.horizon)
            if period_end_idx >= len(orig_prices):
                continue
            period_end_date = str(orig_prices[period_end_idx].get("trade_date") or "")
            if not period_end_date or exit_date >= period_end_date:
                continue
            slot_factor = 1.0 + float(original_return)
            next_entry = exit_date
            next_reason = "trailing_stop"
            slot_refills = 0
            slot_weight = float(weights.get(symbol, 0.0))
            while (
                next_reason == "trailing_stop"
                and slot_refills < _MAX_REFILLS_PER_SLOT
                and refill_count < max_period
            ):
                replacement = self._pick_refill_candidate(
                    ranked,
                    used,
                    next_entry,
                    period_end_date,
                )
                if replacement is None:
                    break
                holding_return = self._to_float(replacement.get("holding_return"))
                if holding_return is None:
                    break
                slot_factor *= 1.0 + float(holding_return)
                extra_turnover += slot_weight
                slot_refills += 1
                refill_count += 1
                used.add(str(replacement.get("symbol") or ""))
                next_reason = self._cell_text(replacement.get("exit_reason"))
                next_entry = self._cell_text(replacement.get("exit_date"))
                if not next_entry:
                    break
            if slot_refills:
                out.at[idx, ret_col] = slot_factor - 1.0
        return out, extra_turnover, refill_count

    def _with_crowd_exit_returns(
        self,
        selected: pd.DataFrame,
        ret_col: str,
        *,
        score_date: str,
        crowd_k: int,
    ) -> Tuple[pd.DataFrame, float, int, bool]:
        """If k names hit trailing stop, flatten the rest at that signal's next open."""
        if crowd_k <= 0 or selected is None or selected.empty:
            return selected, 0.0, 0, False
        if not self._price_cache:
            raise ValueError("crowd_exit_k requires a price_cache of daily bars")
        reason_col = self._horizon_field("exit_reason")
        exit_date_col = self._horizon_field("exit_date")
        trigger_col = f"stop_trigger_date_{self.config.horizon}d"
        events: List[Tuple[str, str, Any]] = []
        records: Dict[Any, Dict[str, Any]] = {}
        for idx, raw in selected.iterrows():
            record = self._slot_lookup(score_date, str(raw.get("symbol") or ""), raw.to_dict())
            records[idx] = record
            if self._cell_text(record.get(reason_col)) != "trailing_stop":
                continue
            trigger = self._cell_text(record.get(trigger_col)) or self._cell_text(
                record.get(exit_date_col)
            )
            if not trigger:
                continue
            events.append((trigger, str(record.get("symbol") or ""), idx))
        events.sort(key=lambda item: (item[0], item[1]))
        if len(events) < crowd_k:
            return selected, 0.0, 0, False
        flatten_trigger = events[crowd_k - 1][0]
        out = selected.copy()
        flattened = 0
        extra_turnover = 0.0
        weights = self._compute_weights(out)
        cash_dates = list(self._cash_on_dates) if self.config.intra_period_cash else None
        for idx, raw in out.iterrows():
            record = records.get(idx) or raw.to_dict()
            reason = self._cell_text(record.get(reason_col))
            trigger = self._cell_text(record.get(trigger_col)) or self._cell_text(
                record.get(exit_date_col)
            )
            if reason == "trailing_stop" and trigger and trigger <= flatten_trigger:
                continue
            symbol = str(record.get("symbol") or "")
            prices = self._price_cache.get(symbol) or []
            entry_date = self._cell_text(record.get("entry_date"))
            entry_idx = find_bar_index(prices, entry_date)
            trigger_idx = find_bar_index(prices, flatten_trigger)
            if entry_idx is None or trigger_idx is None:
                continue
            exit_target = int(trigger_idx) + 1
            remaining = exit_target - int(entry_idx)
            if remaining < 1 or exit_target >= len(prices):
                continue
            flatten_exit_date = str(prices[exit_target].get("trade_date") or "")
            orig_exit = self._cell_text(record.get(exit_date_col))
            if orig_exit and flatten_exit_date and orig_exit <= flatten_exit_date:
                continue
            result = simulate_open_holding(
                prices,
                entry_idx=entry_idx,
                horizon=remaining,
                trailing_stop_pct=None,
                cash_exit_dates=cash_dates,
            )
            holding = None if result is None else self._to_float(result.get("holding_return"))
            if holding is None:
                continue
            self._write_simulated_exit(out, idx, ret_col, result, exit_reason="crowd_exit")
            flattened += 1
            extra_turnover += float(weights.get(symbol, 0.0))
        return out, extra_turnover, flattened, True

    def _write_simulated_exit(
        self,
        frame: pd.DataFrame,
        idx: Any,
        ret_col: str,
        result: Dict[str, Any],
        *,
        exit_reason: str,
    ) -> None:
        holding = self._to_float(result.get("holding_return"))
        if holding is None:
            return
        frame.at[idx, ret_col] = holding
        frame.at[idx, self._horizon_field("exit_date")] = result.get("exit_date")
        exit_price = self._to_float(result.get("exit_price"))
        if exit_price is not None:
            frame.at[idx, self._horizon_field("exit_price")] = exit_price
        raw_exit_price = self._to_float(result.get("exit_raw_price"))
        if raw_exit_price is not None:
            frame.at[idx, self._horizon_field("exit_raw_price")] = raw_exit_price
        frame.at[idx, self._horizon_field("exit_reason")] = exit_reason
        frame.at[idx, self._horizon_field("exit_price_type")] = (
            result.get("exit_price_type") or "open"
        )

    def _apply_bull_exposure(self, weights: Dict[str, float]) -> Dict[str, float]:
        exposure = float(self.config.bull_exposure)
        if exposure == 1.0 or not weights:
            return weights
        return {symbol: weight * exposure for symbol, weight in weights.items()}

    def _industry_regime_factor(self, record: Optional[Dict[str, Any]]) -> float:
        """Weight multiplier for one name from its own SW L1 industry regime label.

        A missing label (unmapped industry, empty cell, NaN) falls back to 1.0: an
        unmapped name must not be treated as non-bull, or this axis would silently
        degrade into a mapping-coverage test.
        """
        scale = float(self.config.industry_regime_scale)
        if scale >= 1.0 or record is None:
            return 1.0
        raw = record.get(self.config.industry_regime_column)
        if raw is None or pd.isna(raw):
            return 1.0
        label = str(raw).strip().lower()
        if not label or label == "bull":
            return 1.0
        return scale

    def _apply_industry_regime_scale(
        self,
        weights: Dict[str, float],
        records_by_symbol: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        """Scale each name by its industry regime; the freed weight stays in cash.

        Deliberately applied *after* the volatility branches rather than folded
        into ``scales``: ``volatility_mode="off"`` never consults ``scales``, so
        folding would skip that branch. No renormalisation happens downstream, so
        scaled-down weight becomes cash exactly like ``bull_exposure``.
        """
        if float(self.config.industry_regime_scale) >= 1.0 or not weights:
            return weights
        return {
            symbol: weight * self._industry_regime_factor(records_by_symbol.get(symbol))
            for symbol, weight in weights.items()
        }

    def _compute_weights(
        self,
        selected: pd.DataFrame,
        *,
        weight_n: Optional[int] = None,
    ) -> Dict[str, float]:
        n = len(selected)
        if n == 0:
            return {}
        denom = max(n, int(weight_n)) if weight_n is not None else n
        mode = (self.config.volatility_mode or "off").lower()
        records = selected.to_dict(orient="records")
        symbols = [str(row.get("symbol")) for row in records]
        scales = {symbol: self._volatility_scale(row) for symbol, row in zip(symbols, records)}
        records_by_symbol = dict(zip(symbols, records))
        if self._is_dynamic_selection():
            top_n = int(tuple(self.config.top_n_values)[0])
            base_weight = 1.0 / max(n, top_n)
            base_exposure = min(1.0, n / top_n)
            if mode == "off":
                weights = {symbol: base_weight for symbol in symbols}
            elif mode == "cash_drag":
                weights = {symbol: base_weight * scales[symbol] for symbol in symbols}
            elif mode == "renormalize":
                total_scale = sum(scales.values())
                if total_scale <= 0:
                    weights = {symbol: base_weight for symbol in symbols}
                else:
                    weights = {
                        symbol: base_exposure * scales[symbol] / total_scale
                        for symbol in symbols
                    }
            else:
                raise ValueError(f"Unsupported volatility_mode: {mode}")
            return self._apply_industry_regime_scale(
                self._apply_bull_exposure(weights), records_by_symbol
            )
        invested_frac = n / float(denom)
        if mode == "off":
            weight = 1.0 / denom
            weights = {symbol: weight for symbol in symbols}
        elif mode == "cash_drag":
            base = 1.0 / denom
            weights = {symbol: base * scales[symbol] for symbol in symbols}
        elif mode == "renormalize":
            total_scale = sum(scales.values())
            if total_scale <= 0:
                weight = 1.0 / denom
                weights = {symbol: weight for symbol in symbols}
            else:
                weights = {
                    symbol: invested_frac * scales[symbol] / total_scale
                    for symbol in symbols
                }
        else:
            raise ValueError(f"Unsupported volatility_mode: {mode}")
        return self._apply_industry_regime_scale(
            self._apply_bull_exposure(weights), records_by_symbol
        )

    def _weighted_portfolio_return(
        self,
        selected: pd.DataFrame,
        ret_col: str,
        *,
        weight_n: Optional[int] = None,
    ) -> Dict[str, float]:
        """Split target weights into invested exposure and weight left in cash.

        A name whose order never filled holds no market risk, so its target weight
        is reported as cash rather than as invested exposure earning 0%.
        """
        weights = self._compute_weights(selected, weight_n=weight_n)
        gross_return = 0.0
        exposure = 0.0
        unfilled_weight = 0.0
        unfilled_count = 0
        for symbol, weight in weights.items():
            row = selected[selected["symbol"].astype(str) == symbol]
            record = row.iloc[0] if not row.empty else None
            holding_return = self._to_float(record.get(ret_col)) if record is not None else None
            if holding_return is None:
                if record is None or self._is_unfilled(record, selected.columns):
                    unfilled_weight += weight
                    unfilled_count += 1
                    continue
                exposure += weight
                continue
            gross_return += weight * holding_return
            exposure += weight
        return {
            "gross_return": gross_return,
            "exposure": exposure,
            "unfilled_weight": unfilled_weight,
            "unfilled_count": unfilled_count,
        }

    def _build_trade_rows(
        self,
        score_date: str,
        top_n: int,
        selected: pd.DataFrame,
        ret_col: str,
        turnover: float,
        *,
        equity: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        capital = float(self.config.initial_capital if equity is None else equity)
        entry_col = "entry_price"
        horizon = self.config.horizon
        if self.config.intra_period_cash:
            exit_price_col = live_cash_field("exit_price", horizon)
            exit_date_col = live_cash_field("exit_date", horizon)
            exit_price_type_col = live_cash_field("exit_price_type", horizon)
            exit_delayed_days_col = live_cash_field("exit_delayed_days", horizon)
        else:
            exit_price_col = f"exit_price_{horizon}d"
            exit_date_col = f"exit_date_{horizon}d"
            exit_price_type_col = f"exit_price_type_{horizon}d"
            exit_delayed_days_col = f"exit_delayed_days_{horizon}d"
        if entry_col not in selected.columns or exit_price_col not in selected.columns:
            return []

        weights = self._compute_weights(
            selected,
            weight_n=self._weight_denominator(str(score_date), top_n, len(selected)),
        )
        selected = selected.dropna(subset=[entry_col, exit_price_col]).copy()
        if selected.empty:
            return []

        rows: List[Dict[str, Any]] = []
        for _, stock in selected.iterrows():
            symbol = str(stock.get("symbol"))
            target_position_amount = capital * float(weights.get(symbol, 0.0))
            if target_position_amount <= 0:
                continue
            buy_price_adjusted = self._to_float(stock.get(entry_col))
            sell_price_adjusted = self._to_float(stock.get(exit_price_col))
            buy_price = self._raw_or_adjusted_price(
                stock, "entry_raw_price", entry_col
            )
            sell_price = self._raw_or_adjusted_price(
                stock, exit_price_col.replace("exit_price", "exit_raw_price", 1), exit_price_col
            )
            if buy_price is None or buy_price <= 0 or sell_price is None:
                continue
            quantity = self._round_lot_quantity(target_position_amount, buy_price)
            if quantity <= 0:
                continue
            buy_amount = quantity * buy_price
            sell_amount = quantity * sell_price
            gross_pnl = sell_amount - buy_amount
            fees = self.fee_model.estimate_round_trip(buy_amount, sell_amount)
            estimated_cost = fees["total_fee"]
            net_pnl = gross_pnl - estimated_cost
            holding_return = self._to_float(stock.get(ret_col))
            rows.append(
                {
                    "score_date": score_date,
                    "top_n": top_n,
                    "score_column": self.config.score_column,
                    "selection_mode": self._selection_mode(),
                    "threshold_lookback_days": int(self.config.threshold_lookback_days),
                    "max_positions": int(self.config.max_positions),
                    "construction_mode": self.config.construction_mode,
                    "horizon": self.config.horizon,
                    "symbol": stock.get("symbol"),
                    "name": stock.get("name", ""),
                    "buy_date": stock.get("entry_date") or score_date,
                    "buy_price_type": stock.get("entry_price_type"),
                    "sell_date": stock.get(exit_date_col),
                    "sell_price_type": stock.get(exit_price_type_col),
                    "sell_delayed_days": stock.get(exit_delayed_days_col),
                    "target_position_amount": target_position_amount,
                    "quantity": quantity,
                    "buy_price": buy_price,
                    "buy_price_adjusted": buy_price_adjusted,
                    "sell_price": sell_price,
                    "sell_price_adjusted": sell_price_adjusted,
                    "buy_amount": buy_amount,
                    "sell_amount": sell_amount,
                    "uninvested_cash": target_position_amount - buy_amount,
                    "actual_weight": buy_amount / capital if capital else None,
                    "holding_return": holding_return,
                    "gross_pnl": gross_pnl,
                    "estimated_transaction_cost": estimated_cost,
                    "buy_fee": fees["buy_fee"],
                    "sell_fee": fees["sell_fee"],
                    "commission": fees["commission"],
                    "stamp_tax": fees["stamp_tax"],
                    "transfer_fee": fees["transfer_fee"],
                    "net_pnl": net_pnl,
                    "score_value": self._to_float(stock.get(self.config.score_column)),
                    "status": "closed",
                    "is_unrealized": False,
                }
            )
        return rows

    def _build_open_trade_rows(
        self,
        score_date: str,
        top_n: int,
        group: pd.DataFrame,
        *,
        selected: Optional[pd.DataFrame] = None,
        equity: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Emit MTM rows for names that have an entry but no completed exit.

        ``selected`` is the period's actual book (after refill / crowd overlay).
        When it is set, only those names can appear, at their book weights.
        When it is None the date has no completed forwards, so a book is chosen
        from the mark-only pool (last rebalance of a backtest).
        """
        capital = float(self.config.initial_capital if equity is None else equity)
        horizon = self.config.horizon
        entry_col = "entry_price"
        exit_price_col = f"exit_price_{horizon}d"
        mark_price_col = f"mark_price_{horizon}d"
        mark_date_col = f"mark_date_{horizon}d"
        unreal_col = f"unrealized_return_{horizon}d"
        if entry_col not in group.columns or mark_price_col not in group.columns:
            return []

        if selected is None:
            open_only = group.dropna(
                subset=[
                    "symbol",
                    self.config.score_column,
                    entry_col,
                    mark_price_col,
                ]
            ).copy()
            if exit_price_col in open_only.columns:
                open_only = open_only[open_only[exit_price_col].isna()]
            if open_only.empty:
                return []

            path_gate = (self.config.path_gate or "off").strip().lower()
            gated = apply_path_gate(open_only, path_gate)
            if gated.empty:
                return []
            selected = self._select_portfolio(
                gated.sort_values(self.config.score_column, ascending=False),
                top_n,
            )
            weights = self._compute_weights(
                selected,
                weight_n=self._weight_denominator(str(score_date), top_n, len(selected)),
            )
        else:
            weights = self._compute_weights(
                selected,
                weight_n=self._weight_denominator(str(score_date), top_n, len(selected)),
            )
            selected = selected.dropna(
                subset=["symbol", self.config.score_column, entry_col, mark_price_col]
            ).copy()
            if exit_price_col in selected.columns:
                selected = selected[selected[exit_price_col].isna()]
        if selected.empty:
            return []

        rows: List[Dict[str, Any]] = []
        for _, stock in selected.iterrows():
            symbol = str(stock.get("symbol"))
            target_position_amount = capital * float(weights.get(symbol, 0.0))
            if target_position_amount <= 0:
                continue
            buy_price_adjusted = self._to_float(stock.get(entry_col))
            mark_price_adjusted = self._to_float(stock.get(mark_price_col))
            buy_price = self._raw_or_adjusted_price(stock, "entry_raw_price", entry_col)
            mark_price = self._raw_or_adjusted_price(
                stock,
                mark_price_col.replace("mark_price", "mark_raw_price", 1),
                mark_price_col,
            )
            if buy_price is None or buy_price <= 0 or mark_price is None:
                continue
            quantity = self._round_lot_quantity(target_position_amount, buy_price)
            if quantity <= 0:
                continue
            buy_amount = quantity * buy_price
            mark_amount = quantity * mark_price
            gross_pnl = mark_amount - buy_amount
            fees = self.fee_model.estimate_round_trip(buy_amount, mark_amount)
            estimated_cost = fees["total_fee"]
            net_pnl = gross_pnl - estimated_cost
            holding_return = self._to_float(stock.get(unreal_col))
            rows.append(
                {
                    "score_date": score_date,
                    "top_n": top_n,
                    "score_column": self.config.score_column,
                    "selection_mode": self._selection_mode(),
                    "threshold_lookback_days": int(self.config.threshold_lookback_days),
                    "max_positions": int(self.config.max_positions),
                    "construction_mode": self.config.construction_mode,
                    "horizon": horizon,
                    "symbol": stock.get("symbol"),
                    "name": stock.get("name", ""),
                    "buy_date": stock.get("entry_date") or score_date,
                    "buy_price_type": stock.get("entry_price_type"),
                    "sell_date": None,
                    "sell_price_type": None,
                    "sell_delayed_days": None,
                    "mark_date": stock.get(mark_date_col),
                    "mark_price": mark_price,
                    "mark_price_adjusted": mark_price_adjusted,
                    "target_position_amount": target_position_amount,
                    "quantity": quantity,
                    "buy_price": buy_price,
                    "buy_price_adjusted": buy_price_adjusted,
                    "sell_price": None,
                    "buy_amount": buy_amount,
                    "sell_amount": None,
                    "uninvested_cash": target_position_amount - buy_amount,
                    "actual_weight": buy_amount / capital if capital else None,
                    "holding_return": holding_return,
                    "gross_pnl": gross_pnl,
                    "estimated_transaction_cost": estimated_cost,
                    "buy_fee": fees["buy_fee"],
                    "sell_fee": fees["sell_fee"],
                    "commission": fees["commission"],
                    "stamp_tax": fees["stamp_tax"],
                    "transfer_fee": fees["transfer_fee"],
                    "net_pnl": net_pnl,
                    "score_value": self._to_float(stock.get(self.config.score_column)),
                    "status": "open",
                    "is_unrealized": True,
                }
            )
        return rows

    def _round_lot_quantity(self, target_amount: float, price: float) -> int:
        lot_size = max(1, int(self.config.lot_size))
        raw_quantity = int(target_amount // price)
        return (raw_quantity // lot_size) * lot_size

    def _periods_per_year(self) -> float:
        interval_days = self.config.rebalance_interval_days
        if interval_days is None:
            interval_days = int(self.config.rebalance_step or 1)
        return 252 / max(1, int(interval_days))

    def _sharpe(self, returns: pd.Series) -> Optional[float]:
        if len(returns) <= 1:
            return None
        std = returns.std(ddof=1)
        if std == 0 or pd.isna(std):
            return None
        return float((returns.mean() / std) * math.sqrt(self._periods_per_year()))

    def _sortino(self, returns: pd.Series) -> Optional[float]:
        if len(returns) <= 1:
            return None
        downside = returns.clip(upper=0.0)
        downside_dev = math.sqrt(float((downside ** 2).mean()))
        if downside_dev == 0 or pd.isna(downside_dev):
            return None
        return float((returns.mean() / downside_dev) * math.sqrt(self._periods_per_year()))

    def _information_ratio(self, excess_returns: pd.Series) -> Optional[float]:
        if len(excess_returns) <= 1:
            return None
        std = excess_returns.std(ddof=1)
        if std == 0 or pd.isna(std):
            return None
        return float((excess_returns.mean() / std) * math.sqrt(self._periods_per_year()))

    def _max_drawdown(self, equity: pd.Series) -> Optional[float]:
        if equity.empty:
            return None
        running_max = equity.cummax()
        drawdown = equity / running_max - 1
        return float(drawdown.min())

    def _select_rebalance_dates(self, score_dates: Sequence[str]) -> List[str]:
        score_dates = sorted(str(date) for date in score_dates if date)
        interval_days = self.config.rebalance_interval_days
        if interval_days is None:
            return score_dates[:: max(1, int(self.config.rebalance_step or 1))]
        if not score_dates:
            return []

        calendar = self.trading_calendar
        if not calendar and self.index_price_history:
            calendar = [row["trade_date"] for row in self.index_price_history]
        if not calendar:
            calendar = score_dates

        selected = []
        last_position: Optional[int] = None
        cursor = 0
        for score_date in score_dates:
            while cursor < len(calendar) and calendar[cursor] < score_date:
                cursor += 1
            if cursor >= len(calendar):
                break
            position = cursor
            if last_position is None or position - last_position >= int(interval_days):
                selected.append(score_date)
                last_position = position
        return selected

    def _calculate_index_benchmark_return(
        self,
        score_date: str,
        *,
        until_date: Optional[str] = None,
    ) -> Optional[float]:
        if not self.index_price_history:
            return None

        future_prices = [row for row in self.index_price_history if row.get("trade_date") and row["trade_date"] >= score_date]
        if not future_prices:
            return None
        base_close = self._to_float(future_prices[0].get("close"))
        if until_date:
            end_rows = [row for row in future_prices if row["trade_date"] <= str(until_date)]
            if len(end_rows) < 2:
                return None
            future_close = self._to_float(end_rows[-1].get("close"))
        else:
            if len(future_prices) <= self.config.horizon:
                return None
            future_close = self._to_float(future_prices[self.config.horizon].get("close"))
        if base_close is None or base_close <= 0 or future_close is None:
            return None
        return future_close / base_close - 1

    @classmethod
    def _clean_price_history(cls, price_history: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned = []
        seen_dates = set()
        for row in price_history:
            trade_date = cls._normalize_date(row.get("trade_date") or row.get("date"))
            close = cls._to_float(row.get("close") or row.get("close_price") or row.get("adj_close"))
            if not trade_date or close is None or close <= 0 or trade_date in seen_dates:
                continue
            seen_dates.add(trade_date)
            cleaned.append({"trade_date": trade_date, "close": close})
        return sorted(cleaned, key=lambda row: row["trade_date"])

    @classmethod
    def _clean_trading_calendar(cls, trading_calendar: Optional[Sequence[str]]) -> List[str]:
        if not trading_calendar:
            return []
        dates = []
        seen = set()
        for value in trading_calendar:
            date = cls._normalize_date(value)
            if date and date not in seen:
                dates.append(date)
                seen.add(date)
        return sorted(dates)

    @staticmethod
    def _normalize_date(value: Any) -> Optional[str]:
        if value is None or value == "":
            return None
        text = str(value).replace("-", "").replace("/", "")[:8]
        return text if len(text) == 8 and text.isdigit() else None

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if pd.isna(numeric):
            return None
        return numeric

    def _raw_or_adjusted_price(
        self, stock: Dict[str, Any], raw_col: str, adjusted_col: str
    ) -> Optional[float]:
        """Executable price for lots/cash, falling back for pre-adjustment data."""
        raw = self._to_float(stock.get(raw_col))
        if raw is not None and raw > 0:
            return raw
        return self._to_float(stock.get(adjusted_col))

    def _return_col(self, horizon: int) -> str:
        if self.config.intra_period_cash:
            return live_cash_field("forward_return", horizon)
        return f"forward_return_{horizon}d"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an equal-weight top-N portfolio backtest from stock scores.")
    parser.add_argument("--start-date", help="Inclusive score_date lower bound, YYYYMMDD")
    parser.add_argument("--end-date", help="Inclusive score_date upper bound, YYYYMMDD")
    parser.add_argument("--universe-index", nargs="+", default=[], help="Index universe loaded from index_constituents, e.g. csi1000 or csi2000")
    parser.add_argument("--index-codes", nargs="+", default=[], help="Filter records by stock_scores.index_codes")
    parser.add_argument("--score-column", default="composite_cycle_oriented_score", help="Score column used for ranking")
    parser.add_argument("--horizon", type=int, default=20, help="Forward-return horizon in trading days")
    parser.add_argument("--top-n", nargs="+", type=int, default=[50, 100], help="Top-N portfolio sizes to test")
    parser.add_argument(
        "--selection-mode",
        choices=["fixed_top_n", "dynamic_score_threshold"],
        default="fixed_top_n",
        help="Fixed Top-N or trailing score-threshold selection",
    )
    parser.add_argument(
        "--threshold-lookback-days",
        type=int,
        default=10,
        help="Historical score dates used by dynamic score thresholds",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=20,
        help="Maximum positions selected in dynamic threshold mode",
    )
    parser.add_argument("--rebalance-interval-days", type=int, default=20, help="Minimum trading-day interval between rebalance dates")
    parser.add_argument("--rebalance-step", type=int, help="OBSOLETE: use every Nth score_date as a rebalance date")
    parser.add_argument("--transaction-cost", type=float, default=0.001, help="Legacy one-way cost applied to turnover")
    parser.add_argument("--buy-commission-rate", type=float, help="Buy-side commission rate, e.g. 0.0001")
    parser.add_argument("--sell-commission-rate", type=float, help="Sell-side commission rate, e.g. 0.0001")
    parser.add_argument("--min-commission", type=float, help="Minimum commission per order")
    parser.add_argument("--stamp-tax-rate", type=float, help="Sell-side stamp tax rate")
    parser.add_argument("--transfer-fee-rate", type=float, help="Per-side transfer fee rate")
    parser.add_argument("--index-benchmark-symbol", help="Optional index symbol for market benchmark returns, e.g. 000852.SH")
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0, help="Notional capital for trade detail calculations")
    parser.add_argument("--include-trades", action="store_true", help="Include per-position trade details in JSON output")
    parser.add_argument("--lot-size", type=int, default=100, help="Minimum tradable share lot size")
    parser.add_argument(
        "--construction-mode",
        choices=["top_n", "industry_capped", "industry_neutral"],
        default="top_n",
        help="Portfolio construction mode",
    )
    parser.add_argument("--max-industry-weight", type=float, default=0.3, help="Max single-industry weight for industry_capped mode")
    parser.add_argument(
        "--industry-l1",
        nargs="+",
        default=[],
        help="Shenwan L1 filter (code or Chinese name)",
    )
    parser.add_argument(
        "--trailing-stop-pct",
        type=float,
        help="Trailing stop from peak high: close <= peak*(1-pct) exits next open (disabled when omitted or <=0)",
    )
    parser.add_argument(
        "--bull-exposure",
        type=float,
        default=1.0,
        help="Invested fraction during labeled-bull periods in (0, 1]; default 1.0 (full book)",
    )
    parser.add_argument(
        "--intra-period-refill",
        action="store_true",
        help="After a trailing stop, refill the slot from the same-date ranking before the next rebalance (research opt-in)",
    )
    parser.add_argument(
        "--crowd-exit-k",
        type=int,
        default=0,
        help="If this many names hit trailing stop in a period, flatten the rest to cash (0=off, research opt-in)",
    )
    parser.add_argument(
        "--non-bull-top-n-scale",
        type=float,
        default=0.0,
        help="On cash_on_dates, hold this fraction of TopN instead of cash (0=cash, 0.5=TopN/2). Research opt-in",
    )
    parser.add_argument(
        "--non-bull-keep-parent-slots",
        action="store_true",
        help="With --non-bull-top-n-scale, keep original 1/TopN weights (50%% cash if scale=0.5). Default renormalizes to 100%%",
    )
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument("--enable-tushare", action="store_true", help="Keep TUSHARE_TOKEN enabled when creating ResearchDataAccess")
    return parser.parse_args()


def main() -> int:
    from shared.data_access import open_research_data_access_for_cli

    args = _parse_args()
    db_manager = open_research_data_access_for_cli(enable_tushare=args.enable_tushare)
    catalog = sw_l1_catalog(db_manager)
    industry_l1_codes: tuple[str, ...] = ()
    if args.industry_l1:
        industry_l1_codes = tuple(resolve_industry_l1_codes(args.industry_l1, catalog))
    trailing_stop_pct = args.trailing_stop_pct if args.trailing_stop_pct and args.trailing_stop_pct > 0 else None
    score_config = BacktestConfig(
        horizons=(args.horizon,),
        start_date=args.start_date,
        end_date=args.end_date,
        index_codes=tuple(args.index_codes),
        universe_index_codes=tuple(args.universe_index),
        trailing_stop_pct=trailing_stop_pct,
    )
    score_backtester = ScoreBacktester(db_manager, score_config)
    dataset = score_backtester.build_dataset()
    if not dataset.empty:
        dataset = attach_sw_l1(dataset, db_manager, as_of_column="score_date")
    benchmark_symbol = args.index_benchmark_symbol
    if len(industry_l1_codes) == 1 and not benchmark_symbol:
        benchmark_symbol = industry_l1_codes[0]
    portfolio_config = PortfolioBacktestConfig(
        score_column=args.score_column,
        horizon=args.horizon,
        top_n_values=tuple(args.top_n),
        selection_mode=args.selection_mode,
        threshold_lookback_days=args.threshold_lookback_days,
        max_positions=args.max_positions,
        rebalance_interval_days=None if args.rebalance_step is not None else args.rebalance_interval_days,
        rebalance_step=args.rebalance_step,
        transaction_cost=args.transaction_cost,
        buy_commission_rate=args.buy_commission_rate,
        sell_commission_rate=args.sell_commission_rate,
        min_commission=args.min_commission,
        stamp_tax_rate=args.stamp_tax_rate,
        transfer_fee_rate=args.transfer_fee_rate,
        index_benchmark_symbol=benchmark_symbol,
        initial_capital=args.initial_capital,
        include_trades=args.include_trades,
        lot_size=args.lot_size,
        construction_mode=args.construction_mode if not industry_l1_codes else "top_n",
        max_industry_weight=args.max_industry_weight,
        industry_l1_codes=industry_l1_codes,
        bull_exposure=float(args.bull_exposure),
        intra_period_refill=bool(args.intra_period_refill),
        crowd_exit_k=int(args.crowd_exit_k or 0),
        trailing_stop_pct=trailing_stop_pct,
        non_bull_top_n_scale=float(args.non_bull_top_n_scale or 0.0),
        non_bull_keep_parent_slots=bool(args.non_bull_keep_parent_slots),
    )
    index_price_history = []
    if benchmark_symbol:
        index_price_history = load_index_benchmark_prices(db_manager, benchmark_symbol, args.start_date)
    result = PortfolioBacktester(
        dataset,
        portfolio_config,
        index_price_history=index_price_history,
        trading_calendar=[row.get("trade_date") for row in index_price_history],
        price_cache=score_backtester.price_cache
        if args.intra_period_refill or int(args.crowd_exit_k or 0) > 0
        else None,
    ).run()
    payload = result.to_dict()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
