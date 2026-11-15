#!/usr/bin/env python3
"""Sweep growth-cycle portfolio construction parameters and report results."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path
from statistics import pstdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from analysis.core.benchmark_prices import load_index_benchmark_prices
from shared.industry_universe import (
    attach_sw_l1,
    resolve_industry_l1_codes,
    sw_l1_catalog,
)
from analysis.core.industry_regime import attach_sw_l1_regime, sw_l1_regime_coverage
from analysis.core.path_features import add_path_feature_columns
from analysis.core.portfolio_backtester import PortfolioBacktestConfig, PortfolioBacktester
from shared.regime_label import (
    DEFAULT_UNCERTAIN_GROWTH_WEIGHT,
    REGIME_MODE_BULL_G60_ELSE_CASH,
    REGIME_SCORE_COLUMN,
    REGIME_SCORE_VARIANT,
    build_regime_cash_schedule,
    build_regime_weight_schedule,
    load_regime_rule,
    padded_index_start,
)
from analysis.core.score_backtester import BacktestConfig, ScoreBacktester
from analysis.core.benchmark_metrics import calculate_index_buy_hold_return
from analysis.core.html_format import as_float as _as_float
from analysis.core.html_format import fmt_num as _fmt_num
from analysis.core.html_format import fmt_pct as _fmt_pct
from analysis.core.html_format import html_document as _html_document
from analysis.core.industry_exposure import build_industry_exposure
from shared.universe_benchmark import (
    default_index_benchmark_symbol,
    universe_slug,
    universe_title,
)
from analysis.sweep.params import (
    normalize_bull_exposures,
    normalize_crowd_exit_ks,
    normalize_non_bull_actions,
    normalize_path_gates,
    normalize_refill_modes,
    normalize_industry_regime_scales,
    normalize_regime_modes,
    normalize_regime_rules,
    normalize_trailing_stop_pcts,
    count_regime_mode_arms,
    resolve_regime_modes,
    resolve_regime_rules,
    NON_BULL_ACTION_CASH,
)
from analysis.sweep.score_columns import (
    ScoreSweep,
    add_growth_cycle_weight_columns,
    build_score_sweeps,
    overlay_cycle_score,
)
from analysis.sweep.sweep_engine import (
    SweepCheckpoint,
    SweepOuterJob,
    combo_param_hash,
    merge_volatility_columns,
    run_outer_jobs,
    sweep_worker_count,
)
from analysis.sweep.sweep_result_view import (
    attach_sweep_view,
    format_axis_value,
    infer_sweep_axes,
    stability_group_columns,
    _row_matches_axis,
)

DEFAULT_TOP_N_VALUES = (10, 20, 50, 100)
DEFAULT_ACTIVE_CAPS = (0.2, 0.25, 0.3, 0.4)
DEFAULT_TARGET_VOLS = (0.30, 0.40, 0.50)


@dataclass(frozen=True)
class SweepVariant:
    name: str
    construction_mode: str
    max_industry_weight: Optional[float] = None



@dataclass(frozen=True)
class VolatilityVariant:
    name: str
    mode: str
    target_vol: Optional[float] = None


def build_volatility_variants(
    modes: Optional[Sequence[str]] = None,
    target_vols: Optional[Sequence[float]] = None,
) -> List[VolatilityVariant]:
    """Build volatility sweep variants. ``modes=None`` returns only ``off`` (backward compatible)."""
    if modes is None:
        return [VolatilityVariant("vol_off", "off", None)]

    modes_list = [str(mode).strip().lower() for mode in modes if str(mode).strip()]
    if not modes_list:
        return [VolatilityVariant("vol_off", "off", None)]

    target_vols_list = [float(value) for value in (target_vols or DEFAULT_TARGET_VOLS)]
    variants: List[VolatilityVariant] = []
    seen = set()
    for mode in modes_list:
        if mode == "off":
            key = ("off", None)
            if key not in seen:
                seen.add(key)
                variants.append(VolatilityVariant("vol_off", "off", None))
        elif mode in ("cash_drag", "renormalize"):
            for target_vol in target_vols_list:
                key = (mode, target_vol)
                if key in seen:
                    continue
                seen.add(key)
                pct = int(round(target_vol * 100))
                variants.append(VolatilityVariant(f"vol_{mode}_{pct}", mode, target_vol))
        else:
            raise ValueError(f"Unsupported volatility_mode: {mode}")
    return variants


def build_sweep_variants(active_caps: Sequence[float]) -> List[SweepVariant]:
    variants = [SweepVariant("growth_cycle_topn", "top_n")]
    variants.extend(
        SweepVariant(f"growth_cycle_active_cap{int(round(cap * 100))}", "industry_capped", float(cap))
        for cap in active_caps
    )
    variants.append(SweepVariant("growth_cycle_neutral", "industry_neutral"))
    return variants


def build_sweep_variants_for_request(
    active_caps: Sequence[float],
    industry_l1_codes: Sequence[str] = (),
) -> List[SweepVariant]:
    if industry_l1_codes:
        return [SweepVariant("growth_cycle_topn", "top_n")]
    return build_sweep_variants(active_caps)


def validate_rebalance_intervals(horizon: int, rebalance_intervals: Optional[Sequence[int]]) -> List[int]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not rebalance_intervals:
        return [int(horizon)]

    intervals = sorted(set(int(value) for value in rebalance_intervals))
    invalid = [value for value in intervals if value != int(horizon)]
    if invalid:
        raise ValueError(
            "Portfolio sweep uses tradable holding-period windows, so "
            f"rebalance_interval_days must equal horizon={horizon}. "
            f"Invalid values: {invalid}"
        )
    return intervals


def resolve_tradable_horizons(
    horizon: Optional[int],
    rebalance_intervals: Optional[Sequence[int]] = None,
) -> List[int]:
    """Resolve one or more tradable holding periods.

    Research sweeps keep ``rebalance_interval_days == horizon`` per combo, so a
    multi-value rebalance list is treated as multiple independent horizons.
    """
    values: List[int] = []
    if rebalance_intervals:
        values = sorted({int(value) for value in rebalance_intervals if int(value) > 0})
    if not values and horizon is not None and int(horizon) > 0:
        values = [int(horizon)]
    if not values:
        raise ValueError("at least one positive rebalance/horizon value is required")
    return values









def add_volatility_columns(
    dataset: pd.DataFrame,
    price_cache: Dict[str, List[Dict[str, Any]]],
    *,
    vol_windows: Sequence[int] = (20, 60),
    atr_window: int = 20,
    percentile_lookback: int = 250,
) -> pd.DataFrame:
    """Attach realized volatility features using only prices on or before ``score_date``."""
    if dataset.empty:
        return dataset

    df = dataset.copy()
    vol_windows = tuple(int(window) for window in vol_windows)
    feature_columns = [f"vol_{window}d_annual" for window in vol_windows] + [f"atr_pct_{int(atr_window)}d"]
    for column in feature_columns:
        df[column] = None

    if "symbol" not in df.columns or "score_date" not in df.columns:
        return df

    score_rows = pd.DataFrame(
        {
            "_row_id": range(len(df)),
            "_symbol": df["symbol"].astype(str),
            "_score_dt": pd.to_datetime(df["score_date"].astype(str), format="%Y%m%d", errors="coerce"),
        }
    )
    score_rows = score_rows.dropna(subset=["_score_dt"])

    # Compute rolling features once per symbol, then as-of join each score date
    # to the latest feature row with trade_date <= score_date.
    for symbol, symbol_scores in score_rows.groupby("_symbol", sort=False):
        feature_frame = _build_symbol_volatility_frame(
            price_cache.get(symbol) or [],
            vol_windows=vol_windows,
            atr_window=int(atr_window),
            percentile_lookback=percentile_lookback,
        )
        if feature_frame.empty:
            continue
        merged = pd.merge_asof(
            symbol_scores.sort_values("_score_dt"),
            feature_frame.sort_values("_trade_dt"),
            left_on="_score_dt",
            right_on="_trade_dt",
            direction="backward",
        )
        row_ids = merged["_row_id"].to_numpy()
        for column in feature_columns:
            df.iloc[row_ids, df.columns.get_loc(column)] = merged[column].where(pd.notna(merged[column]), None).to_numpy()
    return df


def _build_symbol_volatility_frame(
    price_rows: Sequence[Dict[str, Any]],
    *,
    vol_windows: Sequence[int],
    atr_window: int,
    percentile_lookback: int,
) -> pd.DataFrame:
    if not price_rows:
        return pd.DataFrame()

    frame = pd.DataFrame(price_rows)
    required = {"trade_date", "high", "low", "close"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()

    frame = frame.sort_values("trade_date").copy()
    frame["_trade_dt"] = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    frame = frame.dropna(subset=["_trade_dt"])
    if frame.empty:
        return pd.DataFrame()

    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    positive_close = close.where(close > 0)
    returns = positive_close.dropna().pct_change()

    for window in vol_windows:
        vol = returns.rolling(window=int(window), min_periods=int(window)).std(ddof=1) * math.sqrt(252)
        frame[f"vol_{int(window)}d_annual"] = vol.reindex(frame.index)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.dropna().rolling(window=int(atr_window), min_periods=int(atr_window)).mean()
    latest_close = positive_close.ffill()
    frame[f"atr_pct_{int(atr_window)}d"] = atr.reindex(frame.index) / latest_close

    # Kept in the signature for compatibility with the previous implementation;
    # percentile features are not emitted by add_volatility_columns today.
    _ = percentile_lookback
    return frame[["_trade_dt", *[f"vol_{int(window)}d_annual" for window in vol_windows], f"atr_pct_{int(atr_window)}d"]]



def _probe_top_return(group: pd.DataFrame, score_col: str, ret_col: str, probe_n: int) -> Optional[float]:
    """某调仓日按 ``score_col`` 取前 ``probe_n`` 名的等权前向收益（探针，纯 top-n）。"""
    sub = group[[score_col, ret_col]].dropna()
    if sub.empty:
        return None
    sub = sub.sort_values(score_col, ascending=False).head(int(probe_n))
    if sub.empty:
        return None
    return float(sub[ret_col].mean())


def compute_dynamic_weight_schedule(
    dataset: pd.DataFrame,
    rebalance_dates: Sequence[str],
    growth_col: str,
    cycle_col: str,
    ret_col: str,
    probe_top_n: int,
    lookback: int,
    sensitivity: float,
    w_floor: float,
    w_cap: float,
):
    """逐调仓期定 growth 权重：跟随近端 N 期 cycle vs growth 探针表现，夹在 [floor, cap]。

    无未来函数：决定第 i 期权重时只用第 0..i-1 期已实现的探针差(rC-rG)。当
    ``rebalance_interval == horizon`` 时，前一期收益恰在本期开仓时已实现，故前序
    各期均可用。信号 z = 近 N 期均值差 / 历史标准差；cycle 近期占优(差>0)则降低
    growth 权重(偏 cycle)，反之提高(偏 growth)。中枢取 (floor+cap)/2。
    """
    center = (w_floor + w_cap) / 2.0
    probe: List[tuple] = []
    for d in rebalance_dates:
        g = dataset[dataset["score_date"] == d]
        rG = _probe_top_return(g, growth_col, ret_col, probe_top_n)
        rC = _probe_top_return(g, cycle_col, ret_col, probe_top_n)
        probe.append((str(d), rG, rC))

    weight_by_date: Dict[str, float] = {}
    schedule: List[Dict[str, Any]] = []
    diffs_hist: List[float] = []  # 已实现的 (rC - rG)，严格早于当前期
    for d, rG, rC in probe:
        if diffs_hist:
            window = diffs_hist[-int(lookback):]
            sbar = sum(window) / len(window)
            sigma = pstdev(diffs_hist) if len(diffs_hist) > 1 else 0.0
            z = (sbar / sigma) if sigma > 0 else 0.0
            w_growth = max(w_floor, min(w_cap, center - sensitivity * z))
        else:
            sbar, z, w_growth = None, 0.0, center
        weight_by_date[d] = w_growth
        schedule.append(
            {
                "score_date": d,
                "w_growth": w_growth,
                "w_cycle": 1.0 - w_growth,
                "trailing_diff_mean": sbar,
                "z": z,
                "n_prior": len(diffs_hist),
                "probe_growth_ret": rG,
                "probe_cycle_ret": rC,
            }
        )
        if rG is not None and rC is not None:
            diffs_hist.append(rC - rG)
    return weight_by_date, schedule, center


def add_dynamic_blend_column(
    dataset: pd.DataFrame,
    growth_col: str,
    cycle_col: str,
    weight_by_date: Dict[str, float],
    out_col: str,
    center: float,
) -> pd.DataFrame:
    """按每行 score_date 对应的动态 growth 权重，写入混合分列（NaN 退化同固定混合）。"""
    df = dataset.copy()
    growth = pd.to_numeric(df[growth_col], errors="coerce")
    cycle = pd.to_numeric(df[cycle_col], errors="coerce")
    w = df["score_date"].astype(str).map(weight_by_date).fillna(center).astype(float)
    available_weight = growth.notna().astype(float) * w + cycle.notna().astype(float) * (1.0 - w)
    weighted_sum = growth.fillna(0) * w + cycle.fillna(0) * (1.0 - w)
    df[out_col] = weighted_sum.where(available_weight > 0) / available_weight.where(available_weight > 0)
    return df





def _log(message: str, verbose: bool = True) -> None:
    if verbose:
        print(message, flush=True)






def run_parameter_sweep(
    dataset: pd.DataFrame,
    score_column: str,
    top_n_values: Sequence[int],
    horizon: int,
    rebalance_intervals: Sequence[int],
    variants: Sequence[SweepVariant],
    transaction_cost: float,
    index_benchmark_symbol: Optional[str],
    buy_commission_rate: Optional[float] = None,
    sell_commission_rate: Optional[float] = None,
    min_commission: Optional[float] = None,
    stamp_tax_rate: Optional[float] = None,
    transfer_fee_rate: Optional[float] = None,
    index_price_history: Optional[Sequence[Dict[str, Any]]] = None,
    score_sweeps: Optional[Sequence[ScoreSweep]] = None,
    vol_variants: Optional[Sequence[VolatilityVariant]] = None,
    volatility_column: str = "vol_20d_annual",
    trailing_stop_pct: Optional[float] = None,
    path_gates: Optional[Sequence[str]] = None,
    regime_modes: Optional[Sequence[str]] = None,
    regime_rules: Optional[Sequence[str]] = None,
    regime_rule: Optional[str] = None,
    cash_on_dates: Optional[Sequence[str]] = None,
    regime_cash_on_dates_by_rule: Optional[Dict[str, Sequence[str]]] = None,
    intra_period_cash: bool = False,
    bull_exposures: Optional[Sequence[float]] = None,
    refill_modes: Optional[Sequence[Any]] = None,
    crowd_exit_ks: Optional[Sequence[Any]] = None,
    non_bull_actions: Optional[Sequence[Any]] = None,
    industry_regime_scales: Optional[Sequence[Any]] = None,
    price_cache: Optional[Dict[str, Any]] = None,
    selection_mode: str = "fixed_top_n",
    threshold_lookback_days: int = 10,
    max_positions: int = 20,
    verbose: bool = False,
    *,
    watermark: Optional[Dict[str, Any]] = None,
    checkpoint: Optional[SweepCheckpoint] = None,
    parallel_workers: Optional[int] = None,
    independent_horizons: Optional[Sequence[int]] = None,
    industry_l1_codes: Sequence[str] = (),
) -> Dict[str, Any]:
    selection_mode = str(selection_mode or "fixed_top_n").strip().lower()
    if selection_mode not in {"fixed_top_n", "dynamic_score_threshold"}:
        raise ValueError(f"Unsupported selection_mode: {selection_mode}")
    if selection_mode == "dynamic_score_threshold":
        if len(top_n_values) != 1:
            raise ValueError(
                "dynamic_score_threshold requires top_n_values to contain exactly one value"
            )
        if int(threshold_lookback_days) < 1:
            raise ValueError(
                "dynamic_score_threshold requires threshold_lookback_days >= 1"
            )
        if int(max_positions) < int(top_n_values[0]):
            raise ValueError(
                "dynamic_score_threshold requires max_positions >= top_n"
            )
    if independent_horizons:
        horizon_values = resolve_tradable_horizons(horizon, independent_horizons)
        horizon_intervals = [(value, value) for value in horizon_values]
    else:
        rebalance_intervals = validate_rebalance_intervals(horizon, rebalance_intervals)
        horizon_intervals = [(int(horizon), int(interval)) for interval in rebalance_intervals]
    score_sweeps = list(score_sweeps or [ScoreSweep(name=score_column, score_column=score_column)])
    vol_variants = list(vol_variants or build_volatility_variants())
    path_gate_values = normalize_path_gates(path_gates)
    regime_mode_values = normalize_regime_modes(regime_modes)
    needs_regime_cash = REGIME_MODE_BULL_G60_ELSE_CASH in regime_mode_values
    regime_rule_values = (
        resolve_regime_rules(regime_rules=regime_rules, regime_rule=regime_rule)
        if needs_regime_cash
        else []
    )
    regime_mode_arm_count = count_regime_mode_arms(regime_mode_values, regime_rule_values)
    bull_exposure_values = normalize_bull_exposures(bull_exposures)
    refill_mode_values = normalize_refill_modes(refill_modes)
    crowd_exit_k_values = normalize_crowd_exit_ks(crowd_exit_ks)
    non_bull_action_values = normalize_non_bull_actions(non_bull_actions)
    industry_regime_scale_values = normalize_industry_regime_scales(industry_regime_scales)
    cash_dates = tuple(str(date) for date in (cash_on_dates or []) if str(date))
    index_prices = list(index_price_history or [])
    trading_calendar = [row.get("trade_date") for row in index_prices]
    industry_codes = tuple(str(code) for code in industry_l1_codes if str(code).strip())
    fee_kwargs = {
        "buy_commission_rate": buy_commission_rate,
        "sell_commission_rate": sell_commission_rate,
        "min_commission": min_commission,
        "stamp_tax_rate": stamp_tax_rate,
        "transfer_fee_rate": transfer_fee_rate,
    }

    outer_jobs: List[SweepOuterJob] = []
    total_jobs = max(
        1,
        len(score_sweeps)
        * len(horizon_intervals)
        * len(variants)
        * len(vol_variants)
        * len(path_gate_values)
        * regime_mode_arm_count
        * len(bull_exposure_values)
        * len(refill_mode_values)
        * len(crowd_exit_k_values)
        * len(non_bull_action_values)
        * len(industry_regime_scale_values),
    )
    job_index = 0

    for score_sweep in score_sweeps:
        for job_horizon, interval in horizon_intervals:
            for variant in variants:
                max_weight = variant.max_industry_weight if variant.max_industry_weight is not None else 0.3
                if selection_mode == "dynamic_score_threshold":
                    exposure_metrics = {}
                else:
                    exposure = build_industry_exposure(
                        dataset,
                        score_column=score_sweep.score_column,
                        top_n_values=top_n_values,
                        horizon=job_horizon,
                        rebalance_interval_days=int(interval),
                        trading_calendar=trading_calendar,
                        construction_mode=variant.construction_mode,
                        max_industry_weight=max_weight,
                    )
                    exposure_metrics = _exposure_metrics_by_top_n(
                        exposure.get("periods", [])
                    )

                for path_gate in path_gate_values:
                    regime_mode_jobs: List[Tuple[str, Optional[str]]] = []
                    if "off" in regime_mode_values:
                        regime_mode_jobs.append(("off", None))
                    if needs_regime_cash:
                        for rule_id in regime_rule_values:
                            regime_mode_jobs.append((REGIME_MODE_BULL_G60_ELSE_CASH, rule_id))
                    for regime_mode, regime_rule_id in regime_mode_jobs:
                        for vol_variant in vol_variants:
                            for bull_exposure in bull_exposure_values:
                                for intra_period_refill in refill_mode_values:
                                    for crowd_exit_k in crowd_exit_k_values:
                                        # The industry-regime scale is orthogonal to
                                        # every axis above, so it multiplies in here
                                        # rather than nesting another block.
                                        for non_bull_action, industry_regime_scale in itertools.product(
                                            non_bull_action_values, industry_regime_scale_values
                                        ):
                                            if (
                                                str(regime_mode or "off") != REGIME_MODE_BULL_G60_ELSE_CASH
                                                and non_bull_action != NON_BULL_ACTION_CASH
                                            ):
                                                continue
                                            job_index += 1
                                            _log(
                                                f"[{job_index}/{total_jobs}] sweep score={score_sweep.name}, "
                                                f"variant={variant.name}, path_gate={path_gate}, "
                                                f"regime={regime_mode}, vol={vol_variant.name}, "
                                                f"bull_exposure={bull_exposure:g}, "
                                                f"refill={'on' if intra_period_refill else 'off'}, "
                                                f"crowd_exit_k={crowd_exit_k}, "
                                                f"non_bull={non_bull_action}, "
                                                f"industry_regime_scale={industry_regime_scale:g}, "
                                                f"interval={interval}d",
                                                verbose,
                                            )
                                            job_params = {
                                                "score_variant": score_sweep.name,
                                                "score_column": score_sweep.score_column,
                                                "growth_weight": score_sweep.growth_weight,
                                                "cycle_weight": score_sweep.cycle_weight,
                                                "variant": variant.name,
                                                "construction_mode": variant.construction_mode,
                                                "max_industry_weight": variant.max_industry_weight,
                                                "selection_mode": selection_mode,
                                                "threshold_lookback_days": int(
                                                    threshold_lookback_days
                                                ),
                                                "max_positions": int(max_positions),
                                                "path_gate": path_gate,
                                                "regime_mode": regime_mode,
                                                **(
                                                    {"regime_rule": regime_rule_id}
                                                    if regime_rule_id
                                                    else {}
                                                ),
                                                "volatility_variant": vol_variant.name,
                                                "volatility_mode": vol_variant.mode,
                                                "target_vol_annual": vol_variant.target_vol,
                                                "volatility_column": volatility_column,
                                                "rebalance_interval_days": int(interval),
                                                "transaction_cost": transaction_cost,
                                                "index_benchmark_symbol": index_benchmark_symbol,
                                                "horizon": job_horizon,
                                                "industry_l1_codes": list(industry_codes),
                                                "bull_exposure": float(bull_exposure),
                                                "intra_period_refill": bool(intra_period_refill),
                                                "crowd_exit_k": int(crowd_exit_k),
                                                "non_bull_action": non_bull_action,
                                                "industry_regime_scale": float(industry_regime_scale),
                                                **fee_kwargs,
                                            }
                                            if score_sweep.weights and not score_sweep.legacy_growth_cycle:
                                                job_params["score_weights"] = score_sweep.weights
                                            job_hash = combo_param_hash(
                                                job_params,
                                                watermark=watermark,
                                                trailing_stop_pct=trailing_stop_pct,
                                            )
                                            outer_jobs.append(
                                                SweepOuterJob(
                                                    job_hash=job_hash,
                                                    score_sweep_name=score_sweep.name,
                                                    score_column=score_sweep.score_column,
                                                    growth_weight=score_sweep.growth_weight,
                                                    cycle_weight=score_sweep.cycle_weight,
                                                    score_weights=(
                                                        score_sweep.weights
                                                        if score_sweep.weights and not score_sweep.legacy_growth_cycle
                                                        else None
                                                    ),
                                                    horizon=job_horizon,
                                                    interval=int(interval),
                                                    variant_name=variant.name,
                                                    construction_mode=variant.construction_mode,
                                                    max_industry_weight=variant.max_industry_weight,
                                                    selection_mode=selection_mode,
                                                    threshold_lookback_days=int(
                                                        threshold_lookback_days
                                                    ),
                                                    max_positions=int(max_positions),
                                                    path_gate=path_gate,
                                                    vol_name=vol_variant.name,
                                                    vol_mode=vol_variant.mode,
                                                    target_vol=vol_variant.target_vol,
                                                    exposure_metrics=exposure_metrics,
                                                    trailing_stop_pct=trailing_stop_pct,
                                                    regime_mode=regime_mode,
                                                    regime_rule=regime_rule_id,
                                                    bull_exposure=float(bull_exposure),
                                                    intra_period_refill=bool(intra_period_refill),
                                                    crowd_exit_k=int(crowd_exit_k),
                                                    non_bull_action=non_bull_action,
                                                    industry_regime_scale=float(industry_regime_scale),
                                                )
                                            )

    rows = run_outer_jobs(
        outer_jobs,
        dataset=dataset,
        top_n_values=top_n_values,
        transaction_cost=transaction_cost,
        fee_kwargs=fee_kwargs,
        index_benchmark_symbol=index_benchmark_symbol,
        index_price_history=index_prices,
        volatility_column=volatility_column,
        risk_adjusted_score_fn=_risk_adjusted_score,
        checkpoint=checkpoint,
        parallel_workers=parallel_workers,
        verbose=verbose,
        industry_l1_codes=industry_codes,
        cash_on_dates=cash_dates,
        regime_cash_on_dates_by_rule=regime_cash_on_dates_by_rule,
        intra_period_cash=bool(intra_period_cash),
        price_cache=price_cache,
    )

    payload = {
        "score_column": score_column,
        "score_sweeps": [sweep.__dict__ for sweep in score_sweeps],
        "volatility_variants": [variant.__dict__ for variant in vol_variants],
        "path_gates": list(path_gate_values),
        "regime_modes": list(regime_mode_values),
        **({"regime_rules": list(regime_rule_values)} if regime_rule_values else {}),
        "bull_exposures": list(bull_exposure_values),
        "refill_modes": ["on" if flag else "off" for flag in refill_mode_values],
        "crowd_exit_ks": list(crowd_exit_k_values),
        "non_bull_actions": list(non_bull_action_values),
        "industry_regime_scales": list(industry_regime_scale_values),
        "intra_period_cash": bool(intra_period_cash),
        "volatility_column": volatility_column,
        "horizon": max(job_horizon for job_horizon, _ in horizon_intervals),
        "top_n_values": list(top_n_values),
        "selection_mode": selection_mode,
        "threshold_lookback_days": int(threshold_lookback_days),
        "max_positions": int(max_positions),
        "rebalance_intervals": [interval for _, interval in horizon_intervals],
        "index_benchmark_symbol": index_benchmark_symbol,
        "rows": sorted(rows, key=lambda row: (_as_float(row.get("risk_adjusted_score")) is None, -(_as_float(row.get("risk_adjusted_score")) or 0))),
        "sweep_parallel_workers": parallel_workers if parallel_workers is not None else sweep_worker_count(),
    }
    if industry_codes:
        payload["industry_l1_codes"] = list(industry_codes)
    if checkpoint is not None:
        payload["sweep_checkpoint_completed"] = checkpoint.completed_count()
    payload["stability"] = summarize_stability(payload["rows"], sweep_axes=infer_sweep_axes(rows=payload["rows"]))
    return attach_sweep_view(payload)


def summarize_stability(
    rows: Sequence[Dict[str, Any]],
    *,
    sweep_axes: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    axes = list(sweep_axes or infer_sweep_axes(rows=rows))
    group_columns = [column for column in stability_group_columns(axes) if column in df.columns]
    if not group_columns:
        group_columns = ["variant", "top_n"]
        if "score_variant" in df.columns:
            group_columns = ["score_variant"] + group_columns
    stability_rows = []
    # dropna=False：分组键本身可能就是空的（如 trailing_stop_pct=None 表示不设止盈）。
    # 默认的 dropna=True 会把这类行整组丢掉，明细表因此系统性缺档且不报错。
    for keys, group in df.groupby(group_columns, dropna=False):
        key_map = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
        key_map = {column: _nan_to_none(value) for column, value in key_map.items()}
        returns = pd.to_numeric(group["cumulative_return"], errors="coerce").dropna()
        excess = pd.to_numeric(group["index_excess_cumulative_return"], errors="coerce").dropna()
        drawdown = pd.to_numeric(group["max_drawdown"], errors="coerce").dropna()
        active_share = pd.to_numeric(group["avg_active_share"], errors="coerce").dropna()
        avg_exposure = (
            pd.to_numeric(group["average_exposure"], errors="coerce").dropna()
            if "average_exposure" in group.columns
            else pd.Series(dtype=float)
        )
        stability_row = dict(key_map)
        if stability_row.get("top_n") is not None:
            stability_row["top_n"] = int(stability_row["top_n"])
        stability_row.update(
            {
                "observations": int(len(group)),
                "avg_cumulative_return": _mean_or_none(returns),
                "min_cumulative_return": _min_or_none(returns),
                "avg_index_excess_return": _mean_or_none(excess),
                "min_index_excess_return": _min_or_none(excess),
                "worst_max_drawdown": _min_or_none(drawdown),
                "avg_active_share": _mean_or_none(active_share),
                "avg_average_exposure": _mean_or_none(avg_exposure),
            }
        )
        stability_rows.append(stability_row)
    return sorted(
        stability_rows,
        key=lambda row: (
            -(_as_float(row.get("avg_index_excess_return")) or -999),
            _as_float(row.get("avg_active_share")) or 999,
        ),
    )


def _overlapping_holding_windows(payload: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Return (horizon, rebalance) pairs that genuinely overlap."""
    pairs: set[Tuple[int, int]] = set()
    has_row_horizons = False
    for row in payload.get("rows") or []:
        if row.get("horizon") is None or row.get("rebalance_interval_days") is None:
            continue
        has_row_horizons = True
        row_horizon = int(row["horizon"])
        row_rebalance = int(row["rebalance_interval_days"])
        if row_rebalance < row_horizon:
            pairs.add((row_horizon, row_rebalance))

    if has_row_horizons:
        return sorted(pairs)

    horizon = int(payload.get("horizon") or 0)
    intervals = [int(value) for value in payload.get("rebalance_intervals", []) if value]
    return [(horizon, interval) for interval in intervals if interval < horizon]


def render_html(payload: Dict[str, Any], title: str, start_date: str, end_date: str, index_buy_hold_return: Optional[float] = None) -> str:
    rows = payload.get("rows", [])
    stability = payload.get("stability", [])
    overlapping_windows = _overlapping_holding_windows(payload)
    parts = [
        '<section class="hero">',
        '<p class="eyebrow">Growth-Cycle Parameter Sweep</p>',
        f"<h1>{escape(title)}</h1>",
        f"<p class=\"subtitle\">区间：{escape(start_date)}-{escape(end_date)} · "
        f"Score: <code>{escape(_score_sweeps_label(payload))}</code> · "
        f"Horizon: {escape(str(payload.get('horizon')))} 个交易日</p>",
        "</section>",
    ]
    if index_buy_hold_return is not None:
        parts.append('<section class="summary-card">')
        parts.append(f"<div class=\"label\">{escape(str(payload.get('index_benchmark_symbol')))} 买入持有收益</div>")
        parts.append(f"<div class=\"big\">{_fmt_pct(index_buy_hold_return)}</div>")
        parts.append("</section>")

    if overlapping_windows:
        parts.append('<section class="warning">')
        parts.append("<h2>口径提示</h2>")
        parts.append(
            "<p>sweep 中存在调仓间隔小于对应持有期的组合："
            f"{escape(', '.join(f'horizon={horizon}d/rebalance={interval}d' for horizon, interval in overlapping_windows))}。"
            "这些组合存在持有窗口重叠，适合观察信号稳定性，不应直接理解为严格实盘资金曲线。</p>"
        )
        parts.append("</section>")

    parts.extend(_experiment_design_sections(payload))
    parts.extend(_facet_best_sections(payload))

    display_limit = 500
    display_rows = rows[:display_limit]
    parts.append("<section><h2>综合排行</h2>")
    if len(rows) > display_limit:
        parts.append(f"<p class=\"muted\">共 {len(rows)} 行，下表显示前 {display_limit} 行（完整数据见 JSON 产物）。</p>")
    else:
        parts.append(f"<p class=\"muted\">共 {len(rows)} 行。</p>")
    parts.append(_rows_table(display_rows, sweep_axes=(payload.get("sweep_view") or {}).get("sweep_axes")))
    parts.append("</section>")

    parts.append("<section><h2>稳定性汇总</h2>")
    parts.append(_stability_table(stability))
    parts.append("</section>")

    parts.append("<section><h2>解读</h2><ul>")
    parts.append("<li><strong>综合排行</strong>按风险调整分排序：指数超额越高越好，回撤、换手和行业偏离越低越好。</li>")
    parts.append("<li><strong>稳定性汇总</strong>按同一策略和 TopN 聚合不同调仓周期/行业上限，用于观察收益是否依赖单一参数。</li>")
    parts.append("<li><strong>Avg Active Share</strong>和<strong>Avg Max Industry Active</strong>来自行业暴露诊断，越低表示行业约束越干净。</li>")
    parts.append("<li><strong>Vol Mode / Target Vol / Avg Exposure</strong>用于波动率降仓 A/B：区分留现金与满仓逆波动率倾斜，并观察平均资金敞口。</li>")
    parts.append("<li><strong>分面最优对比</strong>按每个扫参轴展示各档最优组合，避免全局排序掩盖低排名档位。</li>")
    parts.append("</ul></section>")
    return _html_document(title, "".join(parts))


def _experiment_design_sections(payload: Dict[str, Any]) -> List[str]:
    sweep_view = payload.get("sweep_view") or {}
    sweep_axes = sweep_view.get("sweep_axes") or infer_sweep_axes(rows=payload.get("rows") or [])
    if not sweep_axes:
        return []
    parts = ['<section><h2>实验设计</h2><ul class="axis-list">']
    combo_count = int(sweep_view.get("row_count") or len(payload.get("rows") or []))
    parts.append(f"<li><strong>组合数</strong>：{combo_count}</li>")
    for axis in sweep_axes:
        rows = payload.get("rows") or []
        labels = []
        for value in axis.get("values") or []:
            sample = next(
                (row for row in rows if _row_matches_axis(row, axis["key"], value)),
                None,
            )
            labels.append(format_axis_value(axis["key"], value, sample))
        labels = ", ".join(labels)
        parts.append(f"<li><strong>{escape(axis.get('label') or axis['key'])}</strong>：{escape(labels)}</li>")
    parts.append("</ul></section>")
    return parts


def _facet_best_sections(payload: Dict[str, Any]) -> List[str]:
    sweep_view = payload.get("sweep_view") or {}
    facet_best = sweep_view.get("facet_best") or {}
    if not facet_best:
        return []
    parts: List[str] = ['<section><h2>分面最优对比</h2>']
    for axis_key, entries in facet_best.items():
        if not entries:
            continue
        label = next((axis.get("label") for axis in (sweep_view.get("sweep_axes") or []) if axis.get("key") == axis_key), axis_key)
        parts.append(f"<h3>{escape(str(label))}</h3>")
        parts.append(_facet_best_table(entries, axis_key))
    parts.append("</section>")
    return parts


def _facet_best_table(entries: Sequence[Dict[str, Any]], axis_key: str) -> str:
    headers = ["档位", "Variant", "TopN", "累计收益", "年化", "指数超额", "综合分"]
    body = []
    for entry in entries:
        row = entry.get("row") or {}
        body.append(
            "<tr>"
            f"<td>{escape(str(entry.get('label') or format_axis_value(axis_key, entry.get('value'))))}</td>"
            f"<td class=\"strategy\">{escape(str(row.get('variant') or '-'))}</td>"
            f"<td>Top{escape(str(row.get('top_n') or '-'))}</td>"
            f"<td>{_fmt_pct(row.get('cumulative_return'))}</td>"
            f"<td>{_fmt_pct(row.get('annualized_return'))}</td>"
            f"<td>{_fmt_pct(row.get('index_excess_cumulative_return'))}</td>"
            f"<td>{_fmt_num(row.get('risk_adjusted_score'), 3)}</td>"
            "</tr>"
        )
    return _table(headers, body)


def _exposure_metrics_by_top_n(periods: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
    if not periods:
        return {}
    df = pd.DataFrame(periods)
    metrics: Dict[int, Dict[str, float]] = {}
    for top_n, group in df.groupby("top_n"):
        metrics[int(top_n)] = {
            "avg_active_share": float(group["active_share"].mean()),
            "avg_max_abs_active_weight": float(group["max_abs_active_weight"].mean()),
            "max_abs_active_weight": float(group["max_abs_active_weight"].max()),
        }
    return metrics


def _risk_adjusted_score(summary: Dict[str, Any], exposure_metrics: Dict[str, Any]) -> Optional[float]:
    excess = _as_float(summary.get("index_excess_cumulative_return"))
    drawdown = _as_float(summary.get("max_drawdown"))
    turnover = _as_float(summary.get("average_turnover"))
    active_share = _as_float(exposure_metrics.get("avg_active_share"))
    if excess is None:
        return None
    score = excess
    if drawdown is not None:
        score -= abs(drawdown) * 0.5
    if turnover is not None:
        score -= turnover * 0.15
    if active_share is not None:
        score -= active_share * 0.1
    return float(score)


def _rows_table(rows: Sequence[Dict[str, Any]], sweep_axes: Optional[Sequence[Dict[str, Any]]] = None) -> str:
    if not rows:
        return "<p>No sweep rows.</p>"
    axes = list(sweep_axes or infer_sweep_axes(rows=rows))
    swept_keys = {axis["key"] for axis in axes}
    dynamic_headers = [str(axis.get("label") or axis["key"]) for axis in axes]
    identity_headers: List[str] = []
    if "score_variant" not in swept_keys:
        identity_headers.append("Score")
    identity_headers.extend(["Growth", "Cycle", "Variant", "Vol Mode", "Target Vol"])
    if "top_n" not in swept_keys:
        identity_headers.append("TopN")
    if "rebalance_interval_days" not in swept_keys:
        identity_headers.append("调仓")
    if "max_industry_weight" not in swept_keys:
        identity_headers.append("行业上限")
    metric_headers = [
        "累计收益",
        "年化",
        "指数超额",
        "夏普",
        "最大回撤",
        "换手",
        "Avg Exposure",
        "Avg Active Share",
        "Avg Max Industry Active",
        "综合分",
    ]
    headers = ["Rank", *dynamic_headers, *identity_headers, *metric_headers]
    body = []
    for rank, row in enumerate(rows, start=1):
        dynamic_cells = "".join(
            f"<td>{escape(format_axis_value(axis['key'], row.get(axis['key']), row))}</td>" for axis in axes
        )
        identity_cells: List[str] = []
        if "score_variant" not in swept_keys:
            identity_cells.append(
                f"<td class=\"strategy\">{escape(str(row.get('score_variant') or row.get('score_column') or '-'))}</td>"
            )
        identity_cells.extend(
            [
                f"<td>{_fmt_pct(row.get('growth_weight'))}</td>",
                f"<td>{_fmt_pct(row.get('cycle_weight'))}</td>",
                f"<td class=\"strategy\">{escape(str(row.get('variant')))}</td>",
                f"<td>{escape(str(row.get('volatility_mode') or '-'))}</td>",
                (
                    f"<td>{_fmt_pct(row.get('target_vol_annual'))}</td>"
                    if row.get("target_vol_annual") is not None
                    else "<td>-</td>"
                ),
            ]
        )
        if "top_n" not in swept_keys:
            identity_cells.append(f"<td>Top{escape(str(row.get('top_n')))}</td>")
        if "rebalance_interval_days" not in swept_keys:
            identity_cells.append(f"<td>{escape(str(row.get('rebalance_interval_days')))}d</td>")
        if "max_industry_weight" not in swept_keys:
            identity_cells.append(
                f"<td>{escape(format_axis_value('max_industry_weight', row.get('max_industry_weight'), row))}</td>"
            )
        body.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"{dynamic_cells}"
            f"{''.join(identity_cells)}"
            f"<td>{_fmt_pct(row.get('cumulative_return'))}</td>"
            f"<td>{_fmt_pct(row.get('annualized_return'))}</td>"
            f"<td>{_fmt_pct(row.get('index_excess_cumulative_return'))}</td>"
            f"<td>{_fmt_num(row.get('sharpe'))}</td>"
            f"<td>{_fmt_pct(row.get('max_drawdown'))}</td>"
            f"<td>{_fmt_pct(row.get('average_turnover'))}</td>"
            f"<td>{_fmt_pct(row.get('average_exposure'))}</td>"
            f"<td>{_fmt_pct(row.get('avg_active_share'))}</td>"
            f"<td>{_fmt_pct(row.get('avg_max_abs_active_weight'))}</td>"
            f"<td>{_fmt_num(row.get('risk_adjusted_score'), 3)}</td>"
            "</tr>"
        )
    return _table(headers, body)


def _stability_table(rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        return "<p>No stability rows.</p>"
    headers = [
        "Score",
        "Variant",
        "Vol",
        "TopN",
        "样本数",
        "平均累计收益",
        "最低累计收益",
        "平均指数超额",
        "最低指数超额",
        "最差最大回撤",
        "Avg Exposure",
        "Avg Active Share",
    ]
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td class=\"strategy\">{escape(str(row.get('score_variant') or '-'))}</td>"
            f"<td class=\"strategy\">{escape(str(row.get('variant')))}</td>"
            f"<td class=\"strategy\">{escape(str(row.get('volatility_variant') or '-'))}</td>"
            f"<td>Top{escape(str(row.get('top_n')))}</td>"
            f"<td>{escape(str(row.get('observations')))}</td>"
            f"<td>{_fmt_pct(row.get('avg_cumulative_return'))}</td>"
            f"<td>{_fmt_pct(row.get('min_cumulative_return'))}</td>"
            f"<td>{_fmt_pct(row.get('avg_index_excess_return'))}</td>"
            f"<td>{_fmt_pct(row.get('min_index_excess_return'))}</td>"
            f"<td>{_fmt_pct(row.get('worst_max_drawdown'))}</td>"
            f"<td>{_fmt_pct(row.get('avg_average_exposure'))}</td>"
            f"<td>{_fmt_pct(row.get('avg_active_share'))}</td>"
            "</tr>"
        )
    return _table(headers, body)


def _table(headers: Sequence[str], rows: Sequence[str]) -> str:
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + "".join(f"<th>{escape(header)}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _nan_to_none(value: Any) -> Any:
    """把分组键里的 NaN 还原成 None —— NaN 不是合法 JSON，序列化会直接报错。"""
    try:
        return None if pd.isna(value) else value
    except (TypeError, ValueError):
        return value


def _mean_or_none(series: pd.Series) -> Optional[float]:
    return float(series.mean()) if len(series) else None


def _min_or_none(series: pd.Series) -> Optional[float]:
    return float(series.min()) if len(series) else None


def _score_sweeps_label(payload: Dict[str, Any]) -> str:
    sweeps = payload.get("score_sweeps") or []
    names = [str(sweep.get("name")) for sweep in sweeps if isinstance(sweep, dict) and sweep.get("name")]
    if names:
        return ", ".join(names[:8]) + (" ..." if len(names) > 8 else "")
    return str(payload.get("score_column"))


def _output_score_slug(score_sweeps: Sequence[ScoreSweep], fallback: str) -> str:
    weighted = [sweep for sweep in score_sweeps if sweep.weights]
    if not weighted:
        return fallback
    if len(weighted) > 3:
        return (
            "growth_cycle_weights"
            if all(sweep.legacy_growth_cycle for sweep in weighted)
            else "score_blends"
        )
    return "_".join(sweep.name for sweep in weighted)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep growth-cycle portfolio parameters and generate an HTML report.")
    parser.add_argument("--start-date", default="20250101", help="Inclusive score_date lower bound, YYYYMMDD")
    parser.add_argument("--end-date", default="20260501", help="Inclusive score_date upper bound, YYYYMMDD")
    parser.add_argument("--universe-index", nargs="+", default=["csi1000"], help="Index universe loaded from index_constituents")
    parser.add_argument("--score-column", default="composite_growth_cycle_score", help="Score column used for ranking")
    parser.add_argument(
        "--score-specs",
        nargs="+",
        help=(
            "Score specs to sweep: existing column names, JSON objects, or "
            "dimension:weight lists such as growth:0.3,cycle:0.7"
        ),
    )
    parser.add_argument(
        "--growth-cycle-weights",
        nargs="+",
        help="Growth:cycle weights to sweep, e.g. 20:80 40:60 60:40 or growth weights like 0.6 0.8",
    )
    parser.add_argument("--horizon", type=int, default=20, help="Forward-return horizon in trading days")
    parser.add_argument(
        "--rebalance-interval-days",
        nargs="+",
        type=int,
        help="Rebalance interval for tradable sweeps; defaults to horizon and must equal horizon",
    )
    parser.add_argument("--top-n", nargs="+", type=int, default=list(DEFAULT_TOP_N_VALUES), help="Top-N portfolio sizes to sweep")
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
    parser.add_argument("--active-caps", nargs="+", type=float, default=list(DEFAULT_ACTIVE_CAPS), help="Max industry weights for active capped variants")
    parser.add_argument("--transaction-cost", type=float, default=0.001, help="Legacy one-way cost applied to turnover")
    parser.add_argument("--buy-commission-rate", type=float, help="Buy-side commission rate")
    parser.add_argument("--sell-commission-rate", type=float, help="Sell-side commission rate")
    parser.add_argument("--min-commission", type=float, help="Minimum commission per order")
    parser.add_argument("--stamp-tax-rate", type=float, help="Sell-side stamp tax rate")
    parser.add_argument("--transfer-fee-rate", type=float, help="Per-side transfer fee rate")
    parser.add_argument(
        "--index-benchmark-symbol",
        help="Index symbol for market benchmark returns (defaults to the index matching --universe-index)",
    )
    parser.add_argument(
        "--trailing-stop-pct",
        type=float,
        help="Trailing stop from peak high: close <= peak*(1-pct) exits next open (disabled when omitted or <=0)",
    )
    parser.add_argument(
        "--trailing-stop-pcts",
        nargs="+",
        type=float,
        help="Sweep multiple trailing-stop values for A/B comparison (0 disables); overrides --trailing-stop-pct",
    )
    parser.add_argument("--output", help="HTML output path")
    parser.add_argument("--json-output", help="Optional JSON output path")
    parser.add_argument(
        "--cycle-score-source",
        help="可选：用外部文件(.parquet/.csv[.gz]，含 symbol/score_date/cycle_score)覆盖 cycle_score 列，"
        "用于对照实验版 cycle（如 cycle_v2），不写库、不改打分链路",
    )
    parser.add_argument(
        "--dynamic-weight",
        action="store_true",
        help="额外加入一档动态 growth/cycle 权重(随近端 N 期 Top-N 表现在 [floor,cap] 内微调)，与固定混合同表对照",
    )
    parser.add_argument(
        "--regime-weight",
        action="store_true",
        help="额外加入一档指数 MA 三态 regime 权重(bull 60:40 / uncertain 20:80 / bear 纯 cycle)，与固定混合同表对照",
    )
    parser.add_argument(
        "--regime-cash",
        action="store_true",
        help="只跑非牛空仓（bull_g60_else_cash）。要对照始终满仓时用 --regime-modes off bull_g60_else_cash",
    )
    parser.add_argument(
        "--regime-modes",
        nargs="+",
        help="Regime 仓位模式：off（始终满仓）和/或 bull_g60_else_cash，可单选。默认仅 off。--regime-cash 等价于只跑非牛空仓",
    )
    parser.add_argument(
        "--intra-period-cash",
        action="store_true",
        help="非牛空仓时，持仓期内翻熊即清仓（对齐实盘监控日），而不是等到下一期调仓",
    )
    parser.add_argument(
        "--bull-exposure",
        type=float,
        help="Invested fraction while labeled bull, in (0, 1]. Default 1.0 (full book). Overridden by --bull-exposures",
    )
    parser.add_argument(
        "--bull-exposures",
        nargs="+",
        type=float,
        help="Sweep bull-period invested fractions, e.g. 1.0 0.5 (research opt-in; default 1.0 only)",
    )
    parser.add_argument(
        "--intra-period-refill",
        action="store_true",
        help="After a trailing stop, refill the slot before the next rebalance (research opt-in). Implies refill on unless --refill-modes is set",
    )
    parser.add_argument(
        "--refill-modes",
        nargs="+",
        help="Sweep intra-period trailing-stop refill: off and/or on. Default off only",
    )
    parser.add_argument(
        "--crowd-exit-k",
        type=int,
        help="If K names hit trailing stop in a period, flatten the rest (0=off). Overridden by --crowd-exit-ks",
    )
    parser.add_argument(
        "--crowd-exit-ks",
        nargs="+",
        type=int,
        help="Sweep crowd-exit K values (0=off). Default 0 only. Research opt-in",
    )
    parser.add_argument(
        "--non-bull-actions",
        nargs="+",
        help=(
            "Non-bull book: cash (default), half_topn (TopN/2 fully invested), "
            "half_topn_slot (TopN/2 at original 1/TopN weights). Research opt-in"
        ),
    )
    parser.add_argument(
        "--industry-regime-scale",
        type=float,
        help=(
            "Weight factor for a name whose own SW L1 industry index is not bull, "
            "in [0, 1]. 1.0 (default) is off. Overridden by --industry-regime-scales"
        ),
    )
    parser.add_argument(
        "--industry-regime-scales",
        nargs="+",
        type=float,
        help="Sweep per-name industry-regime weight factors, e.g. 1.0 0.5 0.0. Research opt-in",
    )
    parser.add_argument(
        "--regime-rule",
        default=None,
        help="Preset id (ma120_anchor, ma60_price, ma60_symmetric, ma60_only), JSON path, or inline JSON",
    )
    parser.add_argument(
        "--confirm-days",
        type=int,
        default=5,
        help="Consecutive raw bull/bear days to enter a regime (default 5)",
    )
    parser.add_argument(
        "--exit-confirm-days",
        type=int,
        default=None,
        help="Consecutive raw bear days to leave bull (default: same as --confirm-days)",
    )
    parser.add_argument(
        "--regime-non-bull-exits",
        action="store_true",
        help="Leave bull on the first non-bull raw day (fast cash). Default sticky hysteresis stays",
    )
    parser.add_argument("--dyn-lookback", type=int, default=3, help="动态权重回看的调仓期数 N (默认 3)")
    parser.add_argument("--dyn-sensitivity", type=float, default=0.1, help="动态权重灵敏度 k：|z|>=1 时约触边界 (默认 0.1)")
    parser.add_argument("--dyn-floor", type=float, default=0.5, help="growth 权重下界 (默认 0.5，即 50:50)")
    parser.add_argument("--dyn-cap", type=float, default=0.7, help="growth 权重上界 (默认 0.7，即 70:30)")
    parser.add_argument("--dyn-probe-top-n", type=int, help="动态信号探针的持仓数 (默认取 --top-n 最小值)")
    parser.add_argument(
        "--volatility-modes",
        nargs="+",
        help="Volatility position modes to sweep: off cash_drag renormalize (default: off only)",
    )
    parser.add_argument(
        "--target-vols",
        nargs="+",
        type=float,
        default=list(DEFAULT_TARGET_VOLS),
        help="Annual target vols for cash_drag/renormalize variants (default: 0.30 0.40 0.50)",
    )
    parser.add_argument(
        "--volatility-column",
        default="vol_20d_annual",
        help="Dataset column for realized annualized vol used in position scaling",
    )
    parser.add_argument(
        "--path-gates",
        nargs="+",
        default=["off"],
        help="Path filter gates for filter-then-rank selection: off ret20_pos above_ma60",
    )
    parser.add_argument(
        "--industry-l1",
        nargs="+",
        default=[],
        help="Shenwan L1 filter (code or Chinese name); single-industry runs use sw_daily benchmark when index symbol omitted",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs")
    parser.add_argument("--enable-tushare", action="store_true", help="Keep TUSHARE_TOKEN enabled when creating ResearchDataAccess")
    return parser.parse_args()


def main() -> int:
    from shared.data_access import open_research_data_access_for_cli

    args = _parse_args()
    verbose = not args.quiet
    try:
        rebalance_intervals = validate_rebalance_intervals(args.horizon, args.rebalance_interval_days)
    except ValueError as exc:
        raise SystemExit(str(exc))

    _log("Initializing research data access...", verbose)
    db_manager = open_research_data_access_for_cli(enable_tushare=args.enable_tushare)
    try:
        score_sweeps = build_score_sweeps(
            args.score_column,
            args.growth_cycle_weights,
            args.score_specs,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))
    _log(
        f"Score sweeps: {', '.join(sweep.name for sweep in score_sweeps)}",
        verbose,
    )
    catalog = sw_l1_catalog(db_manager)
    industry_l1_codes: List[str] = []
    if args.industry_l1:
        try:
            industry_l1_codes = resolve_industry_l1_codes(args.industry_l1, catalog)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    user_benchmark = args.index_benchmark_symbol
    if len(industry_l1_codes) == 1 and not user_benchmark:
        args.index_benchmark_symbol = industry_l1_codes[0]
        _log(f"Index benchmark defaulted to Shenwan L1 {args.index_benchmark_symbol}", verbose)
    elif not args.index_benchmark_symbol:
        args.index_benchmark_symbol = default_index_benchmark_symbol(args.universe_index)
        _log(f"Index benchmark defaulted to {args.index_benchmark_symbol}", verbose)
    _log("Loading index benchmark prices...", verbose)
    index_prices = load_index_benchmark_prices(db_manager, args.index_benchmark_symbol, args.start_date)
    trailing_stop_values = normalize_trailing_stop_pcts(args.trailing_stop_pct, args.trailing_stop_pcts)
    variants = build_sweep_variants_for_request(args.active_caps, industry_l1_codes)
    try:
        vol_variants = build_volatility_variants(args.volatility_modes, args.target_vols)
    except ValueError as exc:
        raise SystemExit(str(exc))
    try:
        path_gate_values = normalize_path_gates(args.path_gates)
    except ValueError as exc:
        raise SystemExit(str(exc))
    try:
        regime_mode_values = resolve_regime_modes(
            regime_modes=args.regime_modes,
            regime_cash=getattr(args, "regime_cash", False),
        )
    except ValueError as exc:
        raise SystemExit(str(exc))
    try:
        if args.bull_exposures:
            bull_exposure_values = normalize_bull_exposures(args.bull_exposures)
        elif getattr(args, "bull_exposure", None) is not None:
            bull_exposure_values = normalize_bull_exposures([args.bull_exposure])
        else:
            bull_exposure_values = normalize_bull_exposures()
    except ValueError as exc:
        raise SystemExit(str(exc))
    try:
        if args.refill_modes:
            refill_mode_values = normalize_refill_modes(args.refill_modes)
        elif getattr(args, "intra_period_refill", False):
            refill_mode_values = [True]
        else:
            refill_mode_values = normalize_refill_modes()
    except ValueError as exc:
        raise SystemExit(str(exc))
    try:
        if args.crowd_exit_ks:
            crowd_exit_k_values = normalize_crowd_exit_ks(args.crowd_exit_ks)
        elif getattr(args, "crowd_exit_k", None) is not None:
            crowd_exit_k_values = normalize_crowd_exit_ks([args.crowd_exit_k])
        else:
            crowd_exit_k_values = normalize_crowd_exit_ks()
    except ValueError as exc:
        raise SystemExit(str(exc))
    try:
        non_bull_action_values = normalize_non_bull_actions(
            getattr(args, "non_bull_actions", None)
        )
    except ValueError as exc:
        raise SystemExit(str(exc))
    try:
        if getattr(args, "industry_regime_scales", None):
            industry_regime_scale_values = normalize_industry_regime_scales(
                args.industry_regime_scales
            )
        elif getattr(args, "industry_regime_scale", None) is not None:
            industry_regime_scale_values = normalize_industry_regime_scales(
                [args.industry_regime_scale]
            )
        else:
            industry_regime_scale_values = normalize_industry_regime_scales()
    except ValueError as exc:
        raise SystemExit(str(exc))

    all_rows: List[Dict[str, Any]] = []
    payload: Dict[str, Any] = {}
    dynamic_schedule_local: Optional[List[Dict[str, Any]]] = None
    dynamic_config_local: Optional[Dict[str, Any]] = None
    industry_regime_coverage_local: Optional[Dict[str, Any]] = None
    regime_schedule_local: Optional[List[Dict[str, Any]]] = None
    regime_config_local: Optional[Dict[str, Any]] = None
    regime_cash_schedule_local: Optional[List[Dict[str, Any]]] = None
    cash_on_dates: List[str] = []
    index_rows_for_regime: Optional[List[Dict[str, Any]]] = None
    needs_regime_index = bool(getattr(args, "regime_weight", False)) or (
        REGIME_MODE_BULL_G60_ELSE_CASH in regime_mode_values
    )
    if needs_regime_index:
        padded_start = padded_index_start(args.start_date)
        index_rows_for_regime = load_index_benchmark_prices(db_manager, args.index_benchmark_symbol, padded_start)
        if not index_rows_for_regime:
            raise SystemExit(
                f"regime labels require benchmark prices for {args.index_benchmark_symbol} from {padded_start}"
            )

    score_config = BacktestConfig(
        horizons=(args.horizon,),
        start_date=args.start_date,
        end_date=args.end_date,
        universe_index_codes=tuple(args.universe_index),
        trailing_stop_pct=None,
    )
    backtester = ScoreBacktester(db_manager, score_config)
    _log(f"Preparing shared score panel ({args.start_date}-{args.end_date})...", verbose)
    score_panel = backtester.prepare_score_panel()
    base_dataset = backtester.build_dataset_from_panel(score_panel, trailing_stop_pct=None)
    if not base_dataset.empty:
        base_dataset = attach_sw_l1(base_dataset, db_manager, as_of_column="score_date")
        _log("Labeling SW L1 industry regimes (no lookahead)...", verbose)
        base_dataset = attach_sw_l1_regime(base_dataset, db_manager)
        _log("Computing volatility features (no lookahead)...", verbose)
        base_dataset = add_volatility_columns(base_dataset, backtester.price_cache)
        _log("Computing path features (no lookahead)...", verbose)
        base_dataset = add_path_feature_columns(base_dataset, backtester.price_cache)

    for trailing_value in trailing_stop_values:
        label = trailing_value if trailing_value is not None else 0.0
        _log(
            f"Building dataset for trailing_stop_pct={label} ({args.start_date}-{args.end_date})...",
            verbose,
        )
        if trailing_value is None or float(trailing_value or 0) <= 0:
            dataset = base_dataset.copy()
        else:
            dataset = backtester.build_dataset_from_panel(score_panel, trailing_stop_pct=trailing_value)
            dataset = merge_volatility_columns(base_dataset, dataset)
        _log(
            f"Dataset ready (trailing_stop_pct={label}): rows={len(dataset)}, "
            f"dates={dataset['score_date'].nunique() if 'score_date' in dataset.columns else 0}, "
            f"symbols={dataset['symbol'].nunique() if 'symbol' in dataset.columns else 0}",
            verbose,
        )
        # Overlay must run per trailing variant: non-off datasets are rebuilt from
        # the panel and would otherwise carry the original (non-overlaid) cycle_score.
        if getattr(args, "cycle_score_source", None):
            dataset = overlay_cycle_score(dataset, args.cycle_score_source, verbose)
        try:
            dataset = add_growth_cycle_weight_columns(dataset, score_sweeps)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        dynamic_schedule_local = None
        dynamic_config_local = None
        score_sweeps_local = list(score_sweeps)
        if getattr(args, "dynamic_weight", False):
            missing = [c for c in ("growth_score", "cycle_score") if c not in dataset.columns]
            if missing:
                raise SystemExit(f"--dynamic-weight 需要列 {missing}（请确保数据集含 growth_score/cycle_score）")
            ret_col = f"forward_return_{args.horizon}d"
            if ret_col not in dataset.columns:
                raise SystemExit(f"--dynamic-weight 需要列 {ret_col}")
            trading_calendar = [row.get("trade_date") for row in index_prices]
            probe_cfg = PortfolioBacktestConfig(horizon=args.horizon, rebalance_interval_days=args.horizon)
            probe_bt = PortfolioBacktester(
                dataset, probe_cfg, index_price_history=index_prices, trading_calendar=trading_calendar
            )
            rebalance_dates = probe_bt._select_rebalance_dates(
                sorted(str(d) for d in dataset["score_date"].dropna().unique())
            )
            probe_top_n = int(args.dyn_probe_top_n) if args.dyn_probe_top_n else int(min(args.top_n))
            weight_by_date, dynamic_schedule_local, center = compute_dynamic_weight_schedule(
                dataset,
                rebalance_dates,
                "growth_score",
                "cycle_score",
                ret_col,
                probe_top_n=probe_top_n,
                lookback=args.dyn_lookback,
                sensitivity=args.dyn_sensitivity,
                w_floor=args.dyn_floor,
                w_cap=args.dyn_cap,
            )
            dataset = add_dynamic_blend_column(
                dataset, "growth_score", "cycle_score", weight_by_date, "growth_cycle_dyn_score", center
            )
            score_sweeps_local = score_sweeps_local + [
                ScoreSweep(name="growth_cycle_dynamic", score_column="growth_cycle_dyn_score")
            ]
            dynamic_config_local = {
                "probe_top_n": probe_top_n,
                "lookback": int(args.dyn_lookback),
                "sensitivity": float(args.dyn_sensitivity),
                "w_floor": float(args.dyn_floor),
                "w_cap": float(args.dyn_cap),
                "center": center,
                "rebalance_dates": len(rebalance_dates),
            }
            _log(
                f"Dynamic weight enabled: probe_top_n={probe_top_n}, N={args.dyn_lookback}, "
                f"k={args.dyn_sensitivity}, band=[{args.dyn_floor},{args.dyn_cap}], "
                f"rebalance_periods={len(rebalance_dates)}",
                verbose,
            )

        if getattr(args, "regime_weight", False):
            missing = [c for c in ("growth_score", "cycle_score") if c not in dataset.columns]
            if missing:
                raise SystemExit(f"--regime-weight 需要列 {missing}（请确保数据集含 growth_score/cycle_score）")
            if not index_rows_for_regime:
                raise SystemExit("--regime-weight benchmark index rows are missing")
            trading_calendar = [row.get("trade_date") for row in index_prices]
            probe_cfg = PortfolioBacktestConfig(horizon=args.horizon, rebalance_interval_days=args.horizon)
            probe_bt = PortfolioBacktester(
                dataset, probe_cfg, index_price_history=index_prices, trading_calendar=trading_calendar
            )
            rebalance_dates = probe_bt._select_rebalance_dates(
                sorted(str(d) for d in dataset["score_date"].dropna().unique())
            )
            weight_by_date, regime_schedule_local = build_regime_weight_schedule(
                index_rows_for_regime,
                rebalance_dates,
                confirm_days=int(getattr(args, "confirm_days", 5) or 5),
                exit_confirm_days=getattr(args, "exit_confirm_days", None),
                non_bull_exits=bool(getattr(args, "regime_non_bull_exits", False)),
                regime_rule=getattr(args, "regime_rule", None),
            )
            dataset = add_dynamic_blend_column(
                dataset,
                "growth_score",
                "cycle_score",
                weight_by_date,
                REGIME_SCORE_COLUMN,
                DEFAULT_UNCERTAIN_GROWTH_WEIGHT,
            )
            score_sweeps_local = score_sweeps_local + [
                ScoreSweep(name=REGIME_SCORE_VARIANT, score_column=REGIME_SCORE_COLUMN)
            ]
            regime_rule = load_regime_rule(getattr(args, "regime_rule", None))
            regime_config_local = {
                "benchmark": args.index_benchmark_symbol,
                "padded_start": padded_index_start(args.start_date),
                "confirm_days": int(getattr(args, "confirm_days", 5) or 5),
                "exit_confirm_days": getattr(args, "exit_confirm_days", None),
                "non_bull_exits": bool(getattr(args, "regime_non_bull_exits", False)),
                "regime_rule": regime_rule.to_dict(),
                "weights": {"bull": "60:40", "uncertain": "20:80", "bear": "0:100"},
                "rebalance_dates": len(rebalance_dates),
            }
            _log(
                f"Regime weight enabled: benchmark={args.index_benchmark_symbol}, "
                f"rule={regime_rule.rule_id}, rebalance_periods={len(rebalance_dates)}",
                verbose,
            )

        if REGIME_MODE_BULL_G60_ELSE_CASH in regime_mode_values:
            if not index_rows_for_regime:
                raise SystemExit("regime cash overlay requires padded benchmark index rows")
            score_dates = sorted(str(d) for d in dataset["score_date"].dropna().unique())
            intra_period_cash = bool(getattr(args, "intra_period_cash", False))
            schedule_dates = score_dates
            if intra_period_cash:
                schedule_dates = [
                    str(row.get("trade_date"))
                    for row in index_rows_for_regime
                    if row.get("trade_date")
                ]
            cash_on_dates, full_schedule = build_regime_cash_schedule(
                index_rows_for_regime,
                schedule_dates,
                confirm_days=int(getattr(args, "confirm_days", 5) or 5),
                exit_confirm_days=getattr(args, "exit_confirm_days", None),
                non_bull_exits=bool(getattr(args, "regime_non_bull_exits", False)),
                regime_rule=getattr(args, "regime_rule", None),
            )
            if intra_period_cash:
                wanted = set(score_dates)
                regime_cash_schedule_local = [
                    entry for entry in full_schedule if str(entry.get("score_date")) in wanted
                ]
                dataset = backtester.add_live_cash_return_columns(
                    dataset,
                    cash_on_dates,
                    trailing_stop_pct=trailing_value,
                )
            else:
                regime_cash_schedule_local = full_schedule
            _log(
                f"Regime cash overlay: cash_dates={len(cash_on_dates)} / "
                f"{len(schedule_dates)} dates intra_period_cash={intra_period_cash}",
                verbose,
            )

        # Re-attach on the dataset that is actually swept: a trailing variant is
        # rebuilt from the panel and merge_volatility_columns only carries vol_*/
        # atr_*/path columns, so the SW columns on base_dataset never reach it.
        if not dataset.empty and "symbol" in dataset.columns:
            dataset = attach_sw_l1(dataset, db_manager, as_of_column="score_date")
            dataset = attach_sw_l1_regime(dataset, db_manager)
            industry_regime_coverage_local = sw_l1_regime_coverage(dataset)

        _log(
            f"Running sweep: trailing_stop_pct={label}, score_variants={len(score_sweeps_local)}, "
            f"construction_variants={len(variants)}, vol_variants={len(vol_variants)}, "
            f"top_n={','.join(map(str, args.top_n))}",
            verbose,
        )
        part = run_parameter_sweep(
            dataset=dataset,
            score_column=args.score_column,
            top_n_values=args.top_n,
            horizon=args.horizon,
            rebalance_intervals=rebalance_intervals,
            variants=variants,
            transaction_cost=args.transaction_cost,
            index_benchmark_symbol=args.index_benchmark_symbol,
            buy_commission_rate=args.buy_commission_rate,
            sell_commission_rate=args.sell_commission_rate,
            min_commission=args.min_commission,
            stamp_tax_rate=args.stamp_tax_rate,
            transfer_fee_rate=args.transfer_fee_rate,
            index_price_history=index_prices,
            score_sweeps=score_sweeps_local,
            vol_variants=vol_variants,
            volatility_column=args.volatility_column,
            trailing_stop_pct=trailing_value,
            path_gates=path_gate_values,
            regime_modes=regime_mode_values,
            cash_on_dates=cash_on_dates,
            intra_period_cash=bool(getattr(args, "intra_period_cash", False))
            and REGIME_MODE_BULL_G60_ELSE_CASH in regime_mode_values,
            bull_exposures=bull_exposure_values,
            refill_modes=refill_mode_values,
            crowd_exit_ks=crowd_exit_k_values,
            non_bull_actions=non_bull_action_values,
            industry_regime_scales=industry_regime_scale_values,
            price_cache=backtester.price_cache,
            selection_mode=args.selection_mode,
            threshold_lookback_days=args.threshold_lookback_days,
            max_positions=args.max_positions,
            verbose=verbose,
            industry_l1_codes=industry_l1_codes,
        )
        all_rows.extend(part.get("rows") or [])
        payload = part

    payload["rows"] = all_rows
    payload["trailing_stop_pcts"] = [value if value is not None else 0.0 for value in trailing_stop_values]
    payload["path_gates"] = path_gate_values
    payload["regime_modes"] = regime_mode_values
    payload["bull_exposures"] = list(bull_exposure_values)
    payload["refill_modes"] = ["on" if flag else "off" for flag in refill_mode_values]
    payload["crowd_exit_ks"] = list(crowd_exit_k_values)
    payload["non_bull_actions"] = list(non_bull_action_values)
    payload["industry_regime_scales"] = list(industry_regime_scale_values)
    if industry_regime_coverage_local is not None:
        payload["industry_regime_coverage"] = industry_regime_coverage_local
    if dynamic_schedule_local is not None:
        payload["dynamic_config"] = dynamic_config_local
        payload["dynamic_schedule"] = dynamic_schedule_local
    if regime_schedule_local is not None:
        payload["regime_config"] = regime_config_local
        payload["regime_schedule"] = regime_schedule_local
    if regime_cash_schedule_local is not None:
        payload["regime_cash_schedule"] = regime_cash_schedule_local
        payload["cash_on_dates"] = cash_on_dates

    universe_name = universe_title(args.universe_index)
    universe_name_slug = universe_slug(args.universe_index)
    output_path = Path(args.output) if args.output else Path(
        f"reports/portfolio_sweep_{universe_name_slug}_{_output_score_slug(score_sweeps, args.score_column)}_intervals{'-'.join(map(str, rebalance_intervals))}_{args.start_date}_{args.end_date}.html"
    )
    json_path = Path(args.json_output) if args.json_output else output_path.with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    _log("Writing JSON and HTML reports...", verbose)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    html = render_html(
        payload,
        title=f"{universe_name} Growth-Cycle Portfolio Parameter Sweep",
        start_date=args.start_date,
        end_date=args.end_date,
        index_buy_hold_return=calculate_index_buy_hold_return(index_prices, args.start_date, args.end_date),
    )
    output_path.write_text(html, encoding="utf-8")
    print(f"JSON report: {json_path}")
    print(f"HTML report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
