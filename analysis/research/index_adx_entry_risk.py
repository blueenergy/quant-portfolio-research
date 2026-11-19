"""Frozen ADX/DMI entry-risk overlay for growth-momentum research.

Pre-registered rules (2026-10-10):
  di_bear_strengthening, adx_peak_fade, overheat_fade
Exposure tiers: 1.0 / 0.5 / 0.0 with 2-day enter/exit hysteresis; T close -> T+1.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.research.regime_switch_rule_engine import (
    COST_STOCK,
    G3_TOPS,
    _close_series,
    daily_factors,
    window_summary,
)
from shared.regime_label import label_as_of, label_index_regime, padded_index_start
from utils.market_indicators import calculate_k_indicator, compute_adx_dmi

ADX_PERIOD = 14
ADX_PEAK_LOOKBACK = 10
ADX_PEAK_MIN = 30.0
ADX_PEAK_DROP = 5.0
DI_SPREAD_NARROW_DAYS = 3
HIPOS_WINDOW = 120
HIPOS_OVERHEAT = 0.95
EMA_ALPHA = 0.967
EMA_DIST_LOOKBACK = 250
EMA_DIST_PCTL = 0.90
CONFIRM_DAYS = 2
EXIT_CONFIRM_DAYS = 2

EXPOSURE_FULL = 1.0
EXPOSURE_WARN = 0.5
EXPOSURE_OFF = 0.0

RULE_IDS = ("di_bear_strengthening", "adx_peak_fade", "overheat_fade")


@dataclass(frozen=True)
class OhlcCoverage:
    symbol: str
    n_rows: int
    high_pct: float
    low_pct: float
    close_pct: float
    ok: bool
    message: str


def validate_ohlc_coverage(index_rows: Sequence[Mapping[str, Any]], symbol: str = "") -> OhlcCoverage:
    frame = _index_ohlc_frame(index_rows)
    n = len(frame)
    if n == 0:
        return OhlcCoverage(symbol, 0, 0.0, 0.0, 0.0, False, "empty index series")
    high_pct = float(frame["high"].notna().mean())
    low_pct = float(frame["low"].notna().mean())
    close_pct = float(frame["close"].notna().mean())
    ok = high_pct >= 0.99 and low_pct >= 0.99 and close_pct >= 0.99
    msg = "ok" if ok else f"missing OHLC high={high_pct:.1%} low={low_pct:.1%}"
    return OhlcCoverage(symbol, n, high_pct, low_pct, close_pct, ok, msg)


def _index_ohlc_frame(index_rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(index_rows))
    date_col = next((c for c in ("trade_date", "date") if c in frame.columns), None)
    if date_col is None:
        return pd.DataFrame()
    close_col = next((c for c in ("close", "close_price", "adj_close") if c in frame.columns), None)
    high_col = next((c for c in ("high", "high_price") if c in frame.columns), None)
    low_col = next((c for c in ("low", "low_price") if c in frame.columns), None)
    if close_col is None:
        return pd.DataFrame()
    out = frame[[date_col, close_col]].rename(columns={date_col: "trade_date", close_col: "close"})
    if high_col:
        out["high"] = pd.to_numeric(frame[high_col], errors="coerce")
    else:
        out["high"] = np.nan
    if low_col:
        out["low"] = pd.to_numeric(frame[low_col], errors="coerce")
    else:
        out["low"] = np.nan
    out["trade_date"] = out["trade_date"].astype(str)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["trade_date", "close"]).sort_values("trade_date").drop_duplicates("trade_date")
    if out["high"].isna().all() or out["low"].isna().all():
        raise ValueError("index rows lack high/low — ADX requires full OHLC")
    return out.reset_index(drop=True)


def build_indicator_frame(index_rows: Sequence[Mapping[str, Any]], *, adx_period: int = ADX_PERIOD) -> pd.DataFrame:
    out = _index_ohlc_frame(index_rows)
    if out.empty:
        return out
    idx = out["trade_date"]
    high = pd.Series(out["high"].to_numpy(), index=idx)
    low = pd.Series(out["low"].to_numpy(), index=idx)
    close = pd.Series(out["close"].to_numpy(), index=idx)
    dmi = compute_adx_dmi(high, low, close, period=adx_period)
    for col in dmi.columns:
        out[col] = dmi[col].to_numpy()
    out["di_spread"] = out["plus_di"] - out["minus_di"]
    rolling_high = close.rolling(HIPOS_WINDOW, min_periods=HIPOS_WINDOW // 2).max()
    out["hipos120"] = (close / rolling_high).reindex(idx).to_numpy()
    if len(close) >= 6:
        start = close.index[5]
        ema = calculate_k_indicator(close, start_date=str(start), alpha=EMA_ALPHA)
        out["ema60"] = out["trade_date"].map(ema)
        dist = out["close"] / out["ema60"] - 1.0
        out["ema60_dist"] = dist
        q90 = dist.rolling(EMA_DIST_LOOKBACK, min_periods=60).quantile(EMA_DIST_PCTL)
        out["ema60_dist_high"] = dist >= q90
    else:
        out["ema60"] = np.nan
        out["ema60_dist"] = np.nan
        out["ema60_dist_high"] = False
    out["adx_rising"] = out["adx"] > out["adx"].shift(1)
    out["adx_peak_10"] = out["adx"].rolling(ADX_PEAK_LOOKBACK, min_periods=1).max()
    out["di_spread_narrow_3d"] = (
        out["di_spread"].diff().rolling(DI_SPREAD_NARROW_DAYS).sum() < 0
    ) & (out["di_spread"].diff() < 0)
    return out


def rule_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Boolean columns per frozen rule."""
    out = frame.copy()
    out["di_bear_strengthening"] = (
        (out["minus_di"] > out["plus_di"])
        & out["adx_rising"].fillna(False)
    )
    had_peak = out["adx_peak_10"] >= ADX_PEAK_MIN
    faded = out["adx"] <= (out["adx_peak_10"] - ADX_PEAK_DROP)
    out["adx_peak_fade"] = had_peak & faded & out["di_spread_narrow_3d"].fillna(False)
    overheat = (out["hipos120"] >= HIPOS_OVERHEAT) | out["ema60_dist_high"].fillna(False)
    out["overheat_fade"] = out["adx_peak_fade"] & overheat.fillna(False)
    return out


def raw_exposure_from_rules(frame: pd.DataFrame) -> pd.Series:
    """Map rules to exposure tier before hysteresis."""
    exposure = pd.Series(EXPOSURE_FULL, index=frame.index, dtype=float)
    exposure = exposure.where(~frame["adx_peak_fade"].fillna(False), EXPOSURE_WARN)
    exposure = exposure.where(~frame["overheat_fade"].fillna(False), EXPOSURE_OFF)
    exposure = exposure.where(~frame["di_bear_strengthening"].fillna(False), EXPOSURE_OFF)
    return exposure


def apply_hysteresis(
    target: pd.Series,
    *,
    confirm_days: int = CONFIRM_DAYS,
    exit_confirm_days: int = EXIT_CONFIRM_DAYS,
) -> pd.Series:
    """Asymmetric confirmation: enter lower tier faster, exit slower."""
    values = target.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    current = EXPOSURE_FULL
    pending: Optional[float] = None
    pending_count = 0
    for idx, desired in enumerate(values):
        if math.isnan(desired):
            out[idx] = current
            continue
        desired = float(desired)
        if desired < current:
            need = confirm_days
        elif desired > current:
            need = exit_confirm_days
        else:
            pending = None
            pending_count = 0
            out[idx] = current
            continue
        if pending != desired:
            pending = desired
            pending_count = 1
        else:
            pending_count += 1
        if pending_count >= need:
            current = pending
            pending = None
            pending_count = 0
        out[idx] = current
    return pd.Series(out, index=target.index)


def exposure_schedule_t_plus_one(frame: pd.DataFrame) -> pd.Series:
    """Signal at close T; position applies from T+1 (shift forward)."""
    flagged = rule_flags(frame) if "adx_peak_fade" not in frame.columns else frame
    raw = raw_exposure_from_rules(flagged)
    smooth = apply_hysteresis(raw)
    dates = frame["trade_date"].astype(str)
    series = pd.Series(smooth.to_numpy(), index=dates)
    return series.shift(1).fillna(EXPOSURE_FULL)


def exposure_on_rebalance_dates(exposure: pd.Series, rebalance_dates: Sequence[str]) -> List[float]:
    out: List[float] = []
    for date in rebalance_dates:
        d = str(date)
        if d in exposure.index:
            out.append(float(exposure.loc[d]))
        else:
            prior = exposure[exposure.index < d]
            out.append(float(prior.iloc[-1]) if len(prior) else EXPOSURE_FULL)
    return out


def merge_regime_and_adx(
    regime_labels: pd.DataFrame,
    adx_exposure: pd.Series,
) -> pd.Series:
    """Regime non-bull -> 0; else ADX tier."""
    bull = regime_labels.set_index("trade_date")["label"] == "bull"
    bull = bull.reindex(adx_exposure.index).fillna(False)
    combined = adx_exposure.copy()
    combined = combined.where(bull, EXPOSURE_OFF)
    return combined


def index_daily_factors(
    index_rows: Sequence[Mapping[str, Any]],
    exposure: pd.Series,
    *,
    cost_rate: float = COST_STOCK,
) -> pd.DataFrame:
    pos = exposure.reindex(_close_series(index_rows).index)
    return daily_factors(index_rows, pos, cost_rate=cost_rate)


def peak_event_lags(
    exposure: pd.Series,
    peaks: Sequence[Tuple[str, str]],
    *,
    threshold: float = EXPOSURE_WARN - 1e-9,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for symbol, peak_date in peaks:
        sub = exposure[exposure.index >= peak_date]
        cut = sub[sub <= threshold]
        lag = None
        if len(cut):
            first = str(cut.index[0])
            lag = _trading_day_gap(peak_date, first, exposure.index)
        rows.append({"symbol": symbol, "peak_date": peak_date, "first_reduce_date": cut.index[0] if len(cut) else None, "lag_days": lag})
    return rows


def _trading_day_gap(start: str, end: str, calendar: pd.Index) -> int:
    cal = [str(d) for d in calendar]
    if start not in cal or end not in cal:
        return -1
    return cal.index(end) - cal.index(start)


def forward_index_stats(
    frame: pd.DataFrame,
    exposure: pd.Series,
    horizons: Sequence[int] = (5, 20, 60),
) -> Dict[str, Any]:
    close = pd.Series(frame["close"].to_numpy(), index=frame["trade_date"].astype(str))
    out: Dict[str, Any] = {}
    for h in horizons:
        fwd = close.shift(-h) / close - 1.0
        for tier, mask_fn in (
            ("full", lambda e: e >= 0.99),
            ("warn", lambda e: (e >= 0.45) & (e < 0.99)),
            ("off", lambda e: e < 0.45),
        ):
            mask = mask_fn(exposure.reindex(close.index).fillna(EXPOSURE_FULL))
            sample = fwd[mask].dropna()
            out[f"h{h}_{tier}_mean"] = float(sample.mean()) if len(sample) else None
            out[f"h{h}_{tier}_n"] = int(len(sample))
    return out


def regime_crosstab(frame: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    merged = frame.merge(labels[["trade_date", "label"]], on="trade_date", how="left")
    merged["tier"] = pd.cut(
        raw_exposure_from_rules(merged),
        bins=[-0.1, 0.25, 0.75, 1.1],
        labels=["off", "warn", "full"],
    )
    return pd.crosstab(merged["label"], merged["tier"])


def walk_forward_holdout(
    index_rows: Sequence[Mapping[str, Any]],
    *,
    fold_years: Sequence[int] = tuple(range(2015, 2027)),
    cost_rate: float = COST_STOCK,
) -> Dict[str, Any]:
    frame = rule_flags(build_indicator_frame(index_rows))
    exposure = exposure_schedule_t_plus_one(frame)
    factors = index_daily_factors(index_rows, exposure, cost_rate=cost_rate)
    prod = daily_factors(
        index_rows,
        pd.Series(1.0, index=factors.index),
        cost_rate=cost_rate,
    )
    holdouts: Dict[int, Dict[str, Any]] = {}
    wins = 0
    recent_wins = 0
    for year in fold_years:
        start, end = f"{year}0101", f"{year}1231"
        adx_sum = window_summary(factors, start, end)
        prod_sum = window_summary(prod, start, end)
        win = adx_sum["excess"] > prod_sum["excess"]
        if win:
            wins += 1
        if year in (2024, 2025, 2026):
            if win:
                recent_wins += 1
        holdouts[year] = {
            "adx_excess": adx_sum["excess"],
            "prod_excess": prod_sum["excess"],
            "adx_mdd": adx_sum["mdd"],
            "prod_mdd": prod_sum["mdd"],
            "win": win,
        }
    return {
        "holdouts": holdouts,
        "win_rate": wins / max(len(fold_years), 1),
        "recent_wins_2024_2026": recent_wins,
    }


def run_diagnostics(
    index_rows: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
) -> Dict[str, Any]:
    coverage = validate_ohlc_coverage(index_rows, symbol)
    frame = build_indicator_frame(index_rows)
    flagged = rule_flags(frame)
    exposure = exposure_schedule_t_plus_one(flagged)
    labels = label_index_regime(list(index_rows))
    ctab = regime_crosstab(flagged, labels)
    flips = int((exposure.diff().abs() > 0.25).sum())
    return {
        "symbol": symbol,
        "coverage": coverage.__dict__,
        "n_days": len(frame),
        "rule_hit_rates": {
            rule: float(flagged[rule].mean()) for rule in RULE_IDS if rule in flagged.columns
        },
        "exposure_mean": float(exposure.mean()),
        "exposure_flips": flips,
        "regime_crosstab": ctab.to_dict(),
        "peak_lags": peak_event_lags(exposure, G3_TOPS),
        "forward_stats": forward_index_stats(flagged, exposure),
        "walk_forward": walk_forward_holdout(index_rows),
    }


def load_index_rows_for_research(
    db: Any,
    symbol: str,
    start_date: str,
    end_date: str = "99991231",
) -> List[Dict[str, Any]]:
    from analysis.core.benchmark_prices import load_index_benchmark_prices

    padded = padded_index_start(start_date)
    rows = load_index_benchmark_prices(db, symbol, padded)
    rows = [r for r in rows if str(r.get("trade_date", "")) <= end_date]
    return rows


def write_diagnostics_json(payload: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
