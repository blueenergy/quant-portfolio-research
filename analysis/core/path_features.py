"""Point-in-time path features for filter-then-rank portfolio selection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PATH_FEATURE_COLUMNS = ("ret_20", "above_ma60", "mdd60")
MA_STACK_FEATURE_COLUMNS = ("above_ma5", "above_ma10", "ma_zone")
PATH_AND_MA_FEATURE_COLUMNS = PATH_FEATURE_COLUMNS + MA_STACK_FEATURE_COLUMNS
RET_WINDOW = 20
MA_WINDOW = 60
MA5_WINDOW = 5
MA10_WINDOW = 10
MDD_WINDOW = 60

VALID_PATH_GATES = frozenset({"off", "ret20_pos", "above_ma60"})

MA_ZONE_STRONG = "strong"
MA_ZONE_WARN = "warn"
MA_ZONE_WEAK = "weak"


def classify_ma_zone(
    close: float,
    ma5: float,
    ma10: float,
) -> Optional[str]:
    """strong: close > MA5; warn: MA10 < close <= MA5; weak: close <= MA10."""
    try:
        c = float(close)
        m10 = float(ma10)
    except (TypeError, ValueError):
        return None
    if c <= 0 or np.isnan(c) or np.isnan(m10):
        return None
    if c <= m10:
        return MA_ZONE_WEAK
    try:
        m5 = float(ma5)
    except (TypeError, ValueError):
        return MA_ZONE_STRONG
    if np.isnan(m5):
        return MA_ZONE_STRONG
    if c <= m5:
        return MA_ZONE_WARN
    return MA_ZONE_STRONG


def _build_symbol_path_frame(
    price_rows: Sequence[Dict[str, Any]],
    *,
    ret_window: int = RET_WINDOW,
    ma_window: int = MA_WINDOW,
    mdd_window: int = MDD_WINDOW,
) -> pd.DataFrame:
    """Rolling path features indexed by trade_date (ascending)."""
    if not price_rows:
        return pd.DataFrame()

    frame = pd.DataFrame(price_rows)
    if "trade_date" not in frame.columns or "close" not in frame.columns:
        return pd.DataFrame()

    frame = frame.sort_values("trade_date").copy()
    frame["_trade_dt"] = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    frame = frame.dropna(subset=["_trade_dt"])
    if frame.empty:
        return pd.DataFrame()

    close = pd.to_numeric(frame["close"], errors="coerce")
    positive_close = close.where(close > 0)
    shifted = positive_close.shift(int(ret_window))
    frame["ret_20"] = positive_close / shifted - 1.0
    ma = positive_close.rolling(int(ma_window), min_periods=int(ma_window)).mean()
    frame["above_ma60"] = (positive_close > ma).astype(float)
    roll_high = positive_close.rolling(int(mdd_window), min_periods=max(1, int(mdd_window) // 2)).max()
    frame["mdd60"] = 1.0 - positive_close / roll_high
    ma5 = positive_close.rolling(int(MA5_WINDOW), min_periods=int(MA5_WINDOW)).mean()
    ma10 = positive_close.rolling(int(MA10_WINDOW), min_periods=int(MA10_WINDOW)).mean()
    frame["above_ma5"] = (positive_close > ma5).astype(float)
    frame["above_ma10"] = (positive_close > ma10).astype(float)
    frame["ma_zone"] = [
        classify_ma_zone(c, m5, m10)
        for c, m5, m10 in zip(positive_close.to_numpy(), ma5.to_numpy(), ma10.to_numpy())
    ]
    return frame


def add_path_feature_columns(
    dataset: pd.DataFrame,
    price_cache: Dict[str, List[Dict[str, Any]]],
    *,
    ret_window: int = RET_WINDOW,
    ma_window: int = MA_WINDOW,
    mdd_window: int = MDD_WINDOW,
) -> pd.DataFrame:
    """Attach path features using only prices on or before ``score_date``."""
    if dataset.empty:
        return dataset

    df = dataset.copy()
    for column in PATH_AND_MA_FEATURE_COLUMNS:
        if column not in df.columns:
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

    for symbol, symbol_scores in score_rows.groupby("_symbol", sort=False):
        feature_frame = _build_symbol_path_frame(
            price_cache.get(symbol) or [],
            ret_window=ret_window,
            ma_window=ma_window,
            mdd_window=mdd_window,
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
        for column in PATH_AND_MA_FEATURE_COLUMNS:
            if column not in merged.columns:
                continue
            if column == "ma_zone":
                df.iloc[row_ids, df.columns.get_loc(column)] = merged[column].to_numpy()
            else:
                df.iloc[row_ids, df.columns.get_loc(column)] = merged[column].where(pd.notna(merged[column]), None).to_numpy()
    return df


def path_gate_mask(frame: pd.DataFrame, path_gate: str) -> pd.Series:
    """Return boolean mask for rows passing ``path_gate`` (False when feature missing)."""
    gate = str(path_gate or "off").strip().lower()
    if gate == "off":
        return pd.Series(True, index=frame.index)
    if gate not in VALID_PATH_GATES:
        raise ValueError(f"Unsupported path_gate: {path_gate}")

    if gate == "ret20_pos":
        ret = pd.to_numeric(frame.get("ret_20"), errors="coerce")
        return ret > 0
    if gate == "above_ma60":
        above = pd.to_numeric(frame.get("above_ma60"), errors="coerce")
        return above > 0
    return pd.Series(True, index=frame.index)


def apply_path_gate(frame: pd.DataFrame, path_gate: str) -> pd.DataFrame:
    """Filter ``frame`` to rows passing ``path_gate``; preserves sort order."""
    gate = str(path_gate or "off").strip().lower()
    if gate == "off":
        return frame
    mask = path_gate_mask(frame, gate)
    if not isinstance(mask, pd.Series):
        mask = pd.Series([bool(mask)] * len(frame), index=frame.index)
    else:
        mask = mask.fillna(False)
    return frame.loc[mask].copy()


def future_max_drawdown_from_cache(
    price_cache: Dict[str, List[Dict[str, Any]]],
    dataset: pd.DataFrame,
    horizon: int,
) -> List[Optional[float]]:
    """Per-row max drawdown from entry to horizon exit (same semantics as entry_risk report)."""
    exit_col = f"exit_date_{horizon}d"
    if not price_cache or exit_col not in dataset.columns:
        return [None] * len(dataset)

    cache: Dict[str, pd.DataFrame] = {}
    out: List[Optional[float]] = []
    for row in dataset.to_dict("records"):
        symbol = str(row.get("symbol") or "")
        entry_price = row.get("entry_price")
        try:
            entry_price_f = float(entry_price) if entry_price is not None else None
        except (TypeError, ValueError):
            entry_price_f = None
        entry_date = str(row.get("entry_date") or row.get("score_date") or "")
        exit_date = str(row.get(exit_col) or "")
        if not symbol or entry_price_f is None or entry_price_f <= 0 or not entry_date or not exit_date:
            out.append(None)
            continue
        if symbol not in cache:
            prices = pd.DataFrame(price_cache.get(symbol) or [])
            if prices.empty or "trade_date" not in prices.columns or "close" not in prices.columns:
                cache[symbol] = pd.DataFrame()
            else:
                prices = prices[["trade_date", "close"]].copy()
                prices["trade_date"] = prices["trade_date"].astype(str)
                prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
                cache[symbol] = prices.dropna().sort_values("trade_date")
        prices = cache[symbol]
        if prices.empty:
            out.append(None)
            continue
        window = prices[(prices["trade_date"] >= entry_date) & (prices["trade_date"] <= exit_date)]
        if window.empty:
            out.append(None)
        else:
            out.append(float(window["close"].min() / entry_price_f - 1.0))
    return out
