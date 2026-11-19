import numpy as np
import pandas as pd

# Canonical implementation lives in data-access-lib so the portfolio research
# labeler and the Strategy Lab backtest worker cannot drift apart.
from stock_data_access.indicators.k import calculate_k_indicator  # noqa: F401


def _wilder_smooth(values: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing (RMA) used by ADX/DMI."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    series = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype=float)
    if len(series) < period:
        return out
    first = float(series.iloc[:period].sum())
    out.iloc[period - 1] = first
    for idx in range(period, len(series)):
        prev = float(out.iloc[idx - 1])
        out.iloc[idx] = prev - (prev / period) + float(series.iloc[idx])
    return out


def compute_adx_dmi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """Wilder ADX(period), +DI, -DI aligned on ``close`` index.

    Requires equal-length ``high``, ``low``, ``close``. Rows before warm-up
    carry NaN for adx/plus_di/minus_di/dx.
    """
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    h = pd.to_numeric(high, errors="coerce")
    low_s = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    if not (len(h) == len(low_s) == len(c)):
        raise ValueError("high, low, close must have the same length")
    prev_close = c.shift(1)
    tr = pd.concat(
        [(h - low_s).abs(), (h - prev_close).abs(), (low_s - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    up_move = h.diff()
    down_move = -low_s.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr_n = _wilder_smooth(tr.fillna(0.0), period)
    plus_n = _wilder_smooth(plus_dm.fillna(0.0), period)
    minus_n = _wilder_smooth(minus_dm.fillna(0.0), period)
    plus_di = 100.0 * plus_n / tr_n.replace(0, np.nan)
    minus_di = 100.0 * minus_n / tr_n.replace(0, np.nan)
    di_sum = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
    adx = _wilder_smooth(dx.fillna(0.0), period)

    return pd.DataFrame(
        {
            "plus_di": plus_di,
            "minus_di": minus_di,
            "dx": dx,
            "adx": adx,
            "tr": tr,
        },
        index=c.index,
    )
