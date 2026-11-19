"""Shared loaders and stats for the 阴阳谱 (market breadth) studies.

The breadth series is "share of all tradable A-shares closing above their own
N-day MA", keyed by ``(trade_date, ma_period)``.  Production computes it inside
``quant_data.market_spectrum`` (daily, MA5 + MA30) but only from 2024-01-02, and
``volume_price`` only reaches back to 2010-01-04 — hence the local rebuild and
the Tushare-sourced extension in this package.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"

BREADTH_MONGO = DATA_DIR / "breadth_mongo.csv"
BREADTH_TUSHARE = DATA_DIR / "breadth_tushare.csv"
BREADTH_MERGED = DATA_DIR / "breadth_2005_2026.csv"
DAILY_RANGE = DATA_DIR / "daily_range.csv.gz"

INDEX_NAMES = {
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "000852.SH": "中证1000",
    "932000.CSI": "中证2000",
    "000016.SH": "上证50",
}

# 中证1000 was first published 2014-10-17 and 中证2000 on 2023-10-11; everything
# before that is 中证公司's base-date back-calculation, not a live index.  Fine
# for research, but a study that "predicts" a back-filled index is testing a
# hindsight-selected basket — callers should be able to flag it.
INDEX_PUBLISHED = {
    "000300.SH": "20050408",
    "000905.SH": "20070115",
    "000852.SH": "20141017",
    "932000.CSI": "20231011",
    "000016.SH": "20040102",
}

DEFAULT_ENV_FILE = "/home/shuyolin/trading/quant-infrastructure/apps/env/quant-data-engine.env"


def read_env_key(key: str, env_file: str = DEFAULT_ENV_FILE) -> str:
    """Pull a single key out of an env file without evaluating the whole file.

    Sourcing these files in a shell is unsafe — they contain unquoted values with
    spaces that bash tries to execute.
    """
    val = None
    for line in open(env_file, encoding="utf-8", errors="ignore"):
        m = re.match(rf"\s*{re.escape(key)}\s*=\s*(\S+)", line)
        if m:
            val = m.group(1).strip().strip('"').strip("'")
    if not val:
        raise SystemExit(f"{key} not found in {env_file}")
    return val


def mongo_client():
    from pymongo import MongoClient

    return MongoClient(read_env_key("MONGO_URI"), serverSelectionTimeoutMS=8000)


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def resolve_data(path: Path | str) -> Path:
    """Resolve a CLI data path: absolute as given, relative against ``data/``.

    ``--breadth breadth_mongo.csv`` should mean the file in this package's data
    dir regardless of where the caller's shell happens to be sitting.
    """
    p = Path(path)
    return p if p.is_absolute() else (DATA_DIR / p.name)


def load_breadth(path: Path | str = BREADTH_MERGED) -> pd.DataFrame:
    """Load a breadth CSV indexed by trade_date, with ma5/ma30 and n5/n30."""
    p = resolve_data(path)
    if not p.exists():
        raise SystemExit(f"breadth file not found: {p}\n(run build_breadth.py, or pass an absolute --breadth)")
    df = pd.read_csv(p, dtype={"trade_date": str}).set_index("trade_date").sort_index()
    df.index.name = "trade_date"
    return df


def load_index_closes(codes: Optional[Sequence[str]] = None, client=None) -> pd.DataFrame:
    """Raw close levels for the study indices, indexed by trade_date."""
    codes = list(codes) if codes else list(INDEX_NAMES)
    c = client or mongo_client()
    cur = c["quant_data"]["index_prices"].find(
        {"ts_code": {"$in": codes}}, {"ts_code": 1, "trade_date": 1, "close": 1, "_id": 0}
    )
    ip = pd.DataFrame(list(cur))
    ip["trade_date"] = ip["trade_date"].astype(str)
    ip["close"] = pd.to_numeric(ip["close"], errors="coerce")
    px = ip.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    return px[[c_ for c_ in codes if c_ in px.columns]].rename(columns=INDEX_NAMES)


def load_index_ohlc(codes: Optional[Sequence[str]] = None, client=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(close, open) levels — open is needed to model next-open entry."""
    codes = list(codes) if codes else list(INDEX_NAMES)
    c = client or mongo_client()
    cur = c["quant_data"]["index_prices"].find(
        {"ts_code": {"$in": codes}},
        {"ts_code": 1, "trade_date": 1, "close": 1, "open": 1, "_id": 0},
    )
    ip = pd.DataFrame(list(cur))
    ip["trade_date"] = ip["trade_date"].astype(str)
    for f in ("close", "open"):
        ip[f] = pd.to_numeric(ip[f], errors="coerce")
    cl = ip.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    op = ip.pivot(index="trade_date", columns="ts_code", values="open").sort_index()
    keep = [c_ for c_ in codes if c_ in cl.columns]
    return cl[keep].rename(columns=INDEX_NAMES), op[keep].rename(columns=INDEX_NAMES)


def align_to_breadth(px: pd.DataFrame, breadth_index: pd.Index) -> pd.DataFrame:
    """Reindex price levels onto the breadth calendar, carrying the last close.

    The breadth calendar is the master clock: a stock trading day that the index
    lacks would otherwise silently drop the observation.
    """
    full = pd.Index(sorted(set(breadth_index) | set(px.index)))
    return px.reindex(full).ffill().reindex(breadth_index)


# --------------------------------------------------------------------------- #
# event / stats helpers
# --------------------------------------------------------------------------- #
def episodes(flag: pd.Series) -> pd.Index:
    """First day of each consecutive run where ``flag`` is True.

    Consecutive days below a threshold are one market event, not N samples —
    without this a single crash inflates the sample count enormously.
    """
    f = flag.fillna(False).astype(bool).to_numpy()
    return flag.index[f & ~np.r_[False, f[:-1]]]


def episode_groups(ep: pd.Index, calendar: pd.Index, gap: int = 60) -> np.ndarray:
    """Label episodes into distinct market events, splitting on >``gap`` quiet days."""
    if len(ep) == 0:
        return np.array([], dtype=int)
    pos = np.array([calendar.get_loc(d) for d in ep])
    return np.r_[0, np.cumsum(np.diff(pos) > gap)]


def forward_return(px: pd.Series, horizon: int, entry_lag: int = 0) -> pd.Series:
    """Forward return over ``horizon`` bars, entered ``entry_lag`` bars later.

    ``entry_lag=0`` is same-day close entry; ``entry_lag=1`` is next-open-ish
    (caller passes the open series).
    """
    return px.shift(-(horizon + entry_lag)) / px.shift(-entry_lag) - 1


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def circular_perm_p(cond_up: float, cond_index: pd.Index, fwd: pd.Series,
                    n_perm: int = 3000, seed: int = 20261013) -> float:
    """Two-sided p for a conditional up-rate, by circularly shifting ``fwd``.

    Circular shifting preserves the autocorrelation of the return series and
    keeps the conditioning set's own clustering intact while destroying the
    cross-relationship — a plain random shuffle would understate the null.
    """
    rng = np.random.default_rng(seed)
    arr = fwd.to_numpy(dtype=float)
    pos = [fwd.index.get_loc(d) for d in cond_index]
    null = np.empty(n_perm)
    for i in range(n_perm):
        shifted = np.roll(arr, int(rng.integers(1, len(arr))))
        v = shifted[pos]
        v = v[~np.isnan(v)]
        null[i] = (v > 0).mean() if len(v) > 3 else np.nan
    null = null[~np.isnan(null)]
    if len(null) == 0:
        return np.nan
    return float(min(1.0, 2 * min((null >= cond_up).mean(), (null <= cond_up).mean())))


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Least squares with intercept prepended; returns (coef, R^2)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    A = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ b
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b, float(r2)


def slice_period(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    out = df
    if start:
        out = out[out.index >= start]
    if end:
        out = out[out.index <= end]
    return out


def is_backfilled(nm: str, asof: str) -> bool:
    """True if ``asof`` predates the index's real publication date.

    For the pre-publication stretch the series is 中证公司's base-date
    back-calculation over a basket chosen with hindsight, so a study that only
    works there is testing a hindsight-selected portfolio.
    """
    code = {v: k for k, v in INDEX_NAMES.items()}.get(nm)
    pub = INDEX_PUBLISHED.get(code or "", "")
    return bool(pub) and asof < pub
