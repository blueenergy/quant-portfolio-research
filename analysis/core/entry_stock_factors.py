"""Point-in-time entry-risk factors (hipos120 / mainflow20 / distrib) for sweep masking."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PRICE_PROJECTION = {
    "_id": 0,
    "symbol": 1,
    "ts_code": 1,
    "trade_date": 1,
    "close": 1,
    "vol": 1,
    "volume": 1,
    "amount": 1,
}


def _to_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _price_amount_wan(row: Dict[str, Any]) -> Optional[float]:
    amount = _to_float(row.get("amount"))
    if amount is None:
        volume = _to_float(row.get("vol") if row.get("vol") is not None else row.get("volume"))
        close = _to_float(row.get("close"))
        if volume is not None and close is not None:
            return volume * 100 * close / 10000
    if amount is None or amount <= 0:
        return None
    return amount / 10


def _main_net_from_money_flow(row: Dict[str, Any]) -> Optional[float]:
    buy_elg = _to_float(row.get("buy_elg_amount"))
    buy_lg = _to_float(row.get("buy_lg_amount"))
    sell_elg = _to_float(row.get("sell_elg_amount"))
    sell_lg = _to_float(row.get("sell_lg_amount"))
    if any(value is not None for value in (buy_elg, buy_lg, sell_elg, sell_lg)):
        return (buy_elg or 0.0) + (buy_lg or 0.0) - (sell_elg or 0.0) - (sell_lg or 0.0)
    return _to_float(row.get("net_mf_amount"))


def build_symbol_entry_factor_frame(
    db: Any,
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    flow_window: int = 20,
    high_window: int = 120,
    vol_fast: int = 5,
    vol_slow: int = 20,
) -> Optional[pd.DataFrame]:
    try:
        price_rows = db.get_volume_price_data(
            symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="none",
            projection=PRICE_PROJECTION,
        )
    except TypeError:
        price_rows = db.get_volume_price_data(symbol, start_date=start_date, end_date=end_date, adjust="none")
    if not price_rows:
        return None

    prices = pd.DataFrame(price_rows)
    if prices.empty or "trade_date" not in prices.columns or "close" not in prices.columns:
        return None
    prices["trade_date"] = prices["trade_date"].astype(str)
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices.dropna(subset=["close"])
    prices = prices[prices["close"] > 0].sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    if prices.empty:
        return None

    out = pd.DataFrame({"trade_date": prices["trade_date"].to_numpy()})
    close = prices["close"].reset_index(drop=True)
    rolling_high = close.rolling(high_window, min_periods=high_window // 2).max()
    out["hipos120"] = close / rolling_high
    out["distance_to_120d_high_pct"] = (out["hipos120"] - 1.0) * 100.0
    out["ret_20d"] = close / close.shift(20) - 1.0
    out["ret_60d"] = close / close.shift(60) - 1.0

    volume_col = "vol" if "vol" in prices.columns else "volume" if "volume" in prices.columns else None
    if volume_col:
        volume = pd.to_numeric(prices[volume_col], errors="coerce").reset_index(drop=True)
        out["volratio"] = (
            volume.rolling(vol_fast, min_periods=vol_fast).mean()
            / volume.rolling(vol_slow, min_periods=vol_slow).mean()
        )
    else:
        out["volratio"] = np.nan

    amount_wan = pd.Series([_price_amount_wan(row) for row in prices.to_dict("records")])
    money_rows = db.get_money_flow_data(symbol, start_date=start_date, end_date=end_date)
    if money_rows:
        money = pd.DataFrame(money_rows)
        money["trade_date"] = money["trade_date"].astype(str)
        money["main_net"] = [_main_net_from_money_flow(row) for row in money.to_dict("records")]
        money = money[["trade_date", "main_net"]].dropna().drop_duplicates("trade_date", keep="last")
        merged = out.merge(money, on="trade_date", how="left")
        num = pd.to_numeric(merged["main_net"], errors="coerce").rolling(
            flow_window, min_periods=flow_window // 2
        ).sum()
        den = amount_wan.rolling(flow_window, min_periods=flow_window // 2).sum()
        merged["mainflow20"] = num / den.where(den > 0)
        out["main_net_20d"] = num
        out["amount_20d_wan"] = den
        out["mainflow20"] = merged["mainflow20"].to_numpy()
    else:
        out["main_net_20d"] = np.nan
        out["amount_20d_wan"] = np.nan
        out["mainflow20"] = np.nan

    out["distrib"] = out["hipos120"] * (-out["mainflow20"])
    out["symbol"] = symbol
    return out[
        [
            "symbol",
            "trade_date",
            "hipos120",
            "distance_to_120d_high_pct",
            "mainflow20",
            "main_net_20d",
            "amount_20d_wan",
            "volratio",
            "distrib",
            "ret_20d",
            "ret_60d",
        ]
    ]


def attach_factors_asof(dataset: pd.DataFrame, factor_frame: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty or factor_frame.empty:
        return dataset
    df = dataset.copy()
    df["symbol"] = df["symbol"].astype(str)
    df["score_date"] = df["score_date"].astype(str)
    df["_row_id"] = np.arange(len(df))
    df["_sd"] = pd.to_numeric(df["score_date"], errors="coerce").astype("Int64")

    ff = factor_frame.copy()
    ff["symbol"] = ff["symbol"].astype(str)
    ff["trade_date"] = ff["trade_date"].astype(str)
    ff["_td"] = pd.to_numeric(ff["trade_date"], errors="coerce").astype("Int64")
    ff = ff.dropna(subset=["_td"])

    parts: List[pd.DataFrame] = []
    ff_by_symbol = dict(tuple(ff.groupby("symbol", sort=False)))
    for symbol, group in df.groupby("symbol", sort=False):
        right = ff_by_symbol.get(symbol)
        left = group.dropna(subset=["_sd"]).sort_values("_sd")
        if right is None or right.empty or left.empty:
            parts.append(group)
            continue
        merged = pd.merge_asof(
            left.astype({"_sd": "int64"}),
            right.sort_values("_td").drop(columns=["symbol", "trade_date"]).astype({"_td": "int64"}),
            left_on="_sd",
            right_on="_td",
            direction="backward",
        )
        parts.append(merged)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values("_row_id").drop(columns=["_row_id", "_sd", "_td"], errors="ignore").reset_index(drop=True)


def build_entry_factor_panel(
    db: Any,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    *,
    max_workers: int = 8,
) -> pd.DataFrame:
    unique = sorted({str(s) for s in symbols if s})
    frames: List[pd.DataFrame] = []

    def _one(symbol: str) -> Optional[pd.DataFrame]:
        return build_symbol_entry_factor_frame(db, symbol, start_date, end_date)

    if max_workers <= 1:
        for symbol in unique:
            frame = _one(symbol)
            if frame is not None:
                frames.append(frame)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_one, symbol): symbol for symbol in unique}
            for future in as_completed(futures):
                frame = future.result()
                if frame is not None:
                    frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["symbol", "trade_date", "hipos120", "distrib"])
    out = pd.concat(frames, ignore_index=True)
    out["symbol"] = out["symbol"].astype(str)
    out["trade_date"] = out["trade_date"].astype(str)
    return out.drop_duplicates(["symbol", "trade_date"], keep="last")


def factors_aligned_to_dataset(dataset: pd.DataFrame, factor_panel: pd.DataFrame) -> pd.DataFrame:
    if factor_panel.empty or dataset.empty:
        return pd.DataFrame(columns=["symbol", "score_date", "hipos120", "distrib"])
    keys = dataset[["symbol", "score_date"]].drop_duplicates().copy()
    attached = attach_factors_asof(keys, factor_panel)
    cols = [c for c in ("symbol", "score_date", "hipos120", "distrib") if c in attached.columns]
    return attached[cols]


def exclusion_masks(factors: pd.DataFrame) -> pd.DataFrame:
    out = factors[["symbol", "score_date", "hipos120", "distrib"]].copy()
    out["exclude_hipos97"] = pd.to_numeric(out["hipos120"], errors="coerce") >= 0.97
    distrib = pd.to_numeric(out["distrib"], errors="coerce")
    out["exclude_distrib05"] = distrib >= 0.05
    return out


def mask_dataset_scores(
    dataset: pd.DataFrame,
    score_column: str,
    masks: pd.DataFrame,
    exclude_column: str,
) -> pd.DataFrame:
    work = dataset.merge(
        masks[["symbol", "score_date", exclude_column]],
        on=["symbol", "score_date"],
        how="left",
    )
    out = dataset.copy()
    drop = work[exclude_column].fillna(False).astype(bool).to_numpy()
    out.loc[drop, score_column] = np.nan
    return out
