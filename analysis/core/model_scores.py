"""Load model scores from ``quant_ml.model_predictions`` and merge them into panels.

This lives in ``analysis.core`` because the portfolio research worker (an
``entry``/``tools`` layer module) consumes it. The draft ``analysis.research``
layer may import core, but production code must not import draft code.

The adapter stays opt-in by default: ``MODEL_SCORE_ADAPTER_ENABLED=1`` is
required, and production scoring never writes ``stock_scores``.
"""

from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd
from pymongo import MongoClient


def adapter_enabled() -> bool:
    return os.getenv("MODEL_SCORE_ADAPTER_ENABLED", "0").strip() in {"1", "true", "yes", "on"}


def load_model_scores(
    model_id: str,
    *,
    mongo_uri: Optional[str] = None,
    db_name: Optional[str] = None,
    trade_date: Optional[str] = None,
) -> pd.DataFrame:
    """Return ts_code/score for one model and optional as-of date."""
    if not adapter_enabled():
        return pd.DataFrame(columns=["ts_code", "model_score"])
    uri = mongo_uri or os.getenv("MONGO_URI", "")
    db = db_name or os.getenv("QUANT_ML_DB", "quant_ml")
    if not uri:
        raise RuntimeError("MONGO_URI required when MODEL_SCORE_ADAPTER_ENABLED=1")
    client = MongoClient(uri)
    coll = client[db]["model_predictions"]
    query = {"model_id": model_id}
    if trade_date:
        query["trade_date"] = trade_date
    rows = list(coll.find(query, {"_id": 0, "ts_code": 1, "score": 1, "trade_date": 1}))
    if not rows:
        return pd.DataFrame(columns=["ts_code", "model_score", "trade_date"])
    frame = pd.DataFrame(rows)
    frame = frame.rename(columns={"score": "model_score"})
    return frame


def load_model_scores_range(
    model_id: str,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    mongo_uri: Optional[str] = None,
    db_name: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> pd.DataFrame:
    """Bulk load one model's scores over a date range.

    The per-date loader above is fine for a handful of dates; a multi-year
    research panel asks for thousands and would issue one query each. This
    version uses a single range query.
    """
    if not (adapter_enabled() if enabled is None else enabled):
        return pd.DataFrame(columns=["ts_code", "score", "trade_date"])
    uri = mongo_uri or os.getenv("MONGO_URI", "")
    db = db_name or os.getenv("QUANT_ML_DB", "quant_ml")
    if not uri:
        raise RuntimeError("MONGO_URI required when MODEL_SCORE_ADAPTER_ENABLED=1")
    query = {"model_id": model_id}
    date_filter = {}
    if start_date:
        date_filter["$gte"] = str(start_date)
    if end_date:
        date_filter["$lte"] = str(end_date)
    if date_filter:
        query["trade_date"] = date_filter
    client = MongoClient(uri)
    try:
        rows = list(
            client[db]["model_predictions"].find(
                query, {"_id": 0, "ts_code": 1, "score": 1, "trade_date": 1}
            )
        )
    finally:
        client.close()
    if not rows:
        return pd.DataFrame(columns=["ts_code", "score", "trade_date"])
    return pd.DataFrame(rows)


def load_model_score_dates(
    model_id: str,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    research_only: bool = True,
    mongo_uri: Optional[str] = None,
    db_name: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> List[str]:
    """Distinct prediction dates for one model, optionally research-only.

    The portfolio-research "anchor on model dates" mode needs the schedule,
    not the cross-sections. Research-only is the default because a live model
    can also carry daily shadow rows after its research backfill ends; mixing
    those into the anchor would add off-grid dates.
    """
    if not (adapter_enabled() if enabled is None else enabled):
        return []
    uri = mongo_uri or os.getenv("MONGO_URI", "")
    db = db_name or os.getenv("QUANT_ML_DB", "quant_ml")
    if not uri:
        raise RuntimeError("MONGO_URI required when MODEL_SCORE_ADAPTER_ENABLED=1")
    query: dict = {"model_id": model_id}
    if research_only:
        query["research"] = True
    date_filter = {}
    if start_date:
        date_filter["$gte"] = str(start_date)
    if end_date:
        date_filter["$lte"] = str(end_date)
    if date_filter:
        query["trade_date"] = date_filter
    client = MongoClient(uri)
    try:
        dates = client[db]["model_predictions"].distinct("trade_date", query)
    finally:
        client.close()
    return sorted(str(value) for value in dates if value)


def merge_model_score_into_dataset(
    dataset: pd.DataFrame,
    model_id: str,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    column: str = "model_score",
    ffill_limit: int = 20,
    enabled: Optional[bool] = None,
) -> pd.DataFrame:
    """Merge model scores onto a research dataset keyed by ``score_date``/``symbol``.

    The model is retrained every holding period, so its score exists only on
    the rebalance dates. Carry it forward for at most ``ffill_limit`` sessions
    so a daily dataset sees the score that was live during each holding period,
    without letting a stale score survive past the next retrain.
    """
    if "score_date" not in dataset.columns or "symbol" not in dataset.columns:
        raise ValueError("dataset must include score_date and symbol")
    scores = load_model_scores_range(
        model_id,
        start_date=start_date,
        end_date=end_date,
        enabled=enabled,
    )
    out = dataset.copy()
    if scores.empty:
        out[column] = float("nan")
        return out
    scores = scores.rename(
        columns={"ts_code": "symbol", "trade_date": "score_date", "score": column}
    )
    scores = scores.drop_duplicates(subset=["score_date", "symbol"], keep="last")
    out = out.merge(scores, on=["score_date", "symbol"], how="left")
    out = out.sort_values(["symbol", "score_date"], kind="mergesort").reset_index(drop=True)
    out[column] = out.groupby("symbol", sort=False)[column].ffill(limit=int(ffill_limit))
    out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def merge_model_score_column(panel: pd.DataFrame, model_id: str, column: str = "model_score") -> pd.DataFrame:
    """Left-merge model scores onto a panel with trade_date + ts_code."""
    if not adapter_enabled():
        return panel
    if "trade_date" not in panel.columns or "ts_code" not in panel.columns:
        raise ValueError("panel must include trade_date and ts_code")
    dates = panel["trade_date"].unique().tolist()
    chunks = []
    for d in dates:
        scores = load_model_scores(model_id, trade_date=str(d))
        if scores.empty:
            continue
        scores = scores.assign(trade_date=str(d))
        chunks.append(scores[["trade_date", "ts_code", "model_score"]])
    if not chunks:
        out = panel.copy()
        out[column] = pd.NA
        return out
    merged = panel.merge(pd.concat(chunks, ignore_index=True), on=["trade_date", "ts_code"], how="left")
    if column != "model_score":
        merged = merged.rename(columns={"model_score": column})
    return merged
