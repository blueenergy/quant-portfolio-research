"""Load model scores from ``quant_ml.model_predictions`` and merge them into panels.

This lives in ``analysis.core`` because the portfolio research worker (an
``entry``/``tools`` layer module) consumes it. The draft ``analysis.research``
layer may import core, but production code must not import draft code.

The adapter stays opt-in by default: ``MODEL_SCORE_ADAPTER_ENABLED=1`` is
required, and production scoring never writes ``stock_scores``.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import pandas as pd
from pymongo import MongoClient

log = logging.getLogger(__name__)


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
    prediction_kind: Optional[str] = None,
    prediction_source: Optional[str] = None,
    mongo_uri: Optional[str] = None,
    db_name: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> pd.DataFrame:
    """Bulk load one model's scores over a date range.

    ``prediction_kind`` separates the two streams that can share a model_id:
    ``research`` (point-in-time backfill, ``research=true``) and ``shadow``
    (live artifact rows, ``shadow=true``). ``None`` keeps the legacy behaviour
    and merges every row for the model_id; new research jobs pass ``research``
    explicitly so the end date cannot silently pull in shadow scores.

    A model_id may also carry several research sources. Disjoint sources are
    treated as segments of one series (the 2017-2024 backfill plus its 2026
    extension); overlapping sources are ambiguous and refused unless
    ``prediction_source`` names one. Silently picking ``keep="last"`` after
    merging two protocols would make the result depend on read order.
    """
    if not (adapter_enabled() if enabled is None else enabled):
        return pd.DataFrame(columns=["ts_code", "score", "trade_date"])
    uri = mongo_uri or os.getenv("MONGO_URI", "")
    db = db_name or os.getenv("QUANT_ML_DB", "quant_ml")
    if not uri:
        raise RuntimeError("MONGO_URI required when MODEL_SCORE_ADAPTER_ENABLED=1")
    query = {"model_id": model_id}
    if prediction_kind == "research":
        # Legacy backfills predate the research/shadow flags; treat unflagged
        # rows as research, but never pull shadow rows into a research job.
        query["$or"] = [
            {"research": True},
            {"research": {"$ne": True}, "shadow": {"$ne": True}},
        ]
    elif prediction_kind == "shadow":
        query["shadow"] = True
    elif prediction_kind:
        raise ValueError(
            f"prediction_kind must be 'research', 'shadow' or None, got {prediction_kind!r}"
        )
    if prediction_source:
        query["source"] = prediction_source
    date_filter = {}
    if start_date:
        date_filter["$gte"] = str(start_date)
    if end_date:
        date_filter["$lte"] = str(end_date)
    if date_filter:
        query["trade_date"] = date_filter
    client = MongoClient(uri)
    try:
        predictions = client[db]["model_predictions"]
        if not prediction_source:
            _reject_overlapping_sources(
                predictions,
                query,
                model_id=model_id,
                prediction_kind=prediction_kind,
            )
        rows = list(
            predictions.find(
                query, {"_id": 0, "ts_code": 1, "score": 1, "trade_date": 1}
            )
        )
    finally:
        client.close()
    if not rows:
        return pd.DataFrame(columns=["ts_code", "score", "trade_date"])
    return pd.DataFrame(rows)


def _reject_overlapping_sources(
    predictions,
    query: dict,
    *,
    model_id: str,
    prediction_kind: Optional[str],
) -> None:
    """Refuse a model_id whose prediction sources overlap in the window.

    Disjoint sources are intentional segments (old backfill + extension) and
    are allowed; overlapping sources are two protocols competing for the same
    (trade_date, ts_code), so the caller must name one with
    ``prediction_source``.
    """
    groups = list(
        predictions.aggregate(
            [
                {"$match": query},
                {
                    "$group": {
                        "_id": "$source",
                        "first": {"$min": "$trade_date"},
                        "last": {"$max": "$trade_date"},
                    }
                },
                {"$sort": {"first": 1}},
            ]
        )
    )
    if len(groups) <= 1:
        return
    for previous, current in zip(groups, groups[1:]):
        if str(current.get("first") or "") <= str(previous.get("last") or ""):
            raise ValueError(
                f"ambiguous prediction sources for model_id={model_id} "
                f"kind={prediction_kind or 'legacy'}: "
                f"{previous.get('_id')!r} ({previous.get('first')}..{previous.get('last')}) "
                f"overlaps {current.get('_id')!r} ({current.get('first')}..{current.get('last')}); "
                "pass prediction_source to select one"
            )
    log.warning(
        "model_id=%s kind=%s has %d disjoint prediction sources: %s",
        model_id,
        prediction_kind or "legacy",
        len(groups),
        ", ".join(
            f"{group.get('_id')!r} ({group.get('first')}..{group.get('last')})"
            for group in groups
        ),
    )


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
    prediction_kind: Optional[str] = None,
    prediction_source: Optional[str] = None,
    column: str = "model_score",
    ffill_limit: int = 20,
    enabled: Optional[bool] = None,
) -> pd.DataFrame:
    """Merge model scores onto a research dataset keyed by ``score_date``/``symbol``.

    The model is retrained every holding period, so its score exists only on
    the rebalance dates. Carry it forward for at most ``ffill_limit`` sessions
    so a daily dataset sees the score that was live during each holding period,
    without letting a stale score survive past the next retrain.

    ``prediction_kind`` selects ``research`` or ``shadow`` rows; ``None`` keeps
    the pre-P0 behaviour (all rows for the model_id) for legacy jobs.
    ``prediction_source`` disambiguates a model_id with overlapping research
    sources; disjoint sources are loaded together as one series.
    """
    if "score_date" not in dataset.columns or "symbol" not in dataset.columns:
        raise ValueError("dataset must include score_date and symbol")
    scores = load_model_scores_range(
        model_id,
        start_date=start_date,
        end_date=end_date,
        prediction_kind=prediction_kind,
        prediction_source=prediction_source,
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

    # ``ffill`` counts rows, so a symbol that leaves the panel and re-enters
    # years later would carry its old score across the missing rows. Measure
    # staleness in panel sessions instead: a score may fill only the next
    # ``ffill_limit`` dates, even when the symbol has no rows in between.
    session_index = {
        trade_date: index
        for index, trade_date in enumerate(sorted(out["score_date"].unique()))
    }
    out["_session_index"] = out["score_date"].map(session_index)
    out["_score_origin"] = out["_session_index"].where(out[column].notna())
    out["_score_origin"] = out.groupby("symbol", sort=False)["_score_origin"].ffill()
    gap = out["_session_index"] - out["_score_origin"]
    out[column] = out.groupby("symbol", sort=False)[column].ffill(limit=int(ffill_limit))
    out.loc[gap > int(ffill_limit), column] = float("nan")
    out = out.drop(columns=["_session_index", "_score_origin"])
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
