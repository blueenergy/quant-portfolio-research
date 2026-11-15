"""SW L1 industry regime labels, one per (symbol, score_date) dataset row.

The portfolio-level overlay in ``shared/regime_label`` judges a single broad
index and moves the whole book to cash. This module supplies the second,
orthogonal layer: the bull/bear/uncertain label of *each name's own* SW L1
industry index, which the backtester turns into a per-name weight factor.

Labels come from the same K-indicator rule as the portfolio overlay, so an
industry index and the benchmark are judged by identical machinery. The row's
label is always the last label visible on or before its ``score_date`` -- no
look-ahead.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from analysis.core.benchmark_prices import load_index_benchmark_prices
from shared.industry_universe import (
    SW_L1_CODE_COL,
    SW_L1_REGIME_COL,
    SW_SRC_DEFAULT,
    sw_l1_catalog,
)
from shared.regime_label import label_index_regime, padded_index_start

log = logging.getLogger(__name__)

def _as_object_str(series: pd.Series) -> pd.Series:
    """Pin a column to plain ``object`` dtype regardless of pandas' inference."""
    return pd.Series(series.astype(str).to_numpy(), dtype=object, index=series.index)


_TRADE_DATE_COL = "trade_date"
# ``merge_asof`` needs a numeric key: pandas 3 rejects StringDtype keys outright.
# YYYYMMDD parses to an int that sorts chronologically, so it is a safe proxy.
_TRADE_DATE_KEY = "trade_date_key"


def sw_l1_regime_codes(db_manager, *, src: str = SW_SRC_DEFAULT) -> List[str]:
    """All SW L1 industry index codes, sorted."""
    return sorted({str(entry["code"]) for entry in sw_l1_catalog(db_manager, src=src) if entry.get("code")})


def build_sw_l1_regime_frame(
    db_manager,
    *,
    codes: Sequence[str],
    start_date: str,
    regime_rule: Optional[str] = None,
) -> pd.DataFrame:
    """Label each requested industry index on its own trading days.

    Returns a long frame of ``sw_l1_code`` / ``trade_date_key`` / ``sw_l1_regime``.
    Industry indices with no price history are skipped with a warning rather
    than raising: one missing index must not sink a whole research run.
    """
    start = padded_index_start(start_date)
    frames: List[pd.DataFrame] = []
    for code in codes:
        code_text = str(code or "").strip()
        if not code_text:
            continue
        rows = load_index_benchmark_prices(db_manager, code_text, start)
        if not rows:
            log.warning("no price history for SW L1 index %s; leaving its names unscaled", code_text)
            continue
        labels = label_index_regime(rows, regime_rule=regime_rule)
        if labels.empty:
            continue
        date_keys = pd.to_numeric(labels[_TRADE_DATE_COL].astype(str), errors="coerce")
        keep = date_keys.notna()
        if not keep.any():
            continue
        frames.append(
            pd.DataFrame(
                {
                    SW_L1_CODE_COL: pd.Series([code_text] * int(keep.sum()), dtype=object),
                    _TRADE_DATE_KEY: date_keys[keep].astype("int64").to_numpy(),
                    SW_L1_REGIME_COL: labels.loc[keep, "label"].astype(str).to_numpy(),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=[SW_L1_CODE_COL, _TRADE_DATE_KEY, SW_L1_REGIME_COL])
    return pd.concat(frames, ignore_index=True)


def attach_sw_l1_regime(
    dataset: pd.DataFrame,
    db_manager,
    *,
    regime_rule: Optional[str] = None,
) -> pd.DataFrame:
    """Attach ``sw_l1_regime`` as of each row's ``score_date``.

    Requires ``sw_l1_code`` (from :func:`attach_sw_l1`). Rows whose industry is
    unmapped, or whose index has no history, get an empty label -- the
    backtester treats an empty label as "no intervention" so that an unmapped
    name is never silently treated as non-bull.
    """
    if dataset is None or dataset.empty:
        return dataset
    if SW_L1_CODE_COL not in dataset.columns or "score_date" not in dataset.columns:
        log.warning("attach_sw_l1_regime needs sw_l1_code and score_date columns; skipping")
        return dataset

    frame = dataset.copy()
    frame[SW_L1_REGIME_COL] = ""
    codes = sorted({str(code).strip() for code in frame[SW_L1_CODE_COL].dropna() if str(code).strip()})
    if not codes:
        return frame

    score_raw = frame["score_date"].astype(str)
    score_keys = pd.to_numeric(score_raw, errors="coerce")
    placeable = score_keys.notna()
    if not placeable.any():
        return frame

    labels = build_sw_l1_regime_frame(
        db_manager,
        codes=codes,
        start_date=str(score_raw.min()),
        regime_rule=regime_rule,
    )
    if labels.empty:
        return frame

    # merge_asof returns a fresh RangeIndex, so the dataset's own index travels
    # as a column and the labels are written back through it. Both join keys are
    # pinned to plain object dtype: pandas 3 otherwise infers StringDtype on one
    # side and object on the other, which merge_asof refuses to match.
    labels = labels.copy()
    labels[SW_L1_CODE_COL] = _as_object_str(labels[SW_L1_CODE_COL])
    left = pd.DataFrame(
        {
            "_row_id": frame.index.to_numpy(),
            "_score_key": score_keys.to_numpy(),
            SW_L1_CODE_COL: _as_object_str(frame[SW_L1_CODE_COL]),
        }
    ).loc[placeable.to_numpy()]
    # pandas 3 requires the right frame sorted by the on-key globally, not per
    # group; a global ascending date sort is also ascending within every code.
    merged = pd.merge_asof(
        left.sort_values("_score_key"),
        labels.sort_values(_TRADE_DATE_KEY, kind="stable"),
        left_on="_score_key",
        right_on=_TRADE_DATE_KEY,
        by=SW_L1_CODE_COL,
        direction="backward",
    )
    frame.loc[merged["_row_id"], SW_L1_REGIME_COL] = (
        merged[SW_L1_REGIME_COL].fillna("").astype(str).to_numpy()
    )
    return frame


def sw_l1_regime_coverage(dataset: pd.DataFrame) -> Dict[str, Any]:
    """Share of rows carrying a usable industry regime label."""
    if dataset is None or dataset.empty or SW_L1_REGIME_COL not in dataset.columns:
        return {"rows": 0, "labelled_rows": 0, "labelled_share": None, "bull_share": None}
    labelled = dataset[SW_L1_REGIME_COL].fillna("").astype(str).str.strip() != ""
    rows = int(len(dataset))
    labelled_rows = int(labelled.sum())
    bull_rows = int((dataset.loc[labelled, SW_L1_REGIME_COL].astype(str).str.lower() == "bull").sum())
    return {
        "rows": rows,
        "labelled_rows": labelled_rows,
        "labelled_share": (labelled_rows / rows) if rows else None,
        "bull_share": (bull_rows / labelled_rows) if labelled_rows else None,
    }
