"""Model score merge keeps the ffill limit in trading sessions.

``ffill(limit=N)`` counts rows in the panel. A symbol that leaves the
universe and re-enters later has no rows in between, so a row-based ffill
would carry a years-old score across the gap. These tests pin the session
gap semantics instead.
"""

from __future__ import annotations

import pandas as pd

from analysis.core import model_scores


def _dates(count: int) -> list[str]:
    return [f"202001{day:02d}" for day in range(1, count + 1)]


def test_ffill_limit_counts_sessions_not_panel_rows(monkeypatch):
    dates = _dates(31)
    scores = pd.DataFrame(
        {
            "ts_code": ["A", "B"],
            "trade_date": [dates[0], dates[0]],
            "score": [1.0, 2.0],
        }
    )
    monkeypatch.setattr(
        model_scores,
        "load_model_scores_range",
        lambda *args, **kwargs: scores,
    )
    dataset = pd.DataFrame(
        {
            "score_date": dates + [dates[0], dates[-1]],
            "symbol": ["A"] * len(dates) + ["B", "B"],
        }
    )

    merged = model_scores.merge_model_score_into_dataset(
        dataset,
        "m1",
        column="model_score",
        ffill_limit=20,
        enabled=True,
    )

    a = merged[merged["symbol"] == "A"].reset_index(drop=True)
    b = merged[merged["symbol"] == "B"].reset_index(drop=True)
    # Origin date plus 20 sessions; the 22nd row is stale.
    assert a.loc[:20, "model_score"].notna().all()
    assert a.loc[21:, "model_score"].isna().all()
    # B has no rows between its first and last date; the 31st date is 30
    # sessions after the score and must not inherit it.
    assert b.loc[0, "model_score"] == 2.0
    assert pd.isna(b.loc[1, "model_score"])


def test_new_score_resets_the_session_gap(monkeypatch):
    dates = _dates(31)
    scores = pd.DataFrame(
        {
            "ts_code": ["A", "A"],
            "trade_date": [dates[0], dates[25]],
            "score": [1.0, 3.0],
        }
    )
    monkeypatch.setattr(
        model_scores,
        "load_model_scores_range",
        lambda *args, **kwargs: scores,
    )
    dataset = pd.DataFrame(
        {"score_date": dates, "symbol": ["A"] * len(dates)}
    )

    merged = model_scores.merge_model_score_into_dataset(
        dataset,
        "m1",
        column="model_score",
        ffill_limit=20,
        enabled=True,
    )

    # The first score expires after 20 sessions, so 20200122..20200125 are
    # NaN; the fresh score on 20200126 starts a new carry window.
    assert merged.loc[20, "model_score"] == 1.0
    assert merged.loc[21:24, "model_score"].isna().all()
    assert merged.loc[25, "model_score"] == 3.0
    assert merged.loc[26, "model_score"] == 3.0
