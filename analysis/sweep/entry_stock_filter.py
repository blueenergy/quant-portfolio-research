"""Point-in-time stock entry filters (hipos / distrib) for portfolio sweep."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from analysis.core.entry_stock_factors import (
    build_entry_factor_panel,
    exclusion_masks,
    factors_aligned_to_dataset,
    mask_dataset_scores,
)

ENTRY_STOCK_FILTER_OFF = "off"
VALID_ENTRY_STOCK_FILTERS = frozenset(
    {
        ENTRY_STOCK_FILTER_OFF,
        "distrib03",
        "distrib05",
        "distrib08",
        "hipos95",
        "hipos97",
        "hipos97_distrib",
        # Direct style-exposure caps: keep names whose cross-sectional
        # percentile is at or below the threshold on each score_date.
        "vol80",
        "vol70",
        "vol60",
        "cycle80",
        "cycle70",
        "vol70_cycle70",
    }
)

_EXPOSURE_FILTER_SPECS: Dict[str, tuple[tuple[str, float], ...]] = {
    "vol80": (("vol_20d_annual", 0.80),),
    "vol70": (("vol_20d_annual", 0.70),),
    "vol60": (("vol_20d_annual", 0.60),),
    "cycle80": (("cycle_score", 0.80),),
    "cycle70": (("cycle_score", 0.70),),
    "vol70_cycle70": (("vol_20d_annual", 0.70), ("cycle_score", 0.70)),
}


def requires_entry_factor_panel(mode: str) -> bool:
    """True only for the legacy hipos/distrib filters that need the factor panel."""
    return str(mode or "").strip().lower() in _EXCLUDE_COLUMN

_EXCLUDE_COLUMN: Dict[str, str] = {
    "distrib03": "exclude_distrib03",
    "distrib05": "exclude_distrib05",
    "distrib08": "exclude_distrib08",
    "hipos95": "exclude_hipos95",
    "hipos97": "exclude_hipos97",
    "hipos97_distrib": "exclude_hipos97_distrib",
}


def normalize_entry_stock_filters(
    values: Optional[Sequence[str]] = None,
) -> List[str]:
    if values is None:
        return [ENTRY_STOCK_FILTER_OFF]
    out: List[str] = []
    for raw in values:
        mode = str(raw).strip().lower()
        if not mode:
            continue
        if mode not in VALID_ENTRY_STOCK_FILTERS:
            raise ValueError(f"Unsupported entry_stock_filter: {raw!r}")
        if mode not in out:
            out.append(mode)
    return out or [ENTRY_STOCK_FILTER_OFF]


def _masks_with_thresholds(aligned: pd.DataFrame) -> pd.DataFrame:
    base = exclusion_masks(aligned)
    distrib = pd.to_numeric(base["distrib"], errors="coerce")
    hipos = pd.to_numeric(base["hipos120"], errors="coerce")
    base["exclude_distrib03"] = distrib >= 0.03
    base["exclude_distrib08"] = distrib >= 0.08
    base["exclude_hipos95"] = hipos >= 0.95
    base["exclude_hipos97_distrib"] = base["exclude_hipos97"] | base["exclude_distrib05"]
    return base


def _apply_exposure_percentile_filter(
    dataset: pd.DataFrame,
    score_column: str,
    specs: Sequence[tuple[str, float]],
) -> pd.DataFrame:
    """Null scores for rows above a cross-sectional exposure percentile cap."""
    missing = [column for column, _ in specs if column not in dataset.columns]
    if missing:
        raise ValueError(
            "entry filter exposure column(s) missing from dataset: "
            + ", ".join(missing)
        )
    work = dataset[["symbol", "score_date", *[column for column, _ in specs]]].copy()
    exclude = pd.Series(False, index=work.index)
    for column, threshold in specs:
        pct = work.groupby("score_date")[column].rank(pct=True)
        exclude |= pct.gt(float(threshold)) | pct.isna()
    masks = work[["symbol", "score_date"]].copy()
    masks["exclude_exposure"] = exclude.to_numpy()
    return mask_dataset_scores(dataset, score_column, masks, "exclude_exposure")


def apply_entry_stock_filter(
    db: Any,
    dataset: pd.DataFrame,
    score_column: str,
    mode: str,
    start_date: str,
    end_date: str,
    *,
    factor_panel: Optional[pd.DataFrame] = None,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Return dataset copy with ``score_column`` nulled for excluded (symbol, score_date) rows."""
    filter_mode = str(mode or ENTRY_STOCK_FILTER_OFF).strip().lower()
    if filter_mode == ENTRY_STOCK_FILTER_OFF:
        return dataset
    exposure_specs = _EXPOSURE_FILTER_SPECS.get(filter_mode)
    if exposure_specs is not None:
        return _apply_exposure_percentile_filter(dataset, score_column, exposure_specs)
    exclude_col = _EXCLUDE_COLUMN.get(filter_mode)
    if exclude_col is None:
        raise ValueError(f"Unsupported entry_stock_filter: {mode!r}")

    panel = factor_panel
    if panel is None:
        symbols = dataset["symbol"].astype(str).unique().tolist()
        panel = build_entry_factor_panel(
            db, symbols, start_date, end_date, max_workers=max_workers
        )
    aligned = factors_aligned_to_dataset(dataset, panel)
    masks = _masks_with_thresholds(aligned)
    if exclude_col not in masks.columns:
        raise ValueError(f"Missing mask column {exclude_col}")
    return mask_dataset_scores(dataset, score_column, masks, exclude_col)


def dataset_cache_key(trailing_key: str, entry_filter: Optional[str]) -> str:
    mode = str(entry_filter or ENTRY_STOCK_FILTER_OFF).strip().lower()
    if mode in ("", ENTRY_STOCK_FILTER_OFF):
        return trailing_key
    return f"{trailing_key}__ef_{mode}"


def mask_score_column_stats(dataset: pd.DataFrame, score_column: str) -> Dict[str, Any]:
    """Diagnostics: how many score cells are NaN after masking."""
    if score_column not in dataset.columns:
        return {
            "score_column": score_column,
            "rows": len(dataset),
            "nan_count": None,
            "nan_frac": None,
            "error": "missing_column",
        }
    series = pd.to_numeric(dataset[score_column], errors="coerce")
    nan_count = int(series.isna().sum())
    total = int(len(series))
    return {
        "score_column": score_column,
        "rows": total,
        "nan_count": nan_count,
        "nan_frac": float(nan_count / total) if total else 0.0,
        "finite_count": total - nan_count,
    }


def top_n_symbols_by_score_date(
    dataset: pd.DataFrame,
    score_column: str,
    score_dates: Sequence[str],
    top_n: int,
) -> Dict[str, List[str]]:
    """Fixed top-N by score at each score_date (ignores industry cap; diagnostic only)."""
    out: Dict[str, List[str]] = {}
    work = dataset[["symbol", "score_date", score_column]].copy()
    work["_score"] = pd.to_numeric(work[score_column], errors="coerce")
    for raw_date in score_dates:
        score_date = str(raw_date)
        sub = work.loc[work["score_date"].astype(str) == score_date].dropna(subset=["_score"])
        if sub.empty:
            out[score_date] = []
            continue
        picks = (
            sub.sort_values("_score", ascending=False)
            .head(int(top_n))["symbol"]
            .astype(str)
            .tolist()
        )
        out[score_date] = picks
    return out


def compare_top_n_across_filters(
    datasets_by_filter: Dict[str, pd.DataFrame],
    score_column: str,
    score_dates: Sequence[str],
    top_n: int,
) -> Dict[str, Any]:
    """Compare naive top-N lists between filter modes."""
    keys = list(datasets_by_filter.keys())
    if not keys:
        return {"dates_compared": 0, "dates_with_diff": 0, "by_date": {}}
    baseline_key = keys[0]
    baseline = top_n_symbols_by_score_date(
        datasets_by_filter[baseline_key], score_column, score_dates, top_n
    )
    by_date: Dict[str, Any] = {}
    dates_with_diff = 0
    for score_date in score_dates:
        sd = str(score_date)
        base_syms = baseline.get(sd, [])
        entry: Dict[str, Any] = {"baseline": base_syms, "others": {}}
        date_diff = False
        for key in keys[1:]:
            other = top_n_symbols_by_score_date(
                datasets_by_filter[key], score_column, [sd], top_n
            ).get(sd, [])
            entry["others"][key] = other
            if set(base_syms) != set(other):
                date_diff = True
        if date_diff:
            dates_with_diff += 1
            entry["diff"] = True
        by_date[sd] = entry
    return {
        "baseline_filter": baseline_key,
        "dates_compared": len(score_dates),
        "dates_with_diff": dates_with_diff,
        "by_date": by_date,
    }


def build_factor_panel_for_dataset(
    db: Any,
    dataset: pd.DataFrame,
    start_date: str,
    end_date: str,
    *,
    max_workers: int = 8,
) -> pd.DataFrame:
    symbols = dataset["symbol"].astype(str).unique().tolist()
    return build_entry_factor_panel(db, symbols, start_date, end_date, max_workers=max_workers)
