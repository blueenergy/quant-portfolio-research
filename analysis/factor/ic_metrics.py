"""Information-coefficient and quantile metrics for factor evaluation.

Generic scoring diagnostics shared by the alpha factor backtest and the
growth baseline research, so they live in the factor layer instead of
inside either caller.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


def return_col(horizon: int) -> str:
    return f"forward_return_{horizon}d"


def per_date_rank_ic(
    dataset: pd.DataFrame,
    score_col: str,
    ret_col: str,
    min_names: int,
) -> pd.DataFrame:
    """Cross-sectional Spearman correlation between score and forward return per date."""
    rows: List[Dict[str, Any]] = []
    for score_date, group in dataset.groupby("score_date"):
        valid = group[[score_col, ret_col]].dropna()
        if len(valid) < min_names or valid[score_col].nunique() < 2 or valid[ret_col].nunique() < 2:
            continue
        # Spearman is Pearson on ranks. pandas' method="spearman" imports
        # scipy.stats, which requirements-docker.txt leaves out, so the worker
        # image can only take this route.
        ic = float(valid[score_col].rank().corr(valid[ret_col].rank()))
        if math.isnan(ic):
            continue
        rows.append({"score_date": str(score_date), "rank_ic": ic, "count": int(len(valid))})
    return pd.DataFrame(rows)


def summarize_ic(per_date_ic: pd.DataFrame) -> Dict[str, Any]:
    if per_date_ic.empty:
        return {
            "observations": 0,
            "ic_mean": None,
            "ic_std": None,
            "ic_ir": None,
            "t_stat": None,
            "positive_ratio": None,
            "avg_names": None,
        }
    ic = per_date_ic["rank_ic"]
    n = int(len(ic))
    mean = float(ic.mean())
    std = float(ic.std(ddof=1)) if n > 1 else None
    ir = float(mean / std) if std not in (None, 0.0) else None
    t_stat = float(ir * math.sqrt(n)) if ir is not None else None
    return {
        "observations": n,
        "ic_mean": mean,
        "ic_std": std,
        "ic_ir": ir,
        "t_stat": t_stat,
        "positive_ratio": float((ic > 0).mean()),
        "avg_names": float(per_date_ic["count"].mean()),
    }


def per_year_ic(per_date_ic: pd.DataFrame) -> List[Dict[str, Any]]:
    if per_date_ic.empty:
        return []
    df = per_date_ic.copy()
    df["year"] = df["score_date"].str[:4]
    out: List[Dict[str, Any]] = []
    for year, group in df.groupby("year"):
        summary = summarize_ic(group)
        summary["year"] = year
        out.append(summary)
    return out


def quantile_monotonicity(
    dataset: pd.DataFrame,
    score_col: str,
    ret_col: str,
    quantiles: int,
    min_names: int,
) -> Dict[str, Any]:
    """Average per-quantile forward return across dates, then test monotonicity."""
    per_q_returns: Dict[int, List[float]] = {q: [] for q in range(1, quantiles + 1)}
    for _, group in dataset.groupby("score_date"):
        valid = group[[score_col, ret_col]].dropna().copy()
        if len(valid) < max(quantiles, min_names) or valid[score_col].nunique() < 2:
            continue
        try:
            valid["q"] = pd.qcut(valid[score_col].rank(method="first"), quantiles, labels=False) + 1
        except ValueError:
            continue
        means = valid.groupby("q")[ret_col].mean()
        for q, val in means.items():
            per_q_returns[int(q)].append(float(val))

    avg_by_q = {q: (float(np.mean(v)) if v else None) for q, v in per_q_returns.items()}
    available = [avg_by_q[q] for q in sorted(avg_by_q) if avg_by_q[q] is not None]
    monotonic_increasing = all(x < y for x, y in zip(available, available[1:])) if len(available) >= 2 else None
    top = avg_by_q.get(quantiles)
    bottom = avg_by_q.get(1)
    spread = (top - bottom) if (top is not None and bottom is not None) else None
    # Rank correlation between quantile index and mean return (1 == perfectly monotone).
    rank_corr = None
    if len(available) >= 2:
        qs = pd.Series(range(1, len(available) + 1))
        rank_corr = float(qs.corr(pd.Series(available).rank()))
    return {
        "mean_return_by_quantile": {str(q): avg_by_q[q] for q in sorted(avg_by_q)},
        "top_minus_bottom_spread": spread,
        "monotonic_increasing": monotonic_increasing,
        "quantile_rank_corr": rank_corr,
    }


# Two-sided alpha=5% + 80% power: z_alpha/2 + z_beta.
Z_80 = 1.96 + 0.84


def nonoverlap_summary(per_date_ic: pd.DataFrame, horizon: int) -> Dict[str, Any]:
    """IC summary on every ``horizon``-th date, where forward returns stop overlapping.

    ``summarize_ic`` treats each daily IC as an independent draw. A 20-day
    forward return spans 20 signal dates, so neighbouring ICs share 19/20 of
    their return window and ``IR * sqrt(n)`` overstates the evidence. Sampling
    every ``horizon``-th date removes that duplication at the cost of power;
    ``mde80`` is reported so the cost is visible instead of silent.

    The overlapping summary is still the screening statistic (more draws, stable
    ranking). This one is the decision statistic.
    """
    step = max(1, int(horizon))
    if per_date_ic.empty:
        return {
            "step": step,
            "observations": 0,
            "ic_mean": None,
            "ic_std": None,
            "ic_ir": None,
            "t_stat": None,
            "mde80": None,
            "positive_ratio": None,
        }
    ordered = per_date_ic.sort_values("score_date")
    summary = summarize_ic(ordered.iloc[::step])
    n = summary["observations"]
    std = summary["ic_std"]
    se = float(std / math.sqrt(n)) if std is not None and n > 1 else None
    return {
        "step": step,
        "observations": n,
        "ic_mean": summary["ic_mean"],
        "ic_std": std,
        "ic_ir": summary["ic_ir"],
        "t_stat": summary["t_stat"],
        "mde80": float(Z_80 * se) if se is not None else None,
        "positive_ratio": summary["positive_ratio"],
    }


def signal_quality(
    dataset: pd.DataFrame,
    score_columns: Sequence[str],
    horizons: Sequence[int],
    *,
    min_names: int = 20,
) -> Dict[str, Any]:
    """Cross-sectional RankIC / ICIR / non-overlap t / MDE80 per score and horizon.

    This is a *score-level* diagnostic: the same score column yields the same
    values across every top_n / rebalance / construction combo, so callers should
    display it once per score rather than repeating it on each result row.
    """
    out: Dict[str, Any] = {}
    if dataset is None or dataset.empty:
        return out
    for score_column in score_columns:
        column = str(score_column or "").strip()
        if not column or column not in dataset.columns:
            continue
        per_horizon: Dict[str, Any] = {}
        for horizon in horizons:
            ret_col = return_col(int(horizon))
            if ret_col not in dataset.columns:
                continue
            per_ic = per_date_rank_ic(dataset, column, ret_col, min_names)
            if per_ic.empty:
                continue
            per_ic = per_ic.sort_values("score_date")
            summary = summarize_ic(per_ic)
            nonoverlap = nonoverlap_summary(per_ic, horizon)
            summary.update(
                {
                    "nonoverlap_observations": nonoverlap["observations"],
                    "t_nonoverlap": nonoverlap["t_stat"],
                    "mde80": nonoverlap["mde80"],
                    "by_year": [
                        {**summarize_ic(group), "year": str(year)}
                        for year, group in per_ic.groupby(
                            per_ic["score_date"].astype(str).str[:4],
                            sort=True,
                        )
                    ],
                }
            )
            per_horizon[f"h{int(horizon)}"] = summary
        if per_horizon:
            out[column] = per_horizon
    return out
