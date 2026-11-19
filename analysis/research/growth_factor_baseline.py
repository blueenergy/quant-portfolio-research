#!/usr/bin/env python3
"""Phase-0 baseline evaluation for the growth factor.

This module does NOT change scoring logic. It reuses ``ScoreBacktester`` to build
the (score, forward-return) dataset, then reports factor diagnostics that anchor
all later growth-scoring improvements:

- Rank IC mean / std / IC_IR / t-stat / positive ratio per horizon.
- Per-year Rank IC breakdown (regime stability).
- Industry-neutral Rank IC (how much signal is stock selection vs industry bet).
- Quantile monotonicity (is top minus bottom spread monotone?).

Run:
    python analysis/growth_factor_baseline.py --start-date 20220101 \
        --end-date 20241231 --horizons 5 10 20 --output baseline_growth.json
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

import pandas as pd

from analysis.core.score_backtester import BacktestConfig, ScoreBacktester
from analysis.factor.ic_metrics import (
    per_date_rank_ic,
    per_year_ic,
    quantile_monotonicity,
    return_col,
    summarize_ic,
)

# Sub-factor columns we eventually want to track. They are not yet persisted in
# stock_scores, so they are silently skipped if absent. Phase 2 will add them.
GROWTH_FACTOR_COLUMNS = (
    "growth_score",
    "composite_growth_cycle_score",
)








def _add_industry_neutral(dataset: pd.DataFrame, score_col: str) -> str:
    """Add an industry-demeaned score column (per score_date x industry) and return its name."""
    neutral_col = f"{score_col}__ind_neutral"
    if "industry" not in dataset.columns:
        dataset[neutral_col] = dataset[score_col]
        return neutral_col

    industry = dataset["industry"].fillna("__UNKNOWN__").replace("", "__UNKNOWN__")
    group_mean = dataset.groupby(["score_date", industry])[score_col].transform("mean")
    dataset[neutral_col] = dataset[score_col] - group_mean
    return neutral_col



def evaluate(dataset: pd.DataFrame, config: BacktestConfig) -> Dict[str, Any]:
    if dataset.empty:
        return {"error": "empty dataset: no overlapping score records and prices found"}

    available_cols = [c for c in GROWTH_FACTOR_COLUMNS if c in dataset.columns]
    report: Dict[str, Any] = {
        "rows": int(len(dataset)),
        "score_date_range": [dataset["score_date"].min(), dataset["score_date"].max()],
        "distinct_dates": int(dataset["score_date"].nunique()),
        "evaluated_columns": available_cols,
        "factors": {},
    }

    for score_col in available_cols:
        neutral_col = _add_industry_neutral(dataset, score_col)
        per_horizon: Dict[str, Any] = {}
        for horizon in config.horizons:
            ret_col = return_col(horizon)
            if ret_col not in dataset.columns:
                continue
            raw_ic = per_date_rank_ic(dataset, score_col, ret_col, config.min_names_per_date)
            neutral_ic = per_date_rank_ic(dataset, neutral_col, ret_col, config.min_names_per_date)
            per_horizon[str(horizon)] = {
                "raw": summarize_ic(raw_ic),
                "industry_neutral": summarize_ic(neutral_ic),
                "per_year": per_year_ic(raw_ic),
                "quantiles": quantile_monotonicity(
                    dataset, score_col, ret_col, config.quantiles, config.min_names_per_date
                ),
            }
        report["factors"][score_col] = per_horizon

    return report


def _print_human_summary(report: Dict[str, Any]) -> None:
    if report.get("error"):
        print(f"[baseline] {report['error']}")
        return
    print(f"[baseline] rows={report['rows']} dates={report['distinct_dates']} "
          f"range={report['score_date_range']}")
    for score_col, horizons in report["factors"].items():
        print(f"\n=== {score_col} ===")
        header = f"{'h':>4} {'IC':>8} {'IC_IR':>8} {'t':>7} {'pos%':>6} {'IC(ind)':>9} {'spread':>9} {'mono':>5}"
        print(header)
        for h, blk in horizons.items():
            raw = blk["raw"]
            neu = blk["industry_neutral"]
            q = blk["quantiles"]

            def _f(v, p=4):
                return f"{v:.{p}f}" if isinstance(v, (int, float)) else "  n/a"

            print(
                f"{h:>4} {_f(raw['ic_mean']):>8} {_f(raw['ic_ir']):>8} {_f(raw['t_stat'],2):>7} "
                f"{_f(raw['positive_ratio'],2):>6} {_f(neu['ic_mean']):>9} "
                f"{_f(q['top_minus_bottom_spread']):>9} {str(q['monotonic_increasing']):>5}"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-0 baseline evaluation for the growth factor.")
    parser.add_argument("--start-date", help="Inclusive score_date lower bound, YYYYMMDD")
    parser.add_argument("--end-date", help="Inclusive score_date upper bound, YYYYMMDD")
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--min-names", type=int, default=20, help="Min names per cross-section for IC")
    parser.add_argument("--index-codes", nargs="+", default=[])
    parser.add_argument("--universe-index", "--universe-index-codes", nargs="+", default=[])
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument("--enable-tushare", action="store_true")
    return parser.parse_args()


def main() -> int:
    from shared.data_access import open_research_data_access_for_cli

    args = _parse_args()
    config = BacktestConfig(
        horizons=tuple(args.horizons),
        quantiles=args.quantiles,
        start_date=args.start_date,
        end_date=args.end_date,
        index_codes=tuple(args.index_codes),
        universe_index_codes=tuple(args.universe_index),
        min_names_per_date=args.min_names,
    )

    backtester = ScoreBacktester(open_research_data_access_for_cli(enable_tushare=args.enable_tushare), config)
    dataset = backtester.build_dataset()
    report = evaluate(dataset, config)

    _print_human_summary(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"\n[baseline] written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
