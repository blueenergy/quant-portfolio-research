#!/usr/bin/env python3
"""Phase-0 read-only diagnosis: score quality varies across Shenwan L1 industries.

Builds the standard score/return panel, attaches SW2021 L1 codes, and compares
per-industry Rank IC / quantile monotonicity / Top-K excess vs:

- the industry Shenwan index (``sw_daily``)
- the broad index benchmark (e.g. ``000852.SH`` for csi1000)

Verdict rules (fixed before reading results):

- ``proceed``: at least 5 L1 buckets on the primary horizon show materially
  different within-industry IC or Top-K excess vs the industry index compared
  with the full-universe anchor, and average daily names per bucket >= min_names.
- ``stop``: differences mostly vanish vs the industry index (beta only), or
  most L1 buckets lack enough names for stable IC.

Membership is point-in-time: each row is labelled with the industry it belonged
to on its own ``score_date`` (``in_date <= score_date <= out_date``), resolved
from the ``sw_index_member`` stint table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from analysis.core.benchmark_prices import load_index_benchmark_prices
from shared.industry_universe import (
    SW_L1_CODE_COL,
    SW_L1_NAME_COL,
    attach_sw_l1,
    sw_l1_catalog,
    sw_l1_coverage_summary,
)
from analysis.core.score_backtester import BacktestConfig, ScoreBacktester
from analysis.factor.ic_metrics import (
    per_date_rank_ic,
    quantile_monotonicity,
    return_col,
    summarize_ic,
)
from analysis.sweep.score_columns import add_growth_cycle_weight_columns, parse_growth_cycle_weights


def _benchmark_forward_return(
    price_rows: Sequence[Dict[str, Any]],
    score_date: str,
    horizon: int,
) -> Optional[float]:
    if not price_rows:
        return None
    future = [row for row in price_rows if row.get("trade_date") and str(row["trade_date"]) >= score_date]
    if len(future) <= horizon:
        return None
    try:
        base_close = float(future[0].get("close"))
        future_close = float(future[horizon].get("close"))
    except (TypeError, ValueError):
        return None
    if base_close <= 0:
        return None
    return future_close / base_close - 1.0


def _clean_benchmark_prices(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        trade_date = str(row.get("trade_date") or "").replace("-", "")[:8]
        if len(trade_date) != 8 or trade_date in seen:
            continue
        try:
            close = float(row.get("close"))
        except (TypeError, ValueError):
            continue
        if close <= 0:
            continue
        seen.add(trade_date)
        cleaned.append({"trade_date": trade_date, "close": close})
    return sorted(cleaned, key=lambda item: item["trade_date"])


def _evaluate_slice(
    dataset: pd.DataFrame,
    *,
    score_column: str,
    horizon: int,
    top_k_values: Sequence[int],
    min_names: int,
    broad_benchmark_prices: Sequence[Dict[str, Any]],
    industry_benchmark_prices: Optional[Sequence[Dict[str, Any]]] = None,
    slice_label: str = "all",
    l1_code: Optional[str] = None,
    l1_name: Optional[str] = None,
) -> Dict[str, Any]:
    ret_col = return_col(horizon)
    if dataset.empty or score_column not in dataset.columns or ret_col not in dataset.columns:
        return {
            "slice": slice_label,
            "l1_code": l1_code,
            "l1_name": l1_name,
            "rows": 0,
            "skipped": True,
            "reason": "empty_or_missing_columns",
        }

    valid = dataset.dropna(subset=["symbol", score_column, ret_col]).copy()
    if valid.empty:
        return {
            "slice": slice_label,
            "l1_code": l1_code,
            "l1_name": l1_name,
            "rows": 0,
            "skipped": True,
            "reason": "no_valid_rows",
        }

    avg_names = float(valid.groupby("score_date").size().mean())
    if avg_names < min_names:
        return {
            "slice": slice_label,
            "l1_code": l1_code,
            "l1_name": l1_name,
            "rows": int(len(valid)),
            "avg_names_per_date": avg_names,
            "skipped": True,
            "reason": "below_min_names",
        }

    per_ic = per_date_rank_ic(valid, score_column, ret_col, min_names=min_names)
    ic_summary = summarize_ic(per_ic)
    quantiles = quantile_monotonicity(valid, score_column, ret_col, quantiles=5, min_names=min_names)

    top_k_stats: Dict[str, Any] = {}
    for top_k in top_k_values:
        portfolio_returns: List[float] = []
        broad_excess: List[float] = []
        industry_excess: List[float] = []
        for score_date, group in valid.groupby("score_date", sort=True):
            day = group.sort_values(score_column, ascending=False).head(int(top_k))
            if day.empty:
                continue
            port_ret = float(day[ret_col].mean())
            portfolio_returns.append(port_ret)
            broad_bench = _benchmark_forward_return(broad_benchmark_prices, str(score_date), horizon)
            if broad_bench is not None:
                broad_excess.append(port_ret - broad_bench)
            if industry_benchmark_prices:
                ind_bench = _benchmark_forward_return(industry_benchmark_prices, str(score_date), horizon)
                if ind_bench is not None:
                    industry_excess.append(port_ret - ind_bench)

        def _mean(values: List[float]) -> Optional[float]:
            return float(np.mean(values)) if values else None

        top_k_stats[str(top_k)] = {
            "mean_portfolio_return": _mean(portfolio_returns),
            "mean_excess_vs_broad": _mean(broad_excess),
            "mean_excess_vs_industry_index": _mean(industry_excess),
            "observation_days": len(portfolio_returns),
        }

    return {
        "slice": slice_label,
        "l1_code": l1_code,
        "l1_name": l1_name,
        "rows": int(len(valid)),
        "avg_names_per_date": avg_names,
        "skipped": False,
        "ic": ic_summary,
        "quantile_monotonicity": quantiles,
        "top_k": top_k_stats,
    }


def evaluate_verdict(
    anchor: Dict[str, Any],
    industries: Sequence[Dict[str, Any]],
    *,
    primary_horizon: int,
    min_distinct_l1: int = 5,
    ic_delta_threshold: float = 0.03,
    excess_delta_threshold: float = 0.005,
    primary_top_k: int = 8,
) -> Dict[str, Any]:
    reasons: List[str] = []
    anchor_ic = (anchor.get("ic") or {}).get("ic_mean")
    anchor_excess_industry = None
    anchor_top_k = anchor.get("top_k") or {}
    anchor_k = anchor_top_k.get(str(primary_top_k)) or {}
    anchor_excess_broad = anchor_k.get("mean_excess_vs_broad")

    distinct_count = 0
    for row in industries:
        if row.get("skipped"):
            continue
        ic_mean = (row.get("ic") or {}).get("ic_mean")
        top_k = row.get("top_k") or {}
        k_stats = top_k.get(str(primary_top_k)) or {}
        excess_ind = k_stats.get("mean_excess_vs_industry_index")
        excess_broad = k_stats.get("mean_excess_vs_broad")

        ic_diff = None
        if anchor_ic is not None and ic_mean is not None:
            ic_diff = float(ic_mean) - float(anchor_ic)
        excess_ind_diff = None
        if anchor_excess_broad is not None and excess_ind is not None:
            excess_ind_diff = float(excess_ind) - float(anchor_excess_broad)

        material = False
        if ic_diff is not None and abs(ic_diff) >= ic_delta_threshold:
            material = True
        if excess_ind_diff is not None and abs(excess_ind_diff) >= excess_delta_threshold:
            material = True
        if material:
            distinct_count += 1

    proceed = distinct_count >= min_distinct_l1
    if proceed:
        reasons.append(
            f"{distinct_count} L1 buckets differ from anchor on IC or industry-index Top-K excess "
            f"(thresholds ic>={ic_delta_threshold}, excess>={excess_delta_threshold})."
        )
    else:
        reasons.append(
            f"FAIL: only {distinct_count} L1 buckets materially differ from anchor "
            f"(need >= {min_distinct_l1})."
        )

    skipped = sum(1 for row in industries if row.get("skipped"))
    if skipped > len(industries) * 0.6:
        reasons.append(f"WARN: {skipped}/{len(industries)} L1 buckets skipped (low coverage).")

    return {
        "proceed": proceed,
        "verdict": "proceed" if proceed else "stop",
        "distinct_material_l1": distinct_count,
        "reasons": reasons,
        "primary_horizon": primary_horizon,
        "primary_top_k": primary_top_k,
    }


def run_diagnosis(
    dataset: pd.DataFrame,
    db_manager,
    *,
    score_column: str,
    horizons: Sequence[int],
    top_k_values: Sequence[int],
    min_names: int,
    broad_benchmark_symbol: str,
    start_date: str,
) -> Dict[str, Any]:
    enriched = attach_sw_l1(dataset, db_manager, as_of_column="score_date")
    coverage = sw_l1_coverage_summary(enriched)
    catalog = sw_l1_catalog(db_manager)
    code_to_name = {item["code"]: item["name"] for item in catalog}

    broad_prices = _clean_benchmark_prices(
        load_index_benchmark_prices(db_manager, broad_benchmark_symbol, start_date)
    )

    primary_horizon = int(horizons[-1]) if horizons else 20
    anchor_slices: Dict[str, Any] = {}
    industry_slices: List[Dict[str, Any]] = []

    for horizon in horizons:
        anchor_slices[str(horizon)] = _evaluate_slice(
            enriched,
            score_column=score_column,
            horizon=int(horizon),
            top_k_values=top_k_values,
            min_names=min_names,
            broad_benchmark_prices=broad_prices,
            slice_label="all",
        )

    l1_codes = sorted(
        {
            str(code)
            for code in enriched[SW_L1_CODE_COL].fillna("").astype(str).tolist()
            if str(code).strip()
        }
    )

    industry_by_horizon: Dict[str, List[Dict[str, Any]]] = {}
    for horizon in horizons:
        per_horizon: List[Dict[str, Any]] = []
        for l1_code in l1_codes:
            subset = enriched[enriched[SW_L1_CODE_COL].astype(str) == l1_code]
            industry_prices = _clean_benchmark_prices(
                load_index_benchmark_prices(db_manager, l1_code, start_date)
            )
            per_horizon.append(
                _evaluate_slice(
                    subset,
                    score_column=score_column,
                    horizon=int(horizon),
                    top_k_values=top_k_values,
                    min_names=min_names,
                    broad_benchmark_prices=broad_prices,
                    industry_benchmark_prices=industry_prices,
                    slice_label=l1_code,
                    l1_code=l1_code,
                    l1_name=code_to_name.get(l1_code, l1_code),
                )
            )
        industry_by_horizon[str(horizon)] = per_horizon

    primary_industries = industry_by_horizon.get(str(primary_horizon), [])
    verdict = evaluate_verdict(
        anchor_slices.get(str(primary_horizon), {}),
        primary_industries,
        primary_horizon=primary_horizon,
    )

    return {
        "score_column": score_column,
        "horizons": list(horizons),
        "top_k_values": list(top_k_values),
        "min_names": int(min_names),
        "broad_benchmark_symbol": broad_benchmark_symbol,
        "pit_note": (
            "SW L1 membership is point-in-time: resolved from sw_index_member "
            "stints (in_date <= score_date <= out_date)."
        ),
        "coverage": coverage,
        "anchor": anchor_slices,
        "industries_by_horizon": industry_by_horizon,
        "verdict": verdict,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose score quality across Shenwan L1 industries.")
    parser.add_argument("--start-date", required=True, help="Inclusive score_date lower bound YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="Inclusive score_date upper bound YYYYMMDD")
    parser.add_argument("--universe-index", default="csi1000")
    parser.add_argument("--horizons", nargs="+", type=int, default=[20])
    parser.add_argument("--top-k", nargs="+", type=int, default=[5, 8])
    parser.add_argument("--min-names", type=int, default=15)
    parser.add_argument("--score-column", default="composite_growth_cycle_score")
    parser.add_argument("--growth-cycle-weights", default="30:70")
    parser.add_argument("--broad-benchmark-symbol", default="000852.SH")
    parser.add_argument("--output", help="Write JSON report path")
    parser.add_argument("--enable-tushare", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    from shared.data_access import open_research_data_access_for_cli

    score_column = str(args.score_column)
    sweeps = parse_growth_cycle_weights([args.growth_cycle_weights])
    if sweeps and score_column == "composite_growth_cycle_score":
        score_column = sweeps[0].score_column

    horizons = tuple(int(value) for value in args.horizons)
    config = BacktestConfig(
        horizons=horizons,
        start_date=args.start_date,
        end_date=args.end_date,
        universe_index_codes=(args.universe_index,),
        execution_price="next_open",
        min_names_per_date=int(args.min_names),
    )
    backtester = ScoreBacktester(open_research_data_access_for_cli(enable_tushare=args.enable_tushare), config)
    dataset = backtester.build_dataset()
    if sweeps:
        dataset = add_growth_cycle_weight_columns(dataset, sweeps)

    report = run_diagnosis(
        dataset,
        backtester.db_manager,
        score_column=score_column,
        horizons=horizons,
        top_k_values=tuple(int(value) for value in args.top_k),
        min_names=int(args.min_names),
        broad_benchmark_symbol=str(args.broad_benchmark_symbol),
        start_date=str(args.start_date),
    )
    report["params"] = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "universe_index": args.universe_index,
        "growth_cycle_weights": args.growth_cycle_weights,
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"written: {out}", file=sys.stderr)

    verdict = report.get("verdict", {})
    print(f"verdict={verdict.get('verdict')}", file=sys.stderr)
    return 0 if verdict.get("verdict") == "proceed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
