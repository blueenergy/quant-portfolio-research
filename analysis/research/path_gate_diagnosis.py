#!/usr/bin/env python3
"""Diagnose whether growth-cycle Top-N selections overweight falling knives.

Read-only. Builds the standard score/return panel, attaches path features,
and compares Top-N vs full-universe falling-share and forward outcomes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from analysis.core.path_features import add_path_feature_columns, future_max_drawdown_from_cache
from analysis.core.score_backtester import BacktestConfig, ScoreBacktester
from analysis.sweep.score_columns import ScoreSweep, add_growth_cycle_weight_columns, parse_growth_cycle_weights


def _mean_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(nums)) if nums else None


def _summarize_slice(frame: pd.DataFrame, ret_col: str, horizon: int) -> Dict[str, Any]:
    if frame.empty:
        return {
            "count": 0,
            "falling_share": None,
            "below_ma60_share": None,
            "pool_falling_share": None,
            "falling_share_vs_pool": None,
            "falling_fwd_ret": None,
            "rising_fwd_ret": None,
            "fwd_ret_spread_falling_minus_rising": None,
            "falling_fwd_mdd": None,
            "rising_fwd_mdd": None,
            "fwd_mdd_spread_falling_minus_rising": None,
        }

    ret_20 = pd.to_numeric(frame.get("ret_20"), errors="coerce")
    above_ma60 = pd.to_numeric(frame.get("above_ma60"), errors="coerce")
    fwd = pd.to_numeric(frame.get(ret_col), errors="coerce")
    fwd_mdd = pd.to_numeric(frame.get("future_max_drawdown"), errors="coerce")

    falling = ret_20 < 0
    rising = ret_20 >= 0
    n = len(frame)
    pool_falling_share = float(falling.sum() / n) if n else None

    top_falling = frame.loc[falling]
    top_rising = frame.loc[rising]

    return {
        "count": int(n),
        "falling_share": float(falling.sum() / n) if n else None,
        "below_ma60_share": float((above_ma60 <= 0).sum() / n) if n else None,
        "pool_falling_share": pool_falling_share,
        "falling_share_vs_pool": (
            float(falling.sum() / n - pool_falling_share) if pool_falling_share is not None and n else None
        ),
        "falling_fwd_ret": _mean_or_none(fwd.loc[falling].tolist()),
        "rising_fwd_ret": _mean_or_none(fwd.loc[rising].tolist()),
        "fwd_ret_spread_falling_minus_rising": (
            (_mean_or_none(fwd.loc[falling].tolist()) or 0) - (_mean_or_none(fwd.loc[rising].tolist()) or 0)
            if falling.any() and rising.any()
            else None
        ),
        "falling_fwd_mdd": _mean_or_none(fwd_mdd.loc[falling].tolist()),
        "rising_fwd_mdd": _mean_or_none(fwd_mdd.loc[rising].tolist()),
        "fwd_mdd_spread_falling_minus_rising": (
            (_mean_or_none(fwd_mdd.loc[falling].tolist()) or 0) - (_mean_or_none(fwd_mdd.loc[rising].tolist()) or 0)
            if falling.any() and rising.any()
            else None
        ),
    }


def _per_date_rows(
    dataset: pd.DataFrame,
    *,
    score_column: str,
    top_n: int,
    ret_col: str,
    horizon: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for score_date, group in dataset.groupby("score_date", sort=True):
        valid = group.dropna(subset=["symbol", score_column, ret_col]).copy()
        if valid.empty:
            continue
        valid = valid.sort_values(score_column, ascending=False)
        pool_summary = _summarize_slice(valid, ret_col, horizon)
        top = valid.head(int(top_n))
        top_summary = _summarize_slice(top, ret_col, horizon)
        rows.append(
            {
                "score_date": str(score_date),
                "year": str(score_date)[:4],
                "pool_size": int(len(valid)),
                "top_n": int(top_n),
                "top": top_summary,
                "pool": pool_summary,
                "top_vs_pool_falling_share_delta": (
                    (top_summary.get("falling_share") or 0) - (pool_summary.get("falling_share") or 0)
                    if top_summary.get("falling_share") is not None and pool_summary.get("falling_share") is not None
                    else None
                ),
            }
        )
    return rows


def _aggregate_daily(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Any]:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return {"mean": None, "median": None, "positive_days": 0, "days": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "positive_days": int((arr > 0).sum()),
        "days": int(len(arr)),
    }


def _aggregate_by_year(daily_rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_year: Dict[str, List[Dict[str, Any]]] = {}
    for row in daily_rows:
        by_year.setdefault(str(row.get("year")), []).append(row)

    out: Dict[str, Dict[str, Any]] = {}
    for year, items in sorted(by_year.items()):
        top_falling = [item["top"].get("falling_share") for item in items if item.get("top")]
        pool_falling = [item["pool"].get("falling_share") for item in items if item.get("pool")]
        deltas = [item.get("top_vs_pool_falling_share_delta") for item in items]
        fwd_spread = [item["top"].get("fwd_ret_spread_falling_minus_rising") for item in items if item.get("top")]
        mdd_spread = [item["top"].get("fwd_mdd_spread_falling_minus_rising") for item in items if item.get("top")]
        out[year] = {
            "days": len(items),
            "top_falling_share": _aggregate_daily([{"v": v} for v in top_falling if v is not None], "v"),
            "pool_falling_share": _aggregate_daily([{"v": v} for v in pool_falling if v is not None], "v"),
            "top_vs_pool_delta": _aggregate_daily([{"v": v} for v in deltas if v is not None], "v"),
            "top_fwd_ret_spread": _aggregate_daily([{"v": v} for v in fwd_spread if v is not None], "v"),
            "top_fwd_mdd_spread": _aggregate_daily([{"v": v} for v in mdd_spread if v is not None], "v"),
        }
    return out


def evaluate_verdict(overall: Dict[str, Any], by_year: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Phase 0 gate: proceed to Phase 1 engineering only when hypothesis holds."""
    reasons: List[str] = []
    delta_mean = (overall.get("top_vs_pool_delta") or {}).get("mean")
    delta_positive_days = (overall.get("top_vs_pool_delta") or {}).get("positive_days", 0)
    delta_days = (overall.get("top_vs_pool_delta") or {}).get("days", 0)
    fwd_spread_mean = (overall.get("top_fwd_ret_spread") or {}).get("mean")
    mdd_spread_mean = (overall.get("top_fwd_mdd_spread") or {}).get("mean")

    cond1 = delta_mean is not None and (delta_mean > 0.05 or (delta_days and delta_positive_days > delta_days / 2))
    if cond1:
        reasons.append("Top-N falling_share systematically above pool (delta mean > 5pt or majority of days positive).")
    else:
        reasons.append("FAIL: Top-N falling_share not systematically above pool.")

    cond2 = fwd_spread_mean is not None and fwd_spread_mean <= 0
    if cond2:
        reasons.append("Falling subset forward return not better than rising (spread <= 0).")
    else:
        reasons.append("FAIL: Falling subset forward return not worse than rising.")

    cond3 = mdd_spread_mean is not None and mdd_spread_mean < 0
    if cond3:
        reasons.append("Falling subset future MDD deeper than rising (spread < 0).")
    else:
        reasons.append("FAIL: Falling subset future MDD not clearly worse.")

    proceed = bool(cond1 and cond2 and cond3)
    return {
        "proceed": proceed,
        "verdict": "proceed" if proceed else "stop",
        "conditions": {"falling_share_elevated": cond1, "fwd_ret_worse_when_falling": cond2, "fwd_mdd_worse_when_falling": cond3},
        "reasons": reasons,
        "by_year_snapshot": {
            year: {
                "top_vs_pool_delta_mean": (data.get("top_vs_pool_delta") or {}).get("mean"),
                "top_fwd_ret_spread_mean": (data.get("top_fwd_ret_spread") or {}).get("mean"),
                "top_fwd_mdd_spread_mean": (data.get("top_fwd_mdd_spread") or {}).get("mean"),
            }
            for year, data in by_year.items()
        },
    }


def run_diagnosis(
    dataset: pd.DataFrame,
    price_cache: Dict[str, List[Dict[str, Any]]],
    *,
    score_column: str,
    top_n: int,
    horizon: int,
) -> Dict[str, Any]:
    ret_col = f"forward_return_{horizon}d"
    if dataset.empty:
        raise RuntimeError("Empty dataset")
    missing = [col for col in (score_column, ret_col) if col not in dataset.columns]
    if missing:
        raise RuntimeError(f"Dataset missing columns: {missing}")

    enriched = add_path_feature_columns(dataset, price_cache)
    enriched["future_max_drawdown"] = future_max_drawdown_from_cache(price_cache, enriched, horizon)

    daily = _per_date_rows(enriched, score_column=score_column, top_n=top_n, ret_col=ret_col, horizon=horizon)
    by_year = _aggregate_by_year(daily)

    deltas = [row.get("top_vs_pool_falling_share_delta") for row in daily if row.get("top_vs_pool_falling_share_delta") is not None]
    fwd_spreads = [row["top"].get("fwd_ret_spread_falling_minus_rising") for row in daily if row.get("top")]
    mdd_spreads = [row["top"].get("fwd_mdd_spread_falling_minus_rising") for row in daily if row.get("top")]

    overall = {
        "days": len(daily),
        "top_falling_share": _aggregate_daily([{"v": row["top"].get("falling_share")} for row in daily], "v"),
        "pool_falling_share": _aggregate_daily([{"v": row["pool"].get("falling_share")} for row in daily], "v"),
        "top_vs_pool_delta": _aggregate_daily([{"v": v} for v in deltas], "v"),
        "top_fwd_ret_spread": _aggregate_daily([{"v": v} for v in fwd_spreads if v is not None], "v"),
        "top_fwd_mdd_spread": _aggregate_daily([{"v": v} for v in mdd_spreads if v is not None], "v"),
    }
    verdict = evaluate_verdict(overall, by_year)

    return {
        "score_column": score_column,
        "top_n": int(top_n),
        "horizon": int(horizon),
        "overall": overall,
        "by_year": by_year,
        "daily_preview": daily[:5],
        "daily_count": len(daily),
        "verdict": verdict,
        "overlap_note": (
            "Compare top_vs_pool_delta to distinguish selection bias from broad-market falling regimes."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose falling-knife concentration in growth-cycle Top-N.")
    parser.add_argument("--start-date", required=True, help="Inclusive score_date lower bound YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="Inclusive score_date upper bound YYYYMMDD")
    parser.add_argument("--universe-index", default="csi1000")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--growth-cycle-weights", default="30:70", help="e.g. 30:70")
    parser.add_argument("--json-output", help="Write JSON report path")
    parser.add_argument("--enable-tushare", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    from shared.data_access import open_research_data_access_for_cli

    sweeps = parse_growth_cycle_weights([args.growth_cycle_weights])
    if not sweeps:
        sweeps = [ScoreSweep(name="growth_cycle_g30_c70", score_column="growth_cycle_g30_c70", growth_weight=0.3, cycle_weight=0.7)]
    score_column = sweeps[0].score_column

    config = BacktestConfig(
        horizons=(int(args.horizon),),
        start_date=args.start_date,
        end_date=args.end_date,
        universe_index_codes=(args.universe_index,),
        execution_price="next_open",
    )
    backtester = ScoreBacktester(open_research_data_access_for_cli(enable_tushare=args.enable_tushare), config)
    dataset = backtester.build_dataset()
    dataset = add_growth_cycle_weight_columns(dataset, sweeps)

    report = run_diagnosis(
        dataset,
        backtester.price_cache,
        score_column=score_column,
        top_n=int(args.top_n),
        horizon=int(args.horizon),
    )
    report["params"] = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "universe_index": args.universe_index,
        "growth_cycle_weights": args.growth_cycle_weights,
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Wrote {out}", file=sys.stderr)
    return 0 if report["verdict"]["proceed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
