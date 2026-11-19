#!/usr/bin/env python3
"""Evaluate industry L1 filter A/B sweep results against rollout gate criteria."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class IndustryGateDecision:
    passed: bool
    reasons: List[str]
    comparisons: List[Dict[str, Any]]


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _group_key(row: Dict[str, Any]) -> tuple:
    return (
        row.get("score_variant") or row.get("score_column"),
        row.get("variant"),
        row.get("top_n"),
        row.get("rebalance_interval_days"),
        row.get("trailing_stop_pct"),
        row.get("path_gate"),
        row.get("volatility_variant"),
    )


def _row_industry_l1(row: Dict[str, Any]) -> Optional[str]:
    value = row.get("industry_l1")
    if value is None or value == "":
        codes = row.get("industry_l1_codes")
        if isinstance(codes, list) and codes:
            return str(codes[0])
        return None
    if isinstance(value, list):
        return str(value[0]) if value else None
    return str(value)


def evaluate_industry_ab(
    rows: Sequence[Dict[str, Any]],
    *,
    industry_l1: str,
    min_selected_count: float = 3.0,
    max_turnover_increase: float = 0.15,
    max_index_excess_drop: float = 0.02,
) -> IndustryGateDecision:
    """Compare full-universe baseline rows vs rows filtered to ``industry_l1``."""
    baseline_by_key: Dict[tuple, Dict[str, Any]] = {}
    candidate_by_key: Dict[tuple, Dict[str, Any]] = {}
    target = str(industry_l1).strip()

    for row in rows:
        key = _group_key(row)
        industry = _row_industry_l1(row)
        if industry:
            if industry == target:
                candidate_by_key[key] = row
        else:
            baseline_by_key[key] = row

    comparisons: List[Dict[str, Any]] = []
    reasons: List[str] = []
    passed_pairs = 0
    total_pairs = 0

    for key, baseline in baseline_by_key.items():
        candidate = candidate_by_key.get(key)
        if not candidate:
            continue
        total_pairs += 1
        base_excess = _as_float(baseline.get("index_excess_cumulative_return"))
        cand_excess = _as_float(candidate.get("index_excess_cumulative_return"))
        base_mdd = _as_float(baseline.get("max_drawdown"))
        cand_mdd = _as_float(candidate.get("max_drawdown"))
        base_turnover = _as_float(baseline.get("average_turnover"))
        cand_turnover = _as_float(candidate.get("average_turnover"))
        cand_selected = _as_float(candidate.get("average_selected_count"))

        excess_drop = None
        if base_excess is not None and cand_excess is not None:
            excess_drop = base_excess - cand_excess
        mdd_improve = None
        if base_mdd is not None and cand_mdd is not None:
            mdd_improve = cand_mdd - base_mdd
        turnover_delta = None
        if base_turnover is not None and cand_turnover is not None:
            turnover_delta = cand_turnover - base_turnover

        pair_pass = True
        if excess_drop is not None and excess_drop > max_index_excess_drop:
            pair_pass = False
            reasons.append(f"{key}: index excess drop {excess_drop:.4f} > {max_index_excess_drop}")
        if cand_selected is not None and cand_selected < min_selected_count:
            pair_pass = False
            reasons.append(f"{key}: average_selected_count {cand_selected:.1f} < {min_selected_count}")
        if turnover_delta is not None and turnover_delta > max_turnover_increase:
            pair_pass = False
            reasons.append(f"{key}: turnover increase {turnover_delta:.4f} > {max_turnover_increase}")

        if pair_pass:
            passed_pairs += 1

        comparisons.append(
            {
                "key": key,
                "baseline_index_excess": base_excess,
                "candidate_index_excess": cand_excess,
                "excess_drop": excess_drop,
                "baseline_mdd": base_mdd,
                "candidate_mdd": cand_mdd,
                "mdd_improve": mdd_improve,
                "baseline_turnover": base_turnover,
                "candidate_turnover": cand_turnover,
                "turnover_delta": turnover_delta,
                "candidate_avg_selected": cand_selected,
                "passed": pair_pass,
            }
        )

    if total_pairs == 0:
        return IndustryGateDecision(
            passed=False,
            reasons=["No overlapping baseline/candidate parameter combinations found."],
            comparisons=comparisons,
        )

    passed = passed_pairs == total_pairs
    if passed:
        reasons.insert(0, f"All {total_pairs} matched pairs passed industry A/B gate.")
    else:
        reasons.insert(0, f"{passed_pairs}/{total_pairs} matched pairs passed industry A/B gate.")

    return IndustryGateDecision(passed=passed, reasons=reasons, comparisons=comparisons)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate industry L1 filter A/B sweep JSON.")
    parser.add_argument("--sweep-json", required=True, help="Path to portfolio sweep JSON payload")
    parser.add_argument("--industry-l1", required=True, help="Target Shenwan L1 code, e.g. 801080.SI")
    parser.add_argument("--output", help="Optional JSON output for gate decision")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = json.loads(Path(args.sweep_json).read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    decision = evaluate_industry_ab(rows, industry_l1=args.industry_l1)
    out = {
        "passed": decision.passed,
        "industry_l1": args.industry_l1,
        "reasons": decision.reasons,
        "comparisons": decision.comparisons,
    }
    text = json.dumps(out, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    return 0 if decision.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
