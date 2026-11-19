#!/usr/bin/env python3
"""Evaluate path_gate A/B sweep results against rollout gate criteria."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class PathGateDecision:
    passed: bool
    recommended_path_gate: Optional[str]
    baseline_rows: int
    candidate_rows: int
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
    )


def evaluate_path_gate_ab(
    rows: Sequence[Dict[str, Any]],
    *,
    min_drawdown_improvement: float = 0.03,
    max_index_excess_drop: float = 0.02,
    max_turnover_increase_ratio: float = 0.30,
    preferred_gate: str = "ret20_pos",
) -> PathGateDecision:
    """Compare path_gate=off baseline vs gated modes on drawdown, excess, turnover."""
    baseline_by_key: Dict[tuple, Dict[str, Any]] = {}
    candidates: List[Dict[str, Any]] = []
    reasons: List[str] = []
    comparisons: List[Dict[str, Any]] = []

    for row in rows:
        gate = str(row.get("path_gate") or "off").lower()
        if gate == "off":
            baseline_by_key[_group_key(row)] = row
        elif gate in ("ret20_pos", "above_ma60"):
            candidates.append(row)

    if not baseline_by_key:
        return PathGateDecision(
            passed=False,
            recommended_path_gate=None,
            baseline_rows=0,
            candidate_rows=len(candidates),
            reasons=["No path_gate=off baseline rows found in sweep output."],
            comparisons=[],
        )

    passing: List[Dict[str, Any]] = []
    for candidate in candidates:
        baseline = baseline_by_key.get(_group_key(candidate))
        if not baseline:
            continue
        base_dd = _as_float(baseline.get("max_drawdown"))
        cand_dd = _as_float(candidate.get("max_drawdown"))
        base_excess = _as_float(baseline.get("index_excess_cumulative_return"))
        cand_excess = _as_float(candidate.get("index_excess_cumulative_return"))
        base_turnover = _as_float(baseline.get("average_turnover"))
        cand_turnover = _as_float(candidate.get("average_turnover"))

        drawdown_ok = (
            base_dd is not None
            and cand_dd is not None
            and cand_dd >= base_dd + min_drawdown_improvement
        )
        excess_ok = (
            base_excess is None
            or cand_excess is None
            or cand_excess >= base_excess - max_index_excess_drop
        )
        turnover_ok = True
        if base_turnover is not None and cand_turnover is not None and base_turnover > 0:
            turnover_ok = (cand_turnover - base_turnover) / base_turnover < max_turnover_increase_ratio

        passed_pair = drawdown_ok and excess_ok and turnover_ok
        comparison = {
            "group": _group_key(candidate),
            "path_gate": candidate.get("path_gate"),
            "baseline_max_drawdown": base_dd,
            "candidate_max_drawdown": cand_dd,
            "baseline_index_excess": base_excess,
            "candidate_index_excess": cand_excess,
            "baseline_turnover": base_turnover,
            "candidate_turnover": cand_turnover,
            "passed": passed_pair,
        }
        comparisons.append(comparison)
        if passed_pair:
            passing.append(candidate)

    if not passing:
        reasons.append(
            "No candidate beat off on drawdown (>=3pt), index excess (<=2pt drop), and turnover (<30% increase)."
        )
        return PathGateDecision(
            passed=False,
            recommended_path_gate=None,
            baseline_rows=len(baseline_by_key),
            candidate_rows=len(candidates),
            reasons=reasons,
            comparisons=comparisons,
        )

    gate_rank = {"ret20_pos": 0, "above_ma60": 1}
    passing.sort(
        key=lambda row: (
            gate_rank.get(str(row.get("path_gate") or ""), 99),
            _as_float(row.get("max_drawdown")) or -999,
            -(_as_float(row.get("index_excess_cumulative_return")) or -999),
        ),
        reverse=True,
    )
    preferred = [row for row in passing if str(row.get("path_gate")) == preferred_gate]
    winner = preferred[0] if preferred else passing[0]
    reasons.append("Selected candidate improves drawdown vs off within excess and turnover tolerance.")
    return PathGateDecision(
        passed=True,
        recommended_path_gate=str(winner.get("path_gate")),
        baseline_rows=len(baseline_by_key),
        candidate_rows=len(candidates),
        reasons=reasons,
        comparisons=comparisons,
    )


def load_sweep_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    if not isinstance(rows, list):
        raise ValueError("Sweep JSON must contain a list under 'rows'.")
    return rows


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate path_gate sweep A/B against rollout criteria.")
    parser.add_argument("--sweep-json", required=True, help="Path to portfolio_sweep_report JSON output")
    parser.add_argument("--json-output", help="Optional path to write gate decision JSON")
    args = parser.parse_args()

    decision = evaluate_path_gate_ab(load_sweep_rows(Path(args.sweep_json)))
    payload = {
        "passed": decision.passed,
        "recommended_path_gate": decision.recommended_path_gate,
        "baseline_rows": decision.baseline_rows,
        "candidate_rows": decision.candidate_rows,
        "reasons": decision.reasons,
        "comparisons": decision.comparisons,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.json_output:
        Path(args.json_output).write_text(text, encoding="utf-8")
    return 0 if decision.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
