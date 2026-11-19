#!/usr/bin/env python3
"""Evaluate volatility position-scaling A/B sweep results against rollout gate criteria."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    recommended_mode: Optional[str]
    recommended_target_vol: Optional[float]
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
    )


def evaluate_volatility_ab(
    rows: Sequence[Dict[str, Any]],
    *,
    min_sharpe_improvement: float = 0.0,
    min_drawdown_improvement: float = 0.0,
    max_index_excess_drop: float = 0.02,
    preferred_mode: str = "cash_drag",
) -> GateDecision:
    """Compare ``off`` baseline vs scaled modes on sharpe, drawdown, and index excess."""
    baseline_by_key: Dict[tuple, Dict[str, Any]] = {}
    candidates: List[Dict[str, Any]] = []
    reasons: List[str] = []
    comparisons: List[Dict[str, Any]] = []

    for row in rows:
        mode = str(row.get("volatility_mode") or "off").lower()
        if mode == "off":
            baseline_by_key[_group_key(row)] = row
        elif mode in ("cash_drag", "renormalize"):
            candidates.append(row)

    if not baseline_by_key:
        return GateDecision(
            passed=False,
            recommended_mode=None,
            recommended_target_vol=None,
            baseline_rows=0,
            candidate_rows=len(candidates),
            reasons=["No volatility_mode=off baseline rows found in sweep output."],
            comparisons=[],
        )

    passing: List[Dict[str, Any]] = []
    for candidate in candidates:
        baseline = baseline_by_key.get(_group_key(candidate))
        if not baseline:
            continue
        base_sharpe = _as_float(baseline.get("sharpe"))
        cand_sharpe = _as_float(candidate.get("sharpe"))
        base_dd = _as_float(baseline.get("max_drawdown"))
        cand_dd = _as_float(candidate.get("max_drawdown"))
        base_excess = _as_float(baseline.get("index_excess_cumulative_return"))
        cand_excess = _as_float(candidate.get("index_excess_cumulative_return"))

        sharpe_ok = (
            base_sharpe is not None
            and cand_sharpe is not None
            and cand_sharpe >= base_sharpe + min_sharpe_improvement
        )
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
        passed_pair = sharpe_ok and drawdown_ok and excess_ok
        comparison = {
            "group": _group_key(candidate),
            "volatility_variant": candidate.get("volatility_variant"),
            "volatility_mode": candidate.get("volatility_mode"),
            "target_vol_annual": candidate.get("target_vol_annual"),
            "baseline_sharpe": base_sharpe,
            "candidate_sharpe": cand_sharpe,
            "baseline_max_drawdown": base_dd,
            "candidate_max_drawdown": cand_dd,
            "baseline_index_excess": base_excess,
            "candidate_index_excess": cand_excess,
            "average_exposure": candidate.get("average_exposure"),
            "passed": passed_pair,
        }
        comparisons.append(comparison)
        if passed_pair:
            passing.append(candidate)

    if not passing:
        reasons.append("No candidate beat baseline on sharpe and drawdown without large index-excess drop.")
        return GateDecision(
            passed=False,
            recommended_mode=None,
            recommended_target_vol=None,
            baseline_rows=len(baseline_by_key),
            candidate_rows=len(candidates),
            reasons=reasons,
            comparisons=comparisons,
        )

    mode_rank = {"cash_drag": 0, "renormalize": 1}
    passing.sort(
        key=lambda row: (
            mode_rank.get(str(row.get("volatility_mode") or ""), 99),
            -(_as_float(row.get("sharpe")) or -999),
            _as_float(row.get("max_drawdown")) or -999,
            -(_as_float(row.get("index_excess_cumulative_return")) or -999),
        )
    )
    if preferred_mode:
        preferred = [row for row in passing if str(row.get("volatility_mode")) == preferred_mode]
        winner = preferred[0] if preferred else passing[0]
    else:
        winner = passing[0]

    reasons.append(
        "Selected candidate improves sharpe and drawdown vs off while keeping index excess within tolerance."
    )
    return GateDecision(
        passed=True,
        recommended_mode=str(winner.get("volatility_mode")),
        recommended_target_vol=_as_float(winner.get("target_vol_annual")),
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


def preset_patch_for_gate(decision: GateDecision) -> Dict[str, Any]:
    if not decision.passed:
        return {}
    return {
        "volatility_position_scaling_enabled": True,
        "target_vol_annual": decision.recommended_target_vol,
    }
