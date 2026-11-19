#!/usr/bin/env python3
"""Evaluate regime-switched growth/cycle weight sweep results against rollout gates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from shared.regime_label import REGIME_SCORE_VARIANT

STATIC_BASELINE_2080 = "growth_cycle_g20_c80"
STATIC_BASELINE_6040 = "growth_cycle_g60_c40"

DEFAULT_MIN_SHARPE_TOLERANCE = 0.05
DEFAULT_MAX_INDEX_EXCESS_DROP = 0.02
DEFAULT_MAX_TURNOVER_RATIO = 1.30
DEFAULT_MIN_DRAWDOWN_IMPROVEMENT = 0.03
DEFAULT_STATIC_EQUIVALENCE_BAND = 0.02


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reasons: List[str]
    comparisons: List[Dict[str, Any]]
    full_window: Dict[str, Any]
    bear_window: Optional[Dict[str, Any]]
    bull_window: Optional[Dict[str, Any]]


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _group_key(row: Dict[str, Any]) -> tuple:
    return (
        row.get("variant"),
        row.get("top_n"),
        row.get("rebalance_interval_days"),
    )


def _pick_row(rows: Sequence[Dict[str, Any]], score_variant: str) -> Optional[Dict[str, Any]]:
    matches = [row for row in rows if str(row.get("score_variant") or "") == score_variant]
    if not matches:
        return None
    matches.sort(
        key=lambda row: (
            -(_as_float(row.get("sharpe")) or -999),
            _as_float(row.get("max_drawdown")) or -999,
            -(_as_float(row.get("index_excess_cumulative_return")) or -999),
        )
    )
    return matches[0]


def _compare_full_window(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    *,
    min_sharpe_tolerance: float,
    max_index_excess_drop: float,
    max_turnover_ratio: float,
) -> Dict[str, Any]:
    base_sharpe = _as_float(baseline.get("sharpe"))
    cand_sharpe = _as_float(candidate.get("sharpe"))
    base_excess = _as_float(baseline.get("index_excess_cumulative_return"))
    cand_excess = _as_float(candidate.get("index_excess_cumulative_return"))
    base_turnover = _as_float(baseline.get("average_turnover"))
    cand_turnover = _as_float(candidate.get("average_turnover"))

    sharpe_ok = (
        base_sharpe is None
        or cand_sharpe is None
        or cand_sharpe >= base_sharpe - min_sharpe_tolerance
    )
    excess_ok = (
        base_excess is None
        or cand_excess is None
        or cand_excess >= base_excess - max_index_excess_drop
    )
    turnover_ok = (
        base_turnover is None
        or cand_turnover is None
        or base_turnover <= 0
        or cand_turnover <= base_turnover * max_turnover_ratio
    )
    return {
        "baseline_variant": baseline.get("score_variant"),
        "candidate_variant": candidate.get("score_variant"),
        "baseline_sharpe": base_sharpe,
        "candidate_sharpe": cand_sharpe,
        "baseline_index_excess": base_excess,
        "candidate_index_excess": cand_excess,
        "baseline_average_turnover": base_turnover,
        "candidate_average_turnover": cand_turnover,
        "sharpe_ok": sharpe_ok,
        "excess_ok": excess_ok,
        "turnover_ok": turnover_ok,
        "passed": sharpe_ok and excess_ok and turnover_ok,
    }


def _compare_bear_window(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    *,
    min_drawdown_improvement: float,
) -> Dict[str, Any]:
    base_dd = _as_float(baseline.get("max_drawdown"))
    cand_dd = _as_float(candidate.get("max_drawdown"))
    drawdown_ok = (
        base_dd is not None
        and cand_dd is not None
        and cand_dd >= base_dd + min_drawdown_improvement
    )
    return {
        "baseline_variant": baseline.get("score_variant"),
        "candidate_variant": candidate.get("score_variant"),
        "baseline_max_drawdown": base_dd,
        "candidate_max_drawdown": cand_dd,
        "drawdown_ok": drawdown_ok,
        "passed": drawdown_ok,
    }


def _compare_bull_window(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    *,
    min_sharpe_tolerance: float,
    max_index_excess_drop: float,
) -> Dict[str, Any]:
    base_sharpe = _as_float(baseline.get("sharpe"))
    cand_sharpe = _as_float(candidate.get("sharpe"))
    base_excess = _as_float(baseline.get("index_excess_cumulative_return"))
    cand_excess = _as_float(candidate.get("index_excess_cumulative_return"))
    sharpe_ok = (
        base_sharpe is None
        or cand_sharpe is None
        or cand_sharpe >= base_sharpe - min_sharpe_tolerance
    )
    excess_ok = (
        base_excess is None
        or cand_excess is None
        or cand_excess >= base_excess - max_index_excess_drop
    )
    return {
        "baseline_variant": baseline.get("score_variant"),
        "candidate_variant": candidate.get("score_variant"),
        "baseline_sharpe": base_sharpe,
        "candidate_sharpe": cand_sharpe,
        "baseline_index_excess": base_excess,
        "candidate_index_excess": cand_excess,
        "sharpe_ok": sharpe_ok,
        "excess_ok": excess_ok,
        "passed": sharpe_ok and excess_ok,
    }


def _regime_near_static_2080(
    payload: Optional[Dict[str, Any]],
    *,
    band: float = DEFAULT_STATIC_EQUIVALENCE_BAND,
) -> bool:
    if not payload:
        return False
    schedule = payload.get("regime_schedule") or []
    if not schedule:
        return False
    weights = [_as_float(item.get("w_growth")) for item in schedule]
    weights = [value for value in weights if value is not None]
    if not weights:
        return False
    mean_weight = sum(weights) / len(weights)
    return abs(mean_weight - 0.2) <= band


def evaluate_regime_weight_ab(
    full_rows: Sequence[Dict[str, Any]],
    *,
    bear_rows: Optional[Sequence[Dict[str, Any]]] = None,
    bull_rows: Optional[Sequence[Dict[str, Any]]] = None,
    full_payload: Optional[Dict[str, Any]] = None,
    min_sharpe_tolerance: float = DEFAULT_MIN_SHARPE_TOLERANCE,
    max_index_excess_drop: float = DEFAULT_MAX_INDEX_EXCESS_DROP,
    max_turnover_ratio: float = DEFAULT_MAX_TURNOVER_RATIO,
    min_drawdown_improvement: float = DEFAULT_MIN_DRAWDOWN_IMPROVEMENT,
) -> GateDecision:
    """Compare static 20:80 / 60:40 baselines against the regime-switched score variant."""
    reasons: List[str] = []
    comparisons: List[Dict[str, Any]] = []

    full_candidate = _pick_row(full_rows, REGIME_SCORE_VARIANT)
    full_baseline_2080 = _pick_row(full_rows, STATIC_BASELINE_2080)
    if full_candidate is None or full_baseline_2080 is None:
        return GateDecision(
            passed=False,
            reasons=[
                "Missing full-window rows for regime candidate or static 20:80 baseline.",
            ],
            comparisons=[],
            full_window={},
            bear_window=None,
            bull_window=None,
        )

    full_result = _compare_full_window(
        full_baseline_2080,
        full_candidate,
        min_sharpe_tolerance=min_sharpe_tolerance,
        max_index_excess_drop=max_index_excess_drop,
        max_turnover_ratio=max_turnover_ratio,
    )
    comparisons.append({"window": "full", **full_result})

    bear_result: Optional[Dict[str, Any]] = None
    if bear_rows:
        bear_candidate = _pick_row(bear_rows, REGIME_SCORE_VARIANT)
        bear_baseline_6040 = _pick_row(bear_rows, STATIC_BASELINE_6040)
        if bear_candidate and bear_baseline_6040:
            bear_result = _compare_bear_window(
                bear_baseline_6040,
                bear_candidate,
                min_drawdown_improvement=min_drawdown_improvement,
            )
            comparisons.append({"window": "algo_bear", **bear_result})
        else:
            reasons.append("Bear window missing regime candidate or static 60:40 baseline.")

    bull_result: Optional[Dict[str, Any]] = None
    if bull_rows:
        bull_candidate = _pick_row(bull_rows, REGIME_SCORE_VARIANT)
        bull_baseline_2080 = _pick_row(bull_rows, STATIC_BASELINE_2080)
        if bull_candidate and bull_baseline_2080:
            bull_result = _compare_bull_window(
                bull_baseline_2080,
                bull_candidate,
                min_sharpe_tolerance=min_sharpe_tolerance,
                max_index_excess_drop=max_index_excess_drop,
            )
            comparisons.append({"window": "algo_bull", **bull_result})
        else:
            reasons.append("Bull window missing regime candidate or static 20:80 baseline.")

    passed = full_result.get("passed", False)
    if not passed:
        reasons.append("Full window failed vs static 20:80 on sharpe/excess/turnover tolerance.")
    if bear_result is not None and not bear_result.get("passed"):
        passed = False
        reasons.append("Algorithm bear segment failed to improve max drawdown vs static 60:40 by 3pt.")
    if bull_result is not None and not bull_result.get("passed"):
        passed = False
        reasons.append("Algorithm bull segment failed vs static 20:80 tolerance (likely lagged risk-on).")
    if _regime_near_static_2080(full_payload) and not (
        full_result.get("passed") and (bear_result is None or bear_result.get("passed"))
    ):
        passed = False
        reasons.append("Regime schedule average weight is near static 20:80 with no clear benefit.")

    if passed:
        reasons.append("Regime-switched weights passed full + algorithm bear/bull segment gates.")

    return GateDecision(
        passed=passed,
        reasons=reasons,
        comparisons=comparisons,
        full_window=full_result,
        bear_window=bear_result,
        bull_window=bull_result,
    )


def load_sweep_payload(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    if not isinstance(rows, list):
        raise ValueError("Sweep JSON must contain a list under 'rows'.")
    return payload
