#!/usr/bin/env python3
"""Evaluate the per-name industry-regime overlay against rollout gate criteria.

Pairs each ``industry_regime_scale`` arm with the ``scale=1.0`` (overlay off)
arm of the *same* portfolio configuration and asks whether shrinking a name
because its own SW L1 industry is not bull earns its keep.

The headline metric is an exposure-matched log excess, not raw cumulative
return. ``scale=0`` removes roughly half the book on average, and a book that
holds less cannot be compared against a full one by cumulative return alone --
in a rising market the smaller book loses mechanically, which says nothing
about the overlay. The matched baseline is a statically scaled buy-and-hold:

    X(W) = ln(1 + cumulative_return) - w_bar * ln(1 + index_cumulative_return)

which is the constant-exposure form of the walk-forward engine's
``X(W) = sum ln f_t - sum ln b_t``. The raw index excess is still reported for
reference; it does not take part in the verdict.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

OVERLAY_OFF_SCALE = 1.0
_SCALE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class IndustryRegimeGateDecision:
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


def _row_scale(row: Dict[str, Any]) -> Optional[float]:
    scale = _as_float(row.get("industry_regime_scale"))
    return scale


def _group_key(row: Dict[str, Any]) -> tuple:
    """Everything that must match for the two arms to be comparable."""
    return (
        row.get("score_variant") or row.get("score_column"),
        row.get("variant"),
        row.get("construction_mode"),
        row.get("top_n"),
        row.get("rebalance_interval_days"),
        row.get("trailing_stop_pct"),
        row.get("path_gate"),
        row.get("volatility_variant"),
        row.get("regime_mode") or "off",
        row.get("regime_rule"),
        row.get("non_bull_action"),
        row.get("bull_exposure"),
        row.get("crowd_exit_k"),
        row.get("intra_period_refill"),
    )


def exposure_matched_log_excess(
    row: Dict[str, Any],
    period_index_returns: Optional[Sequence[float]] = None,
) -> Optional[float]:
    """Log excess over a buy-and-hold scaled to the row's average exposure.

    With ``period_index_returns`` this is the engine's own definition,
    ``sum_t ln(1 + w_bar * r_t)``, evaluated period by period.

    Without them it falls back to the closed form ``w_bar * ln(1 + I)``, which
    is exact only at ``w_bar == 1``. Below that it is **optimistic**: for the
    2021-2026 csi1000 window the closed form reads about +4pp high once
    ``w_bar`` drops to ~0.5. Deltas between two arms at similar exposure mostly
    cancel that error; deltas between a fully invested arm and a half-invested
    one (1.000 vs 0.585) do not, and read ~4pp too favourable. Pass the period
    returns whenever the arms' exposures differ materially.
    """
    cumulative = _as_float(row.get("cumulative_return"))
    exposure = _as_float(row.get("average_exposure"))
    if cumulative is None or exposure is None or 1.0 + cumulative <= 0.0:
        return None
    if period_index_returns:
        matched_log = sum(math.log1p(exposure * float(ret)) for ret in period_index_returns)
        return math.log1p(cumulative) - matched_log
    index_cumulative = _as_float(row.get("index_benchmark_cumulative_return"))
    if index_cumulative is None or 1.0 + index_cumulative <= 0.0:
        return None
    return math.log1p(cumulative) - exposure * math.log1p(index_cumulative)


def _yearly_return(row: Dict[str, Any], year: str) -> Optional[float]:
    yearly = row.get("yearly_returns")
    if not isinstance(yearly, dict):
        return None
    return _as_float(yearly.get(str(year)))


def evaluate_industry_regime_ab(
    rows: Sequence[Dict[str, Any]],
    *,
    baseline_scale: float = OVERLAY_OFF_SCALE,
    candidate_scales: Optional[Sequence[float]] = None,
    crash_year: Optional[str] = None,
    max_turnover_increase: float = 0.15,
    period_index_returns: Optional[Sequence[float]] = None,
) -> IndustryRegimeGateDecision:
    """Compare each scaled arm against the overlay-off arm of the same config."""
    baseline_by_key: Dict[tuple, Dict[str, Any]] = {}
    candidates_by_key: Dict[tuple, Dict[float, Dict[str, Any]]] = {}

    for row in rows:
        scale = _row_scale(row)
        if scale is None:
            continue
        key = _group_key(row)
        if abs(scale - float(baseline_scale)) <= _SCALE_TOLERANCE:
            baseline_by_key[key] = row
            continue
        if candidate_scales is not None and not any(
            abs(scale - float(value)) <= _SCALE_TOLERANCE for value in candidate_scales
        ):
            continue
        candidates_by_key.setdefault(key, {})[scale] = row

    comparisons: List[Dict[str, Any]] = []
    reasons: List[str] = []
    passed_pairs = 0
    total_pairs = 0

    for key, baseline in baseline_by_key.items():
        for scale, candidate in sorted(candidates_by_key.get(key, {}).items()):
            total_pairs += 1
            label = f"{key} scale={scale:g}"

            base_matched = exposure_matched_log_excess(baseline, period_index_returns)
            cand_matched = exposure_matched_log_excess(candidate, period_index_returns)
            base_approx = exposure_matched_log_excess(baseline)
            cand_approx = exposure_matched_log_excess(candidate)
            base_excess = _as_float(baseline.get("index_excess_cumulative_return"))
            cand_excess = _as_float(candidate.get("index_excess_cumulative_return"))
            base_mdd = _as_float(baseline.get("max_drawdown"))
            cand_mdd = _as_float(candidate.get("max_drawdown"))
            base_exposure = _as_float(baseline.get("average_exposure"))
            cand_exposure = _as_float(candidate.get("average_exposure"))
            base_turnover = _as_float(baseline.get("average_turnover"))
            cand_turnover = _as_float(candidate.get("average_turnover"))

            matched_delta = (
                cand_matched - base_matched
                if base_matched is not None and cand_matched is not None
                else None
            )
            mdd_improve = cand_mdd - base_mdd if base_mdd is not None and cand_mdd is not None else None
            turnover_delta = (
                cand_turnover - base_turnover
                if base_turnover is not None and cand_turnover is not None
                else None
            )
            crash_base = _yearly_return(baseline, crash_year) if crash_year else None
            crash_cand = _yearly_return(candidate, crash_year) if crash_year else None
            crash_delta = (
                crash_cand - crash_base
                if crash_base is not None and crash_cand is not None
                else None
            )

            pair_pass = True
            # A delta of exactly zero means the overlay never bound on this
            # configuration (no non-bull name survived into the book). That is
            # "no evidence" rather than "evidence of harm", so it passes -- but
            # it must be visible, because it also means the pair proves nothing.
            binding = matched_delta is not None and abs(matched_delta) > _SCALE_TOLERANCE
            if matched_delta is None:
                pair_pass = False
                reasons.append(f"{label}: exposure-matched excess not computable")
            elif binding and matched_delta < 0:
                pair_pass = False
                reasons.append(f"{label}: matched excess delta {matched_delta:+.4f} < 0")
            if mdd_improve is not None and mdd_improve < 0:
                pair_pass = False
                reasons.append(f"{label}: max_drawdown worsened by {mdd_improve:+.4f}")
            if crash_delta is not None and crash_delta < 0:
                pair_pass = False
                reasons.append(
                    f"{label}: {crash_year} return worsened by {crash_delta:+.4f}"
                )
            if turnover_delta is not None and turnover_delta > max_turnover_increase:
                pair_pass = False
                reasons.append(
                    f"{label}: turnover increase {turnover_delta:.4f} > {max_turnover_increase}"
                )
            # Reported separately, not part of the pair verdict: it asks whether
            # the whole strategy clears a statically scaled index, which is a
            # question about the strategy rather than about this overlay.
            if cand_matched is not None and cand_matched <= 0:
                reasons.append(
                    f"{label}: candidate matched excess {cand_matched:+.4f} <= 0 (strategy-level, not the overlay's verdict)"
                )

            if pair_pass:
                passed_pairs += 1

            comparisons.append(
                {
                    "key": key,
                    "scale": scale,
                    "binding": binding,
                    "baseline_matched_excess": base_matched,
                    "candidate_matched_excess": cand_matched,
                    "matched_excess_delta": matched_delta,
                    "baseline_matched_excess_closed_form": base_approx,
                    "candidate_matched_excess_closed_form": cand_approx,
                    "closed_form_bias": (
                        (cand_approx - cand_matched) if cand_approx is not None and cand_matched is not None else None
                    ),
                    "baseline_index_excess": base_excess,
                    "candidate_index_excess": cand_excess,
                    "baseline_mdd": base_mdd,
                    "candidate_mdd": cand_mdd,
                    "mdd_improve": mdd_improve,
                    "baseline_exposure": base_exposure,
                    "candidate_exposure": cand_exposure,
                    "baseline_turnover": base_turnover,
                    "candidate_turnover": cand_turnover,
                    "turnover_delta": turnover_delta,
                    f"baseline_{crash_year}_return" if crash_year else "baseline_crash_return": crash_base,
                    f"candidate_{crash_year}_return" if crash_year else "candidate_crash_return": crash_cand,
                    "passed": pair_pass,
                }
            )

    if total_pairs == 0:
        return IndustryRegimeGateDecision(
            passed=False,
            reasons=["No overlapping baseline/candidate parameter combinations found."],
            comparisons=comparisons,
        )

    passed = passed_pairs == total_pairs
    if passed:
        reasons.insert(
            0, f"All {total_pairs} matched pairs passed the industry-regime gate."
        )
    else:
        reasons.insert(
            0, f"{passed_pairs}/{total_pairs} matched pairs passed the industry-regime gate."
        )
    return IndustryRegimeGateDecision(passed=passed, reasons=reasons, comparisons=comparisons)


def summarize_comparisons(comparisons: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mean matched-excess delta per scale, for a quick read of the grid."""
    by_scale: Dict[float, List[float]] = {}
    for entry in comparisons:
        delta = entry.get("matched_excess_delta")
        if delta is None:
            continue
        by_scale.setdefault(float(entry["scale"]), []).append(float(delta))
    binding_by_scale: Dict[float, List[float]] = {}
    for entry in comparisons:
        delta = entry.get("matched_excess_delta")
        if delta is None:
            continue
        if abs(float(delta)) > _SCALE_TOLERANCE:
            binding_by_scale.setdefault(float(entry["scale"]), []).append(float(delta))

    out: List[Dict[str, Any]] = []
    for scale in sorted(by_scale):
        values = by_scale[scale]
        binding_values = binding_by_scale.get(scale, [])
        out.append(
            {
                "scale": scale,
                "pairs": len(values),
                "binding_pairs": len(binding_values),
                "mean_matched_excess_delta": sum(values) / len(values),
                # Averaging in the zero-delta no-op pairs would understate the
                # overlay's effect on the configurations where it actually binds.
                "mean_matched_excess_delta_when_binding": (
                    sum(binding_values) / len(binding_values) if binding_values else None
                ),
                "min_matched_excess_delta": min(values),
                "max_matched_excess_delta": max(values),
                "positive_pairs": sum(1 for value in values if value > 0),
            }
        )
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the per-name industry-regime overlay A/B sweep JSON."
    )
    parser.add_argument("--sweep-json", required=True, help="Path to portfolio sweep JSON payload")
    parser.add_argument(
        "--baseline-scale",
        type=float,
        default=OVERLAY_OFF_SCALE,
        help="Overlay-off arm; default 1.0",
    )
    parser.add_argument(
        "--candidate-scales",
        nargs="+",
        type=float,
        help="Scales to judge; default all non-baseline scales present in the payload",
    )
    parser.add_argument(
        "--crash-year",
        help="Year the candidate must not underperform in, e.g. 2021",
    )
    parser.add_argument("--max-turnover-increase", type=float, default=0.15)
    parser.add_argument(
        "--index-period-returns",
        help=(
            "JSON list of the benchmark's per-rebalance-period returns, in order. "
            "Supplying it makes the exposure-matched excess exact; omitting it falls "
            "back to a closed form that is optimistic once exposure drops below 1."
        ),
    )
    parser.add_argument("--output", help="Optional JSON output for the gate decision")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = json.loads(Path(args.sweep_json).read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    period_index_returns = None
    if args.index_period_returns:
        period_index_returns = [
            float(value)
            for value in json.loads(Path(args.index_period_returns).read_text(encoding="utf-8"))
        ]
    decision = evaluate_industry_regime_ab(
        rows,
        baseline_scale=args.baseline_scale,
        candidate_scales=args.candidate_scales,
        crash_year=args.crash_year,
        max_turnover_increase=args.max_turnover_increase,
        period_index_returns=period_index_returns,
    )
    out = {
        "passed": decision.passed,
        "metric": (
            "exact_period_matched" if period_index_returns else "closed_form_approx"
        ),
        "baseline_scale": args.baseline_scale,
        "crash_year": args.crash_year,
        "reasons": decision.reasons,
        "by_scale": summarize_comparisons(decision.comparisons),
        "comparisons": decision.comparisons,
    }
    text = json.dumps(out, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    return 0 if decision.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
