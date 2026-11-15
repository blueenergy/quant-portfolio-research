"""Validated score specifications for portfolio research.

Single-column specifications rank an existing score column. Weighted
specifications blend the six atomic score dimensions with per-row available
weight normalization.

``rank_weighted`` blends each dimension's cross-sectional percentile rank
instead of its raw score. Raw dimensions differ widely in cross-sectional
dispersion (growth_score spreads roughly twice as wide as cycle_score), so a
raw-score blend gives a dimension influence proportional to its dispersion
rather than to its nominal weight. Ranking equalizes that first, which makes
the nominal weights mean what they say.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence


ATOMIC_SCORE_DIMENSIONS = (
    "technical",
    "fundamental",
    "value",
    "growth",
    "money_flow",
    "cycle",
)
ATOMIC_SCORE_COLUMNS = tuple(f"{dimension}_score" for dimension in ATOMIC_SCORE_DIMENSIONS)
ALLOWED_SCORE_COLUMNS = frozenset(
    {
        "composite_score",
        "composite_balanced_score",
        "composite_aggressive_score",
        "composite_conservative_score",
        "composite_defensive_score",
        "composite_value_oriented_score",
        "composite_trading_oriented_score",
        "composite_growth_oriented_score",
        "composite_cycle_oriented_score",
        "composite_growth_cycle_score",
        *ATOMIC_SCORE_COLUMNS,
    }
)

_DIMENSION_TOKENS = {
    "technical": "t",
    "fundamental": "f",
    "value": "v",
    "growth": "g",
    "money_flow": "m",
    "cycle": "c",
}


@dataclass(frozen=True)
class ParsedScoreSpec:
    mode: str
    column: Optional[str] = None
    weights: Optional[Dict[str, float]] = None


def normalize_score_weights(weights: Mapping[str, Any]) -> Dict[str, float]:
    """Validate and normalize atomic-dimension weights in stable key order."""
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("weighted score spec requires a non-empty weights object")

    parsed: Dict[str, float] = {}
    for raw_dimension, raw_weight in weights.items():
        dimension = str(raw_dimension).strip().lower()
        if dimension not in ATOMIC_SCORE_DIMENSIONS:
            raise ValueError(
                f"Unsupported score dimension: {raw_dimension}. "
                f"Allowed: {', '.join(ATOMIC_SCORE_DIMENSIONS)}"
            )
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid weight for {dimension}: {raw_weight}") from exc
        if weight < 0:
            raise ValueError(f"Score weights must be non-negative: {dimension}={raw_weight}")
        if weight > 0:
            parsed[dimension] = weight

    total = sum(parsed.values())
    if total <= 0:
        raise ValueError("Score weights must have a positive total")
    return {dimension: parsed[dimension] / total for dimension in sorted(parsed)}


def canonical_score_weights(weights: Mapping[str, Any]) -> Dict[str, float]:
    """Return JSON/hash-safe normalized weights with deterministic precision."""
    normalized = normalize_score_weights(weights)
    return {dimension: float(f"{weight:.12g}") for dimension, weight in normalized.items()}


def score_weights_identity(weights: Mapping[str, Any]) -> str:
    canonical = canonical_score_weights(weights)
    return json.dumps(canonical, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _weight_label(weight: float) -> str:
    pct = weight * 100.0
    if abs(pct - round(pct)) < 1e-9:
        return str(int(round(pct)))
    return f"{pct:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def weighted_score_name(weights: Mapping[str, Any], rank_based: bool = False) -> str:
    """Stable human-readable blend name; digest protects precision collisions."""
    canonical = canonical_score_weights(weights)
    parts = [
        f"{_DIMENSION_TOKENS[dimension]}{_weight_label(weight)}"
        for dimension, weight in canonical.items()
    ]
    prefix = "rankblend_" if rank_based else "blend_"
    name = prefix + "_".join(parts)
    if any(abs(weight * 100 - round(weight * 100)) >= 1e-9 for weight in canonical.values()):
        digest = hashlib.sha256(score_weights_identity(canonical).encode("utf-8")).hexdigest()[:6]
        name = f"{name}_{digest}"
    return name


def parse_score_spec(raw_spec: Any) -> ParsedScoreSpec:
    """Parse a mapping, JSON object, column name, or ``dimension:weight`` list."""
    if isinstance(raw_spec, str):
        text = raw_spec.strip()
        if not text:
            raise ValueError("score spec cannot be empty")
        if text.startswith("{"):
            try:
                raw_spec = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid score spec JSON: {text}") from exc
        elif ":" in text:
            weights: Dict[str, str] = {}
            for item in text.split(","):
                if ":" not in item:
                    raise ValueError(f"Invalid weighted score spec item: {item}")
                dimension, weight = item.split(":", 1)
                weights[dimension.strip()] = weight.strip()
            return ParsedScoreSpec(mode="weighted", weights=canonical_score_weights(weights))
        else:
            raw_spec = {"mode": "column", "column": text}

    if not isinstance(raw_spec, Mapping):
        raise ValueError(f"Score spec must be a mapping or string: {raw_spec!r}")

    mode = str(raw_spec.get("mode") or "").strip().lower()
    if mode == "column":
        column = str(raw_spec.get("column") or "").strip()
        if column not in ALLOWED_SCORE_COLUMNS:
            raise ValueError(
                f"Unsupported score column: {column}. "
                f"Allowed: {', '.join(sorted(ALLOWED_SCORE_COLUMNS))}"
            )
        return ParsedScoreSpec(mode="column", column=column)
    if mode in ("weighted", "rank_weighted"):
        return ParsedScoreSpec(
            mode=mode,
            weights=canonical_score_weights(raw_spec.get("weights") or {}),
        )
    raise ValueError(f"Unsupported score spec mode: {mode or '<empty>'}")


def parse_score_specs(raw_specs: Optional[Sequence[Any]]) -> list[ParsedScoreSpec]:
    if not raw_specs:
        return []
    return [parse_score_spec(raw_spec) for raw_spec in raw_specs]
