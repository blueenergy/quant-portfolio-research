"""Normalisation of sweep parameters shared across the sweep subsystem.

Kept separate from ``portfolio_sweep_report`` so that result rendering and
combo details can reuse it without importing the sweep entry point.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from analysis.core.path_features import VALID_PATH_GATES
from analysis.core.trailing_stop import normalize_trailing_stop_pcts, normalize_trailing_stop_ratio
from shared.regime_label import (
    DEFAULT_REGIME_RULE_ID,
    REGIME_MODE_BULL_G60_ELSE_CASH,
    VALID_REGIME_MODES,
    coerce_optional_bool,
    load_regime_rule,
)

NON_BULL_ACTION_CASH = "cash"
NON_BULL_ACTION_HALF_TOPN = "half_topn"
NON_BULL_ACTION_HALF_TOPN_SLOT = "half_topn_slot"
VALID_NON_BULL_ACTIONS = (
    NON_BULL_ACTION_CASH,
    NON_BULL_ACTION_HALF_TOPN,
    NON_BULL_ACTION_HALF_TOPN_SLOT,
)


def normalize_bull_exposures(values: Optional[Sequence[float]] = None) -> List[float]:
    """Return bull-period invested fractions in (0, 1]; default full invest."""
    if values is None:
        return [1.0]
    out: List[float] = []
    for raw in values:
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported bull_exposure: {raw}") from exc
        if number <= 0 or number > 1:
            raise ValueError(f"bull_exposure must be in (0, 1], got {raw}")
        if number not in out:
            out.append(number)
    return out or [1.0]


def normalize_non_bull_actions(values: Optional[Sequence[Any]] = None) -> List[str]:
    """Return non-bull book actions; default cash (current overlay)."""
    if values is None:
        return [NON_BULL_ACTION_CASH]
    aliases = {
        "cash": NON_BULL_ACTION_CASH,
        "flat": NON_BULL_ACTION_CASH,
        "half": NON_BULL_ACTION_HALF_TOPN,
        "half_topn": NON_BULL_ACTION_HALF_TOPN,
        "half_names": NON_BULL_ACTION_HALF_TOPN,
        "half_topn_slot": NON_BULL_ACTION_HALF_TOPN_SLOT,
        "half_slot": NON_BULL_ACTION_HALF_TOPN_SLOT,
        "slot": NON_BULL_ACTION_HALF_TOPN_SLOT,
    }
    out: List[str] = []
    for raw in values:
        key = str(raw).strip().lower()
        if not key:
            continue
        action = aliases.get(key)
        if action is None:
            raise ValueError(f"Unsupported non_bull_action: {raw}")
        if action not in out:
            out.append(action)
    return out or [NON_BULL_ACTION_CASH]


def non_bull_action_config(action: str) -> Tuple[float, bool]:
    """Map a named action to ``(non_bull_top_n_scale, keep_parent_slots)``."""
    normalized = normalize_non_bull_actions([action])[0]
    if normalized == NON_BULL_ACTION_HALF_TOPN:
        return 0.5, False
    if normalized == NON_BULL_ACTION_HALF_TOPN_SLOT:
        return 0.5, True
    return 0.0, False


def normalize_crowd_exit_ks(values: Optional[Sequence[Any]] = None) -> List[int]:
    """Return intra-period crowd-exit thresholds; 0 means off (default)."""
    if values is None:
        return [0]
    out: List[int] = []
    for raw in values:
        try:
            number = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported crowd_exit_k: {raw}") from exc
        if number < 0:
            raise ValueError(f"crowd_exit_k must be >= 0, got {raw}")
        if number not in out:
            out.append(number)
    return out or [0]


def normalize_refill_modes(values: Optional[Sequence[Any]] = None) -> List[bool]:
    """Return intra-period trailing-stop refill flags; default off only."""
    if values is None:
        return [False]
    aliases = {
        "off": False,
        "false": False,
        "0": False,
        "on": True,
        "true": True,
        "1": True,
        "refill": True,
    }
    out: List[bool] = []
    seen = set()
    for raw in values:
        if isinstance(raw, bool):
            flag = bool(raw)
        else:
            key = str(raw).strip().lower()
            if key not in aliases:
                raise ValueError(f"Unsupported refill_mode: {raw}")
            flag = aliases[key]
        if flag not in seen:
            seen.add(flag)
            out.append(flag)
    return out or [False]


def normalize_path_gates(path_gates: Optional[Sequence[str]] = None) -> List[str]:
    """Return effective path_gate values; default ``off`` only."""
    if path_gates is None:
        return ["off"]
    values: List[str] = []
    for raw in path_gates:
        gate = str(raw).strip().lower()
        if not gate:
            continue
        if gate not in VALID_PATH_GATES:
            raise ValueError(f"Unsupported path_gate: {raw}")
        values.append(gate)
    return values or ["off"]


def normalize_regime_modes(regime_modes: Optional[Sequence[str]] = None) -> List[str]:
    """Return effective regime modes; default ``off`` (always invested)."""
    if regime_modes is None:
        return ["off"]
    values: List[str] = []
    aliases = {"cash": "bull_g60_else_cash", "regime_cash": "bull_g60_else_cash"}
    for raw in regime_modes:
        mode = str(raw).strip().lower()
        if not mode:
            continue
        mode = aliases.get(mode, mode)
        if mode not in VALID_REGIME_MODES:
            raise ValueError(f"Unsupported regime_mode: {raw}")
        if mode not in values:
            values.append(mode)
    return values or ["off"]


def resolve_regime_modes(
    *,
    regime_modes: Optional[Sequence[str]] = None,
    regime_cash: Any = None,
) -> List[str]:
    """Resolve first-class regime modes from explicit list or cash flag.

    ``regime_modes`` wins when present. ``regime_cash=True`` enables only
    bull-invest / non-bull-cash; it does not secretly add always-invest.
    """
    if regime_modes is not None:
        return normalize_regime_modes(regime_modes)
    if coerce_optional_bool(regime_cash) is True:
        return normalize_regime_modes([REGIME_MODE_BULL_G60_ELSE_CASH])
    return normalize_regime_modes(None)


def normalize_regime_rules(
    regime_rules: Optional[Sequence[str]] = None,
    *,
    regime_rule: Optional[str] = None,
) -> List[str]:
    """Return deduped regime preset ids for non-bull cash sweeps."""
    if regime_rules is not None:
        values: List[str] = []
        for raw in regime_rules:
            text = str(raw or "").strip()
            if not text:
                continue
            rule_id = load_regime_rule(text).rule_id
            if rule_id not in values:
                values.append(rule_id)
        if not values:
            raise ValueError("regime_rules must include at least one preset")
        return values
    text = str(regime_rule or "").strip()
    if text:
        return [load_regime_rule(text).rule_id]
    return [DEFAULT_REGIME_RULE_ID]


def resolve_regime_rules(
    *,
    regime_rules: Optional[Sequence[str]] = None,
    regime_rule: Optional[str] = None,
) -> List[str]:
    """``regime_rules`` wins when present; else single ``regime_rule`` or default."""
    if regime_rules is not None:
        return normalize_regime_rules(regime_rules)
    return normalize_regime_rules(None, regime_rule=regime_rule)


def count_regime_mode_arms(
    regime_mode_values: Sequence[str],
    regime_rule_values: Sequence[str],
) -> int:
    """Sweep arms for regime: always-invest counts once; each cash rule is separate."""
    arms = 0
    if "off" in regime_mode_values:
        arms += 1
    if REGIME_MODE_BULL_G60_ELSE_CASH in regime_mode_values:
        arms += len(regime_rule_values) or 1
    return arms or 1


INDUSTRY_REGIME_SCALE_OFF = 1.0


def normalize_industry_regime_scales(
    values: Optional[Sequence[Any]] = None,
) -> List[float]:
    """Return per-name industry-regime weight factors; default 1.0 (overlay off).

    Any value in [0, 1] is accepted rather than a fixed menu: 0.0 flattens the
    non-bull industries, 1.0 is the off switch, and everything between halves
    the position. Mirrors ``normalize_bull_exposures``.
    """
    if values is None:
        return [INDUSTRY_REGIME_SCALE_OFF]
    out: List[float] = []
    for raw in values:
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported industry_regime_scale: {raw}") from exc
        if number < 0.0 or number > 1.0:
            raise ValueError(f"industry_regime_scale must be in [0, 1], got {raw}")
        if number not in out:
            out.append(number)
    return out or [INDUSTRY_REGIME_SCALE_OFF]


def resolve_industry_regime_scales(
    *,
    industry_regime_scales: Optional[Sequence[Any]] = None,
    industry_regime_scale: Any = None,
) -> List[float]:
    """``industry_regime_scales`` wins when present; else the single value or off."""
    if industry_regime_scales is not None:
        return normalize_industry_regime_scales(industry_regime_scales)
    if industry_regime_scale is None:
        return normalize_industry_regime_scales(None)
    return normalize_industry_regime_scales([industry_regime_scale])


def industry_regime_scale_config(scale: Any) -> float:
    """Decode one axis value into the backtester's ``industry_regime_scale``."""
    return normalize_industry_regime_scales([scale])[0]
