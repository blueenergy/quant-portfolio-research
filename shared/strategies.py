"""Strategy template / plan param contract shared by portfolio and research.

Pure helpers only — no Mongo or portfolio-specific plan/paper logic.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

# Keys that define plan lineage identity (params_hash). Operational toggles
# (trailing_stop_pct, monitor_can_trade, capital_basis, etc.) are excluded.
PLAN_PARAM_KEYS = {
    "universe_index",
    "index_benchmark_symbol",
    "score_type",
    "score_column",
    "model_id",
    "growth_weight",
    "cycle_weight",
    "score_weights",
    "top_n",
    "rebalance_days",
    "construction_mode",
    "max_industry_weight",
    "entry_vol_cap",
}

# Superset used for plan snapshots / display (includes execution tuning fields).
PLAN_SNAPSHOT_PARAM_KEYS = PLAN_PARAM_KEYS | {
    "trailing_stop_pct",
    "lot_size",
    "transaction_cost",
    "buy_commission_rate",
    "sell_commission_rate",
    "min_commission",
    "stamp_tax_rate",
    "transfer_fee_rate",
    "execution_price",
    "cash_buffer",
    "initial_capital",
    "candidate_pool_enabled",
    "candidate_pool_multiplier",
    "min_buy_lots",
    "regime_cash",
    "regime_rule",
}


def effective_strategy_config(
    template: Dict[str, Any],
    preset: Optional[Dict[str, Any]] = None,
    params_override: Optional[Dict[str, Any]] = None,
    *,
    exact_override: bool = False,
) -> Dict[str, Any]:
    """Resolve the config a plan is generated with.

    ``exact_override`` selects how a ``None`` in ``params_override`` is read:

    * ``False`` (default) — "not provided": the preset value survives. This is
      the settings-form semantic, where a blank field must not wipe a preset key.
    * ``True`` — "explicitly None": the ``None`` is applied verbatim to lineage
      keys. Callers that rebuild a plan from an anchor (force rebalance) need
      this, because a lineage key that is genuinely ``None`` (e.g.
      ``max_industry_weight``) must come back as ``None``. Dropping it lets the
      *preset's* value take over, which silently changes ``plan_params_hash``
      and lands the plan on a different lineage.

    Only ``PLAN_PARAM_KEYS`` get the exact treatment. Those are the keys that
    define lineage identity, and narrowing the change keeps a non-lineage
    ``None`` from reaching consumers that would choke on it
    (``int(config.get("lot_size", 100))`` and friends).
    """
    params = dict((preset or {}).get("params") or {})
    override = dict(params_override or {})
    if exact_override:
        params.update({key: value for key, value in override.items() if value is not None or key in PLAN_PARAM_KEYS})
    else:
        params.update({key: value for key, value in override.items() if value is not None})
    return {
        **params,
        "strategy_template_id": template.get("strategy_template_id"),
        "strategy_type": template.get("strategy_type", "template"),
        "name": template.get("name"),
        "algorithm": template.get("algorithm"),
        # A preset/override may select a different scoring path (e.g.
        # score_column for a promoted model). The template value is only the
        # default; it must not clobber an explicit override.
        "score_type": params.get("score_type") or template.get("score_type"),
        "version": template.get("version"),
    }


def normalized_plan_params(config: Dict[str, Any]) -> Dict[str, Any]:
    return {key: config.get(key) for key in sorted(PLAN_SNAPSHOT_PARAM_KEYS) if key in config}


def normalized_lineage_params(config: Dict[str, Any]) -> Dict[str, Any]:
    return {key: config.get(key) for key in sorted(PLAN_PARAM_KEYS) if key in config}


def selected_preset_id_for_plan(
    template: Dict[str, Any],
    preset: Optional[Dict[str, Any]],
    final_config: Dict[str, Any],
) -> Optional[str]:
    if not preset:
        return None
    preset_config = effective_strategy_config(template, preset, None)
    if normalized_plan_params(preset_config) != normalized_plan_params(final_config):
        return None
    return preset.get("preset_id")


def strategy_display_name(config: Dict[str, Any], strategy_template_id: str) -> str:
    """Human-readable lineage name.

    Model-score lineages share the ``growth_cycle_index_enhanced`` template but
    are a different strategy; calling them 成长周期指数增强 is misleading. Derive
    a model-specific label from the config before falling back to the template.
    """
    if str(config.get("score_type") or "") == "score_column" and str(config.get("model_id") or "").strip():
        universe = str(config.get("universe_index") or "").upper()
        parts = [part for part in (universe, "模型评分") if part]
        if config.get("top_n") is not None:
            parts.append(f"Top{int(config['top_n'])}")
        if config.get("rebalance_days") is not None:
            parts.append(f"{int(config['rebalance_days'])}日")
        return " ".join(parts) or "模型评分"
    if config.get("name"):
        return str(config["name"])
    if str(config.get("score_type") or "").startswith("growth_cycle") or str(strategy_template_id).startswith("growth_cycle"):
        return "成长周期指数增强"
    return str(strategy_template_id)


def plan_params_hash(config: Dict[str, Any]) -> str:
    """The lineage hash: 12 hex chars over ``PLAN_PARAM_KEYS`` (11 keys).

    Not to be confused with ``_params_hash`` in quantFinance's
    ``routers/portfolio_research.py``. That one is 16 chars over
    ``PRESET_PARAM_KEYS`` (24 keys) and hashes a preset's raw params rather
    than the normalized lineage subset, so it deliberately includes
    ``trailing_stop_pct`` / ``initial_capital`` / fee rates that this one
    excludes. Both fields are named ``params_hash`` on their documents, but
    the values live in different namespaces and comparing them silently
    never matches.
    """
    payload = json.dumps(normalized_lineage_params(config), sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def build_user_plan_id(strategy_template_id: str, base_date: str, user_id: Optional[str], params_hash: Optional[str] = None) -> str:
    hash_part = f"_{params_hash[:8]}" if params_hash else ""
    if not user_id:
        return f"{strategy_template_id}_{base_date}{hash_part}"
    safe_user = "".join(ch if ch.isalnum() else "" for ch in str(user_id))[:12] or "user"
    return f"{strategy_template_id}_{base_date}{hash_part}_{safe_user}"
