"""Unit tests for shared strategy / params_hash contract (PR-A1)."""

from __future__ import annotations

from shared.strategies import (
    PLAN_PARAM_KEYS,
    PLAN_SNAPSHOT_PARAM_KEYS,
    build_user_plan_id,
    effective_strategy_config,
    normalized_lineage_params,
    normalized_plan_params,
    plan_params_hash,
    selected_preset_id_for_plan,
    strategy_display_name,
)


def test_plan_param_keys_exclude_operational_toggles():
    assert "trailing_stop_pct" not in PLAN_PARAM_KEYS
    assert "monitor_can_trade" not in PLAN_PARAM_KEYS
    assert "regime_cash" not in PLAN_PARAM_KEYS
    assert "trailing_stop_pct" in PLAN_SNAPSHOT_PARAM_KEYS
    assert "regime_cash" in PLAN_SNAPSHOT_PARAM_KEYS
    assert "regime_rule" in PLAN_SNAPSHOT_PARAM_KEYS
    assert "regime_rule" not in PLAN_PARAM_KEYS


def test_plan_params_hash_ignores_regime_cash():
    base = {
        "universe_index": "csi1000",
        "score_type": "growth_cycle",
        "top_n": 30,
        "rebalance_days": 20,
    }
    assert plan_params_hash({**base, "regime_cash": True}) == plan_params_hash(base)
    assert plan_params_hash({**base, "regime_rule": "k_slow_anchor_with_vol_and_atr"}) == plan_params_hash(base)


def test_normalized_lineage_params_only_includes_lineage_keys():
    config = {
        "universe_index": "csi1000",
        "top_n": 30,
        "trailing_stop_pct": 0.08,
        "monitor_can_trade": True,
    }
    assert normalized_lineage_params(config) == {"top_n": 30, "universe_index": "csi1000"}


def test_plan_params_hash_ignores_execution_and_pool_tuning_params():
    base = {
        "universe_index": "csi1000",
        "score_type": "growth_cycle",
        "top_n": 30,
        "rebalance_days": 20,
        "lot_size": 100,
        "transaction_cost": 0.001,
        "candidate_pool_multiplier": 3,
    }
    tuned = {**base, "lot_size": 200, "transaction_cost": 0.002, "candidate_pool_multiplier": 5}
    different_strategy = {**base, "top_n": 25}
    assert plan_params_hash(tuned) == plan_params_hash(base)
    assert plan_params_hash(different_strategy) != plan_params_hash(base)


def test_plan_params_hash_ignores_trailing_stop_pct():
    base = {
        "universe_index": "csi1000",
        "score_type": "growth_cycle",
        "top_n": 30,
        "rebalance_days": 20,
    }
    with_trailing = {**base, "trailing_stop_pct": 0.1}
    without_trailing = {**base, "trailing_stop_pct": None}
    assert plan_params_hash(with_trailing) == plan_params_hash(base)
    assert plan_params_hash(without_trailing) == plan_params_hash(base)


def test_effective_strategy_config_merges_preset_and_override():
    template = {
        "strategy_template_id": "growth_cycle_csi1000",
        "strategy_type": "template",
        "name": "Growth",
        "score_type": "growth_cycle",
        "version": 1,
    }
    preset = {"params": {"top_n": 20, "rebalance_days": 10}}
    config = effective_strategy_config(template, preset, {"top_n": 25})
    assert config["top_n"] == 25
    assert config["rebalance_days"] == 10
    assert config["strategy_template_id"] == "growth_cycle_csi1000"


def test_effective_strategy_config_honors_score_type_override():
    template = {
        "strategy_template_id": "growth_cycle_index_enhanced",
        "score_type": "growth_cycle_weighted",
    }
    config = effective_strategy_config(
        template,
        None,
        {"score_type": "score_column", "score_column": "model_score", "model_id": "m1"},
    )
    assert config["score_type"] == "score_column"
    assert config["model_id"] == "m1"
    assert normalized_lineage_params(config)["model_id"] == "m1"
    # No override keeps the template default.
    assert effective_strategy_config(template, None, None)["score_type"] == "growth_cycle_weighted"


def test_effective_strategy_config_default_reads_none_as_unset():
    """Settings-form semantic: a blank field must not wipe the preset's value."""
    template = {"strategy_template_id": "t1", "score_type": "growth_cycle"}
    preset = {"params": {"top_n": 20, "max_industry_weight": 0.2}}
    config = effective_strategy_config(template, preset, {"top_n": None})
    assert config["top_n"] == 20


def test_exact_override_restores_none_lineage_key():
    template = {"strategy_template_id": "t1", "score_type": "growth_cycle"}
    preset = {"params": {"top_n": 20, "max_industry_weight": 0.2}}
    config = effective_strategy_config(template, preset, {"max_industry_weight": None}, exact_override=True)
    assert config["max_industry_weight"] is None


def test_exact_override_leaves_non_lineage_none_unset():
    """The exact treatment is narrowed to lineage keys: a non-lineage None must
    still fall back, or it reaches consumers like ``int(config.get("lot_size"))``."""
    template = {"strategy_template_id": "t1", "score_type": "growth_cycle"}
    preset = {"params": {"top_n": 20, "lot_size": 100, "trailing_stop_pct": 0.08}}
    config = effective_strategy_config(
        template, preset, {"lot_size": None, "trailing_stop_pct": None}, exact_override=True
    )
    assert config["lot_size"] == 100
    assert config["trailing_stop_pct"] == 0.08


def test_exact_override_keeps_lineage_hash_across_preset_substitution():
    """Force-rebalance regression (8f73d17eb0ab -> 7cd2a2143a5f).

    A lineage whose ``max_industry_weight`` is legitimately ``None`` must keep its
    hash when the fallback preset is substituted in. Without ``exact_override``
    the None is dropped, the preset's value wins, and the plan is filed under a
    different lineage with no error anywhere.
    """
    template = {"strategy_template_id": "t1", "score_type": "growth_cycle"}
    original = {"params": {"top_n": 10, "universe_index": "csi1000", "max_industry_weight": None}}
    snapshot = normalized_plan_params(effective_strategy_config(template, original, None))
    expected = plan_params_hash(effective_strategy_config(template, original, None))

    fallback = {"params": {"top_n": 10, "universe_index": "csi1000", "max_industry_weight": 0.2}}

    assert plan_params_hash(effective_strategy_config(template, fallback, snapshot, exact_override=True)) == expected
    assert plan_params_hash(effective_strategy_config(template, fallback, snapshot)) != expected


def test_selected_preset_id_for_plan_requires_matching_snapshot():
    template = {"strategy_template_id": "t1", "score_type": "growth_cycle"}
    preset = {"preset_id": "p1", "params": {"top_n": 20, "universe_index": "csi1000"}}
    final = effective_strategy_config(template, preset, None)
    assert selected_preset_id_for_plan(template, preset, final) == "p1"
    tweaked = {**final, "lot_size": 100}
    assert selected_preset_id_for_plan(template, preset, tweaked) is None


def test_strategy_display_name_prefers_config_name():
    assert strategy_display_name({"name": "自定义"}, "t1") == "自定义"
    assert strategy_display_name({"score_type": "growth_cycle_v2"}, "other") == "成长周期指数增强"


def test_strategy_display_name_distinguishes_model_score_lineage():
    config = {
        "name": "成长周期指数增强",
        "score_type": "score_column",
        "score_column": "model_score",
        "model_id": "m1",
        "universe_index": "csi1000",
        "top_n": 20,
        "rebalance_days": 20,
    }
    assert strategy_display_name(config, "growth_cycle_index_enhanced") == "CSI1000 模型评分 Top20 20日"


def test_build_user_plan_id_includes_hash_and_user_suffix():
    plan_id = build_user_plan_id("growth_cycle_csi1000", "20240102", "user@example.com", "abcd1234efgh")
    assert plan_id.startswith("growth_cycle_csi1000_20240102_abcd1234_")
    assert build_user_plan_id("t1", "20240102", None, None) == "t1_20240102"
