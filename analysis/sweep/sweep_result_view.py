"""Sweep result view model: axes metadata, facet summaries, and stable combo keys."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from analysis.sweep.combo_detail import research_combo_key
from analysis.sweep.entry_stock_filter import normalize_entry_stock_filters
from analysis.sweep.params import (
    normalize_bull_exposures,
    normalize_crowd_exit_ks,
    normalize_industry_regime_scales,
    normalize_non_bull_actions,
    normalize_path_gates,
    normalize_refill_modes,
    normalize_regime_modes,
    normalize_regime_rules,
    normalize_trailing_stop_pcts,
    normalize_trailing_stop_ratio,
    resolve_regime_modes,
)

# Registry order is stable for UI/HTML column and combo-key suffix ordering.
_AXIS_REGISTRY: Tuple[Dict[str, Any], ...] = (
    {
        "key": "trailing_stop_pct",
        "label": "浮动止盈",
        "param_keys": ("trailing_stop_pcts", "trailing_stop_pct"),
        "row_key": "trailing_stop_pct",
    },
    {
        "key": "regime_mode",
        "label": "指数Regime",
        "param_keys": ("regime_modes", "regime_cash"),
        "row_key": "regime_mode",
    },
    {
        "key": "regime_rule",
        "label": "择时规则",
        "param_keys": ("regime_rules", "regime_rule"),
        "row_key": "regime_rule",
    },
    {
        "key": "bull_exposure",
        "label": "牛段仓位",
        "param_keys": ("bull_exposures", "bull_exposure"),
        "row_key": "bull_exposure",
    },
    {
        "key": "intra_period_refill",
        "label": "止盈补位",
        "param_keys": ("refill_modes", "intra_period_refill"),
        "row_key": "intra_period_refill",
    },
    {
        "key": "crowd_exit_k",
        "label": "拥挤退出",
        "param_keys": ("crowd_exit_ks", "crowd_exit_k"),
        "row_key": "crowd_exit_k",
    },
    {
        "key": "non_bull_action",
        "label": "非牛仓位",
        "param_keys": ("non_bull_actions", "non_bull_action"),
        "row_key": "non_bull_action",
    },
    {
        "key": "industry_regime_scale",
        "label": "行业Regime仓位",
        "param_keys": ("industry_regime_scales", "industry_regime_scale"),
        "row_key": "industry_regime_scale",
    },
    {
        "key": "score_variant",
        "label": "Score",
        "param_keys": ("score_specs", "growth_cycle_weights"),
        "row_key": "score_variant",
    },
    {
        "key": "top_n",
        "label": "Top N",
        "param_keys": ("top_n_values",),
        "row_key": "top_n",
    },
    {
        "key": "max_industry_weight",
        "label": "行业上限",
        "param_keys": ("active_caps",),
        "row_key": "max_industry_weight",
    },
    {
        "key": "rebalance_interval_days",
        "label": "调仓间隔",
        "param_keys": ("rebalance_interval_days",),
        "row_key": "rebalance_interval_days",
    },
    {
        "key": "volatility_variant",
        "label": "波动率",
        "param_keys": ("volatility_modes", "target_vols"),
        "row_key": "volatility_variant",
    },
    {
        "key": "entry_stock_filter",
        "label": "个股入场过滤",
        "param_keys": ("entry_stock_filters", "entry_stock_filter"),
        "row_key": "entry_stock_filter",
    },
)

METRIC_KEYS: Tuple[str, ...] = (
    "cumulative_return",
    "annualized_return",
    "index_excess_cumulative_return",
    "sharpe",
    "max_drawdown",
    "average_turnover",
    "average_exposure",
    "avg_active_share",
    "avg_max_abs_active_weight",
    "risk_adjusted_score",
)

_IDENTITY_ROW_KEYS: Tuple[str, ...] = (
    "score_variant",
    "variant",
    "top_n",
    "rebalance_interval_days",
)


def _safe_slug(value: Any) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "-", str(value)).strip("-") or "na"


def normalize_axis_value(key: str, value: Any) -> Any:
    if key == "trailing_stop_pct":
        return normalize_trailing_stop_ratio(value)
    if key in ("bull_exposure", "industry_regime_scale"):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    if key == "intra_period_refill":
        if isinstance(value, bool):
            return value
        try:
            return normalize_refill_modes([value])[0]
        except ValueError:
            return value
    if key == "crowd_exit_k":
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if key in ("top_n", "rebalance_interval_days"):
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if key == "max_industry_weight":
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return value


def industry_cap_applies(row: Optional[Dict[str, Any]] = None) -> bool:
    """True when ``max_industry_weight`` actually constrains selection.

    ``top_n`` / ``industry_neutral`` still persist a default 0.3 on the row
    (and in combo_key) as an axis placeholder. That number must not be shown
    as a real 30% cap. Passing ``row=None`` keeps the raw numeric label so
    combo_key slugs stay stable.
    """
    if not row:
        return True
    mode = str(row.get("construction_mode") or "").strip().lower()
    if mode == "industry_capped":
        return True
    if mode in {"top_n", "industry_neutral"}:
        return False
    variant = str(row.get("variant") or "").lower()
    if "active_cap" in variant:
        return True
    if "topn" in variant or "neutral" in variant:
        return False
    return True


def format_axis_value(key: str, value: Any, row: Optional[Dict[str, Any]] = None) -> str:
    normalized = normalize_axis_value(key, value)
    if key == "trailing_stop_pct":
        if normalized is None:
            return "关闭"
        return f"{normalized * 100:g}%"
    if key == "bull_exposure" and normalized is not None:
        try:
            return f"{float(normalized) * 100:g}%"
        except (TypeError, ValueError):
            return str(normalized)
    if key == "intra_period_refill":
        return "补位" if normalized else "关闭"
    if key == "crowd_exit_k":
        if normalized in (None, 0, "0"):
            return "关闭"
        return f"K={int(normalized)}"
    if key == "non_bull_action":
        labels = {
            "cash": "非牛空仓",
            "half_topn": "非牛Top减半满仓",
            "half_topn_slot": "非牛Top减半留现金",
        }
        return labels.get(str(normalized or "cash"), str(normalized))
    if key == "industry_regime_scale" and normalized is not None:
        try:
            factor = float(normalized)
        except (TypeError, ValueError):
            return str(normalized)
        if abs(factor - 1.0) <= 1e-12:
            return "关闭"
        if abs(factor) <= 1e-12:
            return "非牛行业清仓"
        return f"非牛行业×{factor:g}"
    if key == "max_industry_weight":
        if not industry_cap_applies(row):
            return "无上限"
        if normalized is not None:
            return f"{float(normalized) * 100:g}%"
    if key == "rebalance_interval_days" and normalized is not None:
        return f"{int(normalized)}d"
    if key == "regime_mode":
        if normalized in (None, "", "off"):
            return "关闭"
        if normalized == "bull_g60_else_cash":
            return "牛市60:40/其余空仓"
        return str(normalized)
    if key == "entry_stock_filter":
        if normalized in (None, "", "off"):
            return "关闭"
        labels = {
            "distrib03": "distrib≥0.03",
            "distrib05": "distrib≥0.05",
            "distrib08": "distrib≥0.08",
            "hipos95": "hipos≥0.95",
            "hipos97": "hipos≥0.97",
            "hipos97_distrib": "hipos97+distrib05",
        }
        text = str(normalized or "").strip().lower()
        return labels.get(text, text or "-")
    if key == "regime_rule":
        labels = {
            "k_slow_anchor": "价格K交叉",
            "k_slow_anchor_with_vol_and_atr": "量价择时",
            "k_slow_anchor_with_volume": "仅放量过滤",
            "k_slow_anchor_with_volatility": "仅低波动过滤",
        }
        text = str(normalized or "").strip()
        return labels.get(text, text or "-")
    if normalized is None or normalized == "":
        return "-"
    return str(normalized)


def _axis_def(key: str) -> Optional[Dict[str, Any]]:
    for item in _AXIS_REGISTRY:
        if item["key"] == key:
            return item
    return None


def _values_from_params(params: Dict[str, Any], axis_key: str) -> Optional[List[Any]]:
    definition = _axis_def(axis_key)
    if definition is None:
        return None
    for param_key in definition["param_keys"]:
        raw = params.get(param_key)
        if raw is None:
            continue
        if param_key == "trailing_stop_pcts":
            return normalize_trailing_stop_pcts(trailing_stop_pcts=list(raw))
        if param_key == "trailing_stop_pct":
            return normalize_trailing_stop_pcts(trailing_stop_pct=raw)
        if param_key == "path_gates":
            return normalize_path_gates(list(raw))
        if param_key == "entry_stock_filters":
            return normalize_entry_stock_filters(list(raw))
        if param_key == "entry_stock_filter":
            return normalize_entry_stock_filters([raw])
        if param_key == "regime_modes":
            return normalize_regime_modes(list(raw))
        if param_key == "regime_cash":
            if raw:
                return resolve_regime_modes(regime_cash=raw)
            continue
        if param_key == "regime_rules":
            values = normalize_regime_rules(list(raw))
            return values if len(values) > 1 else None
        if param_key == "regime_rule":
            values = normalize_regime_rules(None, regime_rule=raw)
            return values if len(values) > 1 else None
        if param_key == "bull_exposures":
            values = normalize_bull_exposures(list(raw) if not isinstance(raw, (str, bytes)) else [raw])
            return values if len(values) > 1 else None
        if param_key == "bull_exposure":
            values = normalize_bull_exposures([raw])
            return values if len(values) > 1 else None
        if param_key == "refill_modes":
            values = normalize_refill_modes(list(raw) if not isinstance(raw, (str, bytes)) else [raw])
            return values if len(values) > 1 else None
        if param_key == "intra_period_refill":
            values = normalize_refill_modes([raw])
            return values if len(values) > 1 else None
        if param_key == "crowd_exit_ks":
            values = normalize_crowd_exit_ks(list(raw) if not isinstance(raw, (str, bytes)) else [raw])
            return values if len(values) > 1 else None
        if param_key == "crowd_exit_k":
            values = normalize_crowd_exit_ks([raw])
            return values if len(values) > 1 else None
        if param_key == "non_bull_actions":
            values = normalize_non_bull_actions(
                list(raw) if not isinstance(raw, (str, bytes)) else [raw]
            )
            return values if len(values) > 1 else None
        if param_key == "non_bull_action":
            values = normalize_non_bull_actions([raw])
            return values if len(values) > 1 else None
        if param_key == "industry_regime_scales":
            values = normalize_industry_regime_scales(
                list(raw) if not isinstance(raw, (str, bytes)) else [raw]
            )
            return values if len(values) > 1 else None
        if param_key == "industry_regime_scale":
            values = normalize_industry_regime_scales([raw])
            return values if len(values) > 1 else None
        if isinstance(raw, (list, tuple)):
            if len(raw) > 1:
                return [normalize_axis_value(axis_key, item) for item in raw]
            return None
    return None


def _values_from_rows(rows: Sequence[Dict[str, Any]], axis_key: str) -> Optional[List[Any]]:
    definition = _axis_def(axis_key)
    if definition is None:
        return None
    row_key = definition["row_key"]
    seen: List[Any] = []
    for row in rows:
        value = normalize_axis_value(axis_key, row.get(row_key))
        if axis_key == "regime_rule" and value in (None, "", "-"):
            continue
        if value not in seen:
            seen.append(value)
    if len(seen) <= 1:
        return None
    return seen


def infer_sweep_axes(
    params: Optional[Dict[str, Any]] = None,
    rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Infer swept parameter axes in stable registry order."""
    params = params or {}
    rows = list(rows or [])
    axes: List[Dict[str, Any]] = []
    for definition in _AXIS_REGISTRY:
        key = definition["key"]
        # score_specs contain config objects, while result rows contain the
        # canonical generated score names used by facet matching.
        values = _values_from_rows(rows, key) if key == "score_variant" and rows else None
        if values is None:
            values = _values_from_params(params, key)
        if values is None and rows:
            values = _values_from_rows(rows, key)
        if not values or len(values) <= 1:
            continue
        axes.append(
            {
                "key": key,
                "label": definition["label"],
                "values": values,
            }
        )
    return axes


def _row_matches_axis(row: Dict[str, Any], axis_key: str, axis_value: Any) -> bool:
    definition = _axis_def(axis_key)
    if definition is None:
        return False
    row_value = normalize_axis_value(axis_key, row.get(definition["row_key"]))
    target = normalize_axis_value(axis_key, axis_value)
    return row_value == target


def _risk_score(row: Dict[str, Any]) -> float:
    try:
        return float(row.get("risk_adjusted_score") or -1e18)
    except (TypeError, ValueError):
        return -1e18


def compute_facet_best(
    rows: Sequence[Dict[str, Any]],
    sweep_axes: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Best row per value for each swept axis (by global risk_adjusted_score)."""
    facet_best: Dict[str, List[Dict[str, Any]]] = {}
    ordered_rows = sorted(rows, key=_risk_score, reverse=True)
    for axis in sweep_axes:
        axis_key = axis["key"]
        bucket: List[Dict[str, Any]] = []
        for axis_value in axis["values"]:
            match = next((row for row in ordered_rows if _row_matches_axis(row, axis_key, axis_value)), None)
            if match is not None:
                bucket.append(
                    {
                        "value": axis_value,
                        "label": format_axis_value(axis_key, axis_value, match),
                        "row": match,
                    }
                )
        if bucket:
            facet_best[axis_key] = bucket
    return facet_best


def infer_fixed_params(params: Dict[str, Any], sweep_axes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    swept_param_keys = set()
    for axis in sweep_axes:
        definition = _axis_def(axis["key"])
        if definition:
            swept_param_keys.update(definition["param_keys"])
    fixed = dict(params)
    for key in swept_param_keys:
        fixed.pop(key, None)
    return fixed


def combo_key_from_dimensions(
    row: Dict[str, Any],
    sweep_axes: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    """Stable combo detail key from row dimensions (all swept axes included)."""
    rebalance = row.get("rebalance_interval_days")
    key = research_combo_key(
        str(row.get("score_variant") or row.get("score_column") or "score"),
        str(row.get("variant") or "variant"),
        int(row.get("top_n") or 0),
        int(rebalance) if rebalance is not None else None,
    )
    axes = list(sweep_axes or infer_sweep_axes(rows=[row]))
    for axis in axes:
        axis_key = axis["key"]
        if axis_key in _IDENTITY_ROW_KEYS:
            continue
        row_key = (_axis_def(axis_key) or {}).get("row_key", axis_key)
        normalized = normalize_axis_value(axis_key, row.get(row_key))
        slug = _safe_slug(format_axis_value(axis_key, normalized))
        key = f"{key}__{axis_key}={slug}"
    return key


def stability_group_columns(sweep_axes: Sequence[Dict[str, Any]]) -> List[str]:
    columns = [axis["key"] for axis in sweep_axes]
    for extra in ("score_variant", "variant"):
        if extra not in columns:
            columns.append(extra)
    return columns


def build_sweep_result_view(
    payload: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build sweep_view metadata consumed by HTML report and Dashboard."""
    params = params or {}
    rows = list(payload.get("rows") or [])
    sweep_axes = infer_sweep_axes(params, rows)
    facet_best = compute_facet_best(rows, sweep_axes)
    # best_row in payload is the global best (highest risk_adjusted_score).
    global_best = rows[0] if rows else None
    return {
        "sweep_axes": sweep_axes,
        "fixed_params": infer_fixed_params(params, sweep_axes),
        "metric_keys": list(METRIC_KEYS),
        "row_count": len(rows),
        "facet_best": facet_best,
        "global_best_row": global_best,
    }


def attach_sweep_view(payload: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = dict(payload)
    payload["sweep_view"] = build_sweep_result_view(payload, params)
    return payload
