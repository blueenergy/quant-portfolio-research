"""Score-column construction for sweep datasets.

Turns score specs and growth/cycle weights into the blended columns a
sweep runs over. Lives below ``portfolio_sweep_report`` so combo details
can build the very same columns without importing the entry point.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from analysis.core.score_spec import (
    ParsedScoreSpec,
    canonical_score_weights,
    parse_score_specs,
    weighted_score_name,
)


@dataclass(frozen=True)
class ScoreSweep:
    name: str
    score_column: str
    growth_weight: Optional[float] = None
    cycle_weight: Optional[float] = None
    weights: Optional[Dict[str, float]] = None
    legacy_growth_cycle: bool = False
    # Blend per-date cross-sectional percentile ranks instead of raw scores.
    rank_based: bool = False


def parse_growth_cycle_weights(values: Optional[Sequence[str]]) -> List[ScoreSweep]:
    if not values:
        return []
    sweeps: List[ScoreSweep] = []
    seen = set()
    for raw_value in values:
        growth_weight, cycle_weight = _parse_growth_cycle_weight(raw_value)
        label = _growth_cycle_weight_label(growth_weight, cycle_weight)
        if label in seen:
            continue
        seen.add(label)
        sweeps.append(
            ScoreSweep(
                name=label,
                score_column=f"{label}_score",
                growth_weight=growth_weight,
                cycle_weight=cycle_weight,
                weights={"cycle": cycle_weight, "growth": growth_weight},
                legacy_growth_cycle=True,
            )
        )
    return sweeps


def _score_sweep_from_spec(spec: ParsedScoreSpec) -> ScoreSweep:
    if spec.mode == "column":
        assert spec.column is not None
        return ScoreSweep(name=spec.column, score_column=spec.column)
    weights = canonical_score_weights(spec.weights or {})
    rank_based = spec.mode == "rank_weighted"
    name = weighted_score_name(weights, rank_based=rank_based)
    return ScoreSweep(
        name=name,
        score_column=f"{name}_score",
        weights=weights,
        rank_based=rank_based,
    )


def build_score_sweeps(
    score_column: str,
    growth_cycle_weights: Optional[Sequence[str]] = None,
    score_specs: Optional[Sequence[Any]] = None,
) -> List[ScoreSweep]:
    if score_specs and growth_cycle_weights:
        raise ValueError("score_specs and growth_cycle_weights are mutually exclusive; provide only one")
    specs = parse_score_specs(score_specs)
    if specs:
        return [_score_sweep_from_spec(spec) for spec in specs]
    parsed = parse_growth_cycle_weights(growth_cycle_weights)
    if parsed:
        return parsed
    return [ScoreSweep(name=score_column, score_column=score_column)]


def add_weighted_score_columns(dataset: pd.DataFrame, score_sweeps: Sequence[ScoreSweep]) -> pd.DataFrame:
    weighted_sweeps = [
        sweep
        for sweep in score_sweeps
        if sweep.weights or (sweep.growth_weight is not None and sweep.cycle_weight is not None)
    ]
    if not weighted_sweeps:
        return dataset

    required = sorted(
        {
            f"{dimension}_score"
            for sweep in weighted_sweeps
            for dimension in (
                sweep.weights
                or {"growth": sweep.growth_weight, "cycle": sweep.cycle_weight}
            )
        }
    )
    missing = [column for column in required if column not in dataset.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns for score weights: {missing}")

    df = dataset.copy()
    for sweep in weighted_sweeps:
        weights = sweep.weights or canonical_score_weights(
            {"growth": sweep.growth_weight, "cycle": sweep.cycle_weight}
        )
        available_weight = pd.Series(0.0, index=df.index)
        weighted_sum = pd.Series(0.0, index=df.index)
        for dimension, weight in weights.items():
            values = pd.to_numeric(df[f"{dimension}_score"], errors="coerce")
            if sweep.rank_based:
                # Rank within each cross-section so a dimension's influence comes
                # from its weight rather than from its raw dispersion.
                values = values.groupby(df["score_date"]).rank(pct=True)
            available_weight = available_weight + values.notna().astype(float) * float(weight)
            weighted_sum = weighted_sum + values.fillna(0) * float(weight)
        df[sweep.score_column] = weighted_sum.where(available_weight > 0) / available_weight.where(
            available_weight > 0
        )
    return df


def add_growth_cycle_weight_columns(
    dataset: pd.DataFrame,
    score_sweeps: Sequence[ScoreSweep],
) -> pd.DataFrame:
    """Backward-compatible wrapper for callers using the old function name."""
    return add_weighted_score_columns(dataset, score_sweeps)


def overlay_cycle_score(dataset: pd.DataFrame, source_path: str, verbose: bool = False) -> pd.DataFrame:
    """用外部预算的 cycle 分按 (symbol, score_date) 覆盖内存 dataset 的 ``cycle_score`` 列。

    仅用于 A/B 对照不同 cycle 口径（如实验版 cycle_v2）：不写库、不改打分链路，
    只在本次 sweep 内把参与 growth/cycle 混合的 cycle_score 整列替换为外部来源。
    源文件需含列 symbol / score_date / cycle_score，支持 .parquet 与 .csv[.gz]。
    未命中的行 cycle_score 置为 NaN，混合时按 available_weight 退化为仅 growth。
    """
    if dataset.empty:
        return dataset
    path = Path(source_path)
    if not path.exists():
        raise SystemExit(f"--cycle-score-source 文件不存在: {path}")
    if path.suffix == ".parquet":
        ext = pd.read_parquet(path)
    else:
        ext = pd.read_csv(path)
    need = {"symbol", "score_date", "cycle_score"}
    missing = need - set(ext.columns)
    if missing:
        raise SystemExit(f"--cycle-score-source 缺少列: {sorted(missing)}")
    ext = ext[["symbol", "score_date", "cycle_score"]].copy()
    ext["symbol"] = ext["symbol"].astype(str)
    ext["score_date"] = ext["score_date"].astype(str)
    ext["cycle_score"] = pd.to_numeric(ext["cycle_score"], errors="coerce")
    ext = ext.dropna(subset=["cycle_score"]).drop_duplicates(["symbol", "score_date"], keep="last")
    ext = ext.rename(columns={"cycle_score": "_cycle_override"})

    df = dataset.copy()
    if "cycle_score" not in df.columns:
        df["cycle_score"] = pd.NA
    df["symbol"] = df["symbol"].astype(str)
    df["score_date"] = df["score_date"].astype(str)
    merged = df.merge(ext, on=["symbol", "score_date"], how="left")
    matched = int(merged["_cycle_override"].notna().sum())
    merged["cycle_score"] = merged["_cycle_override"]
    merged = merged.drop(columns=["_cycle_override"])
    if verbose:
        total = len(merged)
        cov = matched / total if total else 0.0
        print(f"cycle_score 覆盖: {matched}/{total} ({cov:.1%}) 来自 {path.name}", flush=True)
    return merged


def _parse_growth_cycle_weight(value: str) -> tuple:
    text = str(value).strip()
    if not text:
        raise ValueError("growth-cycle weight cannot be empty")
    if ":" in text:
        left, right = text.split(":", 1)
        growth_weight = float(left)
        cycle_weight = float(right)
    else:
        growth_weight = float(text)
        if growth_weight > 1:
            growth_weight = growth_weight / 100
        cycle_weight = 1 - growth_weight
    if growth_weight < 0 or cycle_weight < 0:
        raise ValueError(f"growth-cycle weights must be non-negative: {value}")
    total = growth_weight + cycle_weight
    if total <= 0:
        raise ValueError(f"growth-cycle weights must have positive total weight: {value}")
    return growth_weight / total, cycle_weight / total


def _growth_cycle_weight_label(growth_weight: float, cycle_weight: float) -> str:
    return f"growth_cycle_g{int(round(growth_weight * 100))}_c{int(round(cycle_weight * 100))}"
