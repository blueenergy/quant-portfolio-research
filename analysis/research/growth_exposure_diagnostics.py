#!/usr/bin/env python3
"""Stage-0 growth diagnostics — does any raw exposure have a *shape*?

Consumes the shards written by ``growth_exposure_extract.py`` and answers, before
any modelling, the only questions a model could exploit:

  1. Coverage — which exposures even have data, per year.
  2. IC table — raw vs industry-neutral Rank IC per exposure × horizon, with the
     non-overlap t the 2026-10-04 study established as the only readable t.
  3. Decile shape — mean forward return by cross-sectional decile (date-equal
     weighted). Monotone? Humped? Flat?
  4. Quadratic term — per-date OLS ``ret ~ a + b1*z + b2*z^2``; a significant b2
     is the one "curve" nonlinearity a model could add over the current linear
     composite.
  5. Conditional IC — by regime and by year, because growth's tail spread is
     known to be bull-only.

Gate-0 verdict at the bottom: if nothing shows shape at any horizon, no ML model
can help; if something does, that is what Stage 1 should fit.

Run inside the quant-scorer image:

    python analysis/research/growth_exposure_diagnostics.py \
        --shard-dir /app/logs/growth_stage0 --horizons 5 20 60 120 \
        --universe csi1000 --output /app/logs/growth_stage0_report.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

RAW_EXPOSURES: Tuple[str, ...] = (
    "rev_yoy",
    "rev_cagr",
    "rev_yoy_std",
    "profit_yoy",
    "profit_positive_ratio",
    "roe_level",
    "roe_trend",
    "rd_ratio",
)
# direction: +1 = higher is more growth. Kept in sync with GrowthV2Config.
DIRECTION = {"rev_yoy_std": -1}
DECILES = 10
MIN_GROUP = 5


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_shards(shard_dir: str, pattern: str = "exposures_*.csv") -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(shard_dir) / pattern)))
    if not files:
        raise SystemExit(f"no shards under {shard_dir}")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["score_date"] = df["score_date"].astype(str).str.replace("-", "", regex=False)
    df["year"] = df["score_date"].str[:4]
    return df


# ---------------------------------------------------------------------------
# small stats helpers (shared conventions with score_ic_by_year_regime)
# ---------------------------------------------------------------------------


def nonoverlap_t(series: pd.Series, horizon: int, sampling_days: int = 21) -> Optional[float]:
    """t of the mean, subsampled to kill return overlap.

    ``stride`` counts *observations*, not trading days: the panel is monthly
    (``sampling_days`` ~= 21 trading days per observation), so h=20 barely
    overlaps and needs no thinning, while h=60 spans 3 observations and must be
    thinned 3x. The daily-panel convention (stride = h) would leave 4 points at
    h=20 here and make every t meaningless.
    """
    stride = max(1, int(round(horizon / max(1, sampling_days))))
    ordered = series.dropna().sort_index()
    if stride > 1:
        ordered = ordered.iloc[::stride]
    n = len(ordered)
    if n < 3:
        return None
    std = ordered.std(ddof=1)
    if not std or not np.isfinite(std):
        return None
    return float(ordered.mean() / std * math.sqrt(n))


def per_date_rank_ic(df: pd.DataFrame, score_col: str, ret_col: str, min_names: int = 20) -> pd.Series:
    sub = df[["score_date", score_col, ret_col]].dropna()
    if sub.empty:
        return pd.Series(dtype=float)
    out = {}
    for date, group in sub.groupby("score_date", sort=True):
        if len(group) < min_names or group[score_col].nunique() < 2 or group[ret_col].nunique() < 2:
            continue
        ic = group[score_col].rank().corr(group[ret_col].rank())
        if pd.notna(ic):
            out[date] = float(ic)
    return pd.Series(out).sort_index()


def add_industry_neutral(df: pd.DataFrame, ret_col: str) -> str:
    """Forward return demeaned within (date, industry) — the vault §3 test."""
    neutral_col = f"{ret_col}__ind_neu"
    industry = df["industry"].fillna("__UNKNOWN__").replace("", "__UNKNOWN__")
    group_mean = df.groupby([df["score_date"], industry])[ret_col].transform("mean")
    df[neutral_col] = df[ret_col] - group_mean
    return neutral_col


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def coverage_block(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for col in RAW_EXPOSURES:
        if col not in df.columns:
            continue
        per_year = df.groupby("year")[col].apply(lambda s: float(s.notna().mean()))
        out[col] = {
            "nonnull_overall": float(df[col].notna().mean()),
            "nonnull_by_year": {y: round(v, 4) for y, v in per_year.items()},
        }
    return out


def ic_block(df: pd.DataFrame, columns: Sequence[str], horizons: Sequence[int], sampling_days: int = 21) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    min_names = max(DECILES, 20)
    for col in columns:
        if col not in df.columns:
            continue
        per_horizon: Dict[str, Any] = {}
        for h in horizons:
            ret_col = f"forward_return_{h}d"
            if ret_col not in df.columns:
                continue
            neutral_col = add_industry_neutral(df, ret_col)
            raw_ic = per_date_rank_ic(df, col, ret_col, min_names)
            neu_ic = per_date_rank_ic(df, col, neutral_col, min_names)
            per_horizon[str(h)] = {
                "raw_ic_mean": float(raw_ic.mean()) if len(raw_ic) else None,
                "raw_ic_ir": float(raw_ic.mean() / raw_ic.std(ddof=1)) if len(raw_ic) > 1 and raw_ic.std(ddof=1) else None,
                "raw_nonoverlap_t": nonoverlap_t(raw_ic, h, sampling_days),
                "ind_neutral_ic_mean": float(neu_ic.mean()) if len(neu_ic) else None,
                "ind_neutral_nonoverlap_t": nonoverlap_t(neu_ic, h, sampling_days),
                "n_dates": int(len(raw_ic)),
            }
        out[col] = per_horizon
    return out


def decile_block(
    df: pd.DataFrame, columns: Sequence[str], horizon: int, sampling_days: int = 21
) -> Dict[str, Any]:
    """Mean forward return by cross-sectional decile, date-equal weighted."""
    ret_col = f"forward_return_{horizon}d"
    if ret_col not in df.columns:
        return {}
    out: Dict[str, Any] = {}
    for col in columns:
        if col not in df.columns:
            continue
        sub = df[["score_date", col, ret_col]].dropna()
        if sub["score_date"].nunique() < 5:
            continue
        decile = sub.groupby("score_date")[col].transform(
            lambda s: pd.qcut(s.rank(method="first"), DECILES, labels=False) + 1
            if s.nunique() >= DECILES
            else np.nan
        )
        sub = sub.assign(decile=decile).dropna(subset=["decile"])
        if sub.empty:
            continue
        per_date = sub.groupby(["score_date", "decile"])[ret_col].mean().unstack()
        means = per_date.mean(axis=0)
        top, bottom = means.get(DECILES), means.get(1)
        spread = float(top - bottom) if top is not None and bottom is not None else None
        spread_series = (per_date[DECILES] - per_date[1]).dropna() if DECILES in per_date and 1 in per_date else pd.Series(dtype=float)
        values = [float(means[q]) for q in sorted(means.index) if pd.notna(means.get(q))]
        monotone = all(a < b for a, b in zip(values, values[1:])) if len(values) >= 2 else None
        peak = int(means.idxmax()) if len(means) else None
        out[col] = {
            "mean_by_decile": {int(q): round(float(means[q]), 5) for q in means.index},
            "top_minus_bottom": spread,
            "top_minus_bottom_nonoverlap_t": nonoverlap_t(spread_series, horizon, sampling_days),
            "monotone_increasing": monotone,
            "peak_decile": peak,
            "peak_is_interior": (peak is not None and 2 <= peak <= DECILES - 1),
            "n_dates": int(per_date.shape[0]),
        }
    return out


def quadratic_block(df: pd.DataFrame, columns: Sequence[str], horizon: int, sampling_days: int = 21) -> Dict[str, Any]:
    """Per-date OLS ret ~ a + b1*z + b2*z^2, on cross-sectionally standardized z."""
    ret_col = f"forward_return_{horizon}d"
    if ret_col not in df.columns:
        return {}
    out: Dict[str, Any] = {}
    for col in columns:
        if col not in df.columns:
            continue
        sub = df[["score_date", col, ret_col]].dropna()
        b1s, b2s = {}, {}
        for date, group in sub.groupby("score_date", sort=True):
            if len(group) < 40 or group[col].nunique() < 20:
                continue
            z = group[col].to_numpy(dtype=float)
            z = (z - z.mean()) / (z.std() or 1.0)
            y = group[ret_col].to_numpy(dtype=float)
            design = np.column_stack([np.ones_like(z), z, z * z])
            try:
                coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            b1s[date] = float(coef[1])
            b2s[date] = float(coef[2])
        b1 = pd.Series(b1s).sort_index()
        b2 = pd.Series(b2s).sort_index()
        out[col] = {
            "b1_mean": float(b1.mean()) if len(b1) else None,
            "b1_nonoverlap_t": nonoverlap_t(b1, horizon, sampling_days),
            "b2_mean": float(b2.mean()) if len(b2) else None,
            "b2_nonoverlap_t": nonoverlap_t(b2, horizon, sampling_days),
            "n_dates": int(len(b2)),
        }
    return out


def conditional_ic(
    df: pd.DataFrame,
    columns: Sequence[str],
    horizon: int,
    regime_by_date: Dict[str, str],
    sampling_days: int = 21,
) -> Dict[str, Any]:
    ret_col = f"forward_return_{horizon}d"
    if ret_col not in df.columns:
        return {}
    frame = df.copy()
    frame["regime"] = frame["score_date"].map(regime_by_date).fillna("uncertain")
    out: Dict[str, Any] = {}
    for col in columns:
        if col not in frame.columns:
            continue
        entry: Dict[str, Any] = {}
        for label, group in frame.groupby("regime"):
            ic = per_date_rank_ic(group, col, ret_col)
            entry[f"regime_{label}"] = {
                "ic_mean": float(ic.mean()) if len(ic) else None,
                "nonoverlap_t": nonoverlap_t(ic, horizon, sampling_days),
                "n_dates": int(len(ic)),
            }
        per_year = {}
        for year, group in frame.groupby("year"):
            ic = per_date_rank_ic(group, col, ret_col)
            per_year[year] = {
                "ic_mean": float(ic.mean()) if len(ic) else None,
                "n_dates": int(len(ic)),
            }
        entry["by_year"] = per_year
        out[col] = entry
    return out


def redundancy_block(df: pd.DataFrame) -> Dict[str, Any]:
    z_cols = [f"z_{c}" for c in RAW_EXPOSURES if f"z_{c}" in df.columns]
    if len(z_cols) < 2:
        return {}
    corr = df[z_cols].corr(method="spearman")
    return {
        "columns": z_cols,
        "spearman": {c: {c2: round(float(corr.at[c, c2]), 3) for c2 in z_cols} for c in z_cols},
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def print_ic_table(ic: Dict[str, Any], columns: Sequence[str], horizons: Sequence[int]) -> None:
    print("\n==== IC：原始 vs 行业中性（非重叠 t；|t|<2 一律读作不可辨） ====")
    header = f"{'column':<24}" + "".join(f"{'h' + str(h):>26}" for h in horizons)
    print(header)
    for col in columns:
        if col not in ic:
            continue
        cells = []
        for h in horizons:
            block = ic[col].get(str(h))
            if not block:
                cells.append(f"{'n/a':>26}")
                continue
            cells.append(
                f"{fmt(block['raw_ic_mean'], 4):>9}/{fmt(block['raw_nonoverlap_t'], 2):>5}"
                f" n={fmt(block['ind_neutral_ic_mean'], 4):>9}/{fmt(block['ind_neutral_nonoverlap_t'], 2):>5}"
            )
        print(f"{col:<24}" + "".join(f"{c:>26}" for c in cells))
    print("  (单元格式 = 原始 IC/非重叠 t  行业中性 IC/非重叠 t)")


def print_decile_table(deciles: Dict[str, Any], horizon: int) -> None:
    print(f"\n==== 分位形状 (h={horizon}d，按日等权，D1 最低 D10 最高) ====")
    print(f"{'column':<24}" + "".join(f"{'D' + str(q):>9}" for q in range(1, DECILES + 1)) + f"{'D10-D1':>10}{'t':>7}{'peak':>6}{'mono':>6}")
    for col, block in deciles.items():
        row = "".join(f"{block['mean_by_decile'].get(q, float('nan')) * 100:>8.2f}%" for q in range(1, DECILES + 1))
        print(
            f"{col:<24}{row}"
            f"{(block['top_minus_bottom'] or 0) * 100:>9.2f}%"
            f"{fmt(block['top_minus_bottom_nonoverlap_t'], 2):>7}"
            f"{str(block['peak_decile']):>6}{str(block['monotone_increasing']):>6}"
        )


def print_quadratic_table(quad: Dict[str, Any], horizon: int) -> None:
    print(f"\n==== 二次项 (h={horizon}d)：ret ~ a + b1*z + b2*z^2 ====")
    print(f"{'column':<24}{'b1':>10}{'t(b1)':>8}{'b2':>10}{'t(b2)':>8}   形状")
    for col, block in quad.items():
        b1t, b2t = block["b1_nonoverlap_t"], block["b2_nonoverlap_t"]
        if b2t is not None and b2t < -2:
            shape = "倒U(凸→凹)"
        elif b2t is not None and b2t > 2:
            shape = "U形"
        elif b1t is not None and abs(b1t) >= 2:
            shape = "单调"
        else:
            shape = "平"
        print(
            f"{col:<24}{fmt(block['b1_mean'], 5):>10}{fmt(b1t, 2):>8}"
            f"{fmt(block['b2_mean'], 5):>10}{fmt(b2t, 2):>8}   {shape}"
        )


def print_conditional(cond: Dict[str, Any], horizon: int) -> None:
    print(f"\n==== 条件 IC (h={horizon}d) ====")
    for col, block in cond.items():
        parts = []
        for key, value in block.items():
            if key.startswith("regime_"):
                parts.append(f"{key[7:]} IC={fmt(value['ic_mean'], 4)} t={fmt(value['nonoverlap_t'], 2)}")
        print(f"{col:<24}" + " | ".join(parts))
        years = block.get("by_year", {})
        print(f"{'':<24}" + "  ".join(f"{y}:{fmt(v['ic_mean'], 3)}" for y, v in sorted(years.items())))


def verdict(ic: Dict[str, Any], quad: Dict[str, Any], horizons: Sequence[int]) -> Dict[str, Any]:
    """Gate-0: does any exposure show a shape worth modelling?

    ``quad`` is keyed by horizon-string, each holding the quadratic block for
    every z column. Significant non-overlap |t| >= 2 on IC, the linear term, or
    the quadratic term counts as "a shape exists here".
    """
    findings: List[str] = []
    for col in RAW_EXPOSURES:
        z_col = f"z_{col}"
        signals: List[str] = []
        for h in horizons:
            block = (ic.get(z_col) or {}).get(str(h))
            if block:
                t = block.get("raw_nonoverlap_t")
                if t is not None and abs(t) >= 2:
                    signals.append(f"h{h} IC t={t:.2f}")
            q = (quad.get(str(h)) or {}).get(z_col)
            if q:
                t1, t2 = q.get("b1_nonoverlap_t"), q.get("b2_nonoverlap_t")
                if t2 is not None and abs(t2) >= 2:
                    signals.append(f"h{h} 二次项 t={t2:.2f}")
                elif t1 is not None and abs(t1) >= 2:
                    signals.append(f"h{h} 线性 t={t1:.2f}")
        if signals:
            findings.append(f"{col}: " + "; ".join(signals))
    return {
        "shaped_exposures": findings,
        "gate0_pass": bool(findings),
        "note": (
            "至少一条曝露在某个 horizon 上非重叠 |t|>=2 → 形状存在，Stage 1 有东西可拟合"
            if findings
            else "全部曝露在所有 horizon 上都不可辨 → ML 无信号可拟合，Gate 0 判停"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-0 growth exposure diagnostics.")
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 20, 60, 120])
    parser.add_argument("--universe", default="csi1000")
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-names", type=int, default=20)
    parser.add_argument("--sampling-days", type=int, default=21, help="Trading days between panel observations (monthly=21).")
    # Exposure set overrides — Stage 0's 8 raw growth exposures are the default;
    # pass --exposures to point the same diagnostics at another shard family
    # (e.g. the SUE v1 surprise exposures).
    parser.add_argument("--pattern", default="exposures_*.csv", help="Shard filename glob.")
    parser.add_argument("--exposures", default=None, help="Comma-separated raw exposure names (default: Stage-0 set).")
    parser.add_argument("--extra-cols", default=None, help="Comma-separated non-exposure columns to add to the IC table.")
    parser.add_argument("--focus", default=None, help="Comma-separated columns for the conditional-IC block.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global RAW_EXPOSURES
    if args.exposures:
        RAW_EXPOSURES = tuple(c.strip() for c in args.exposures.split(",") if c.strip())
    df = load_shards(args.shard_dir, args.pattern)
    horizons = [int(h) for h in args.horizons]
    z_cols = [f"z_{c}" for c in RAW_EXPOSURES if f"z_{c}" in df.columns]
    extra = [c.strip() for c in (args.extra_cols or "").split(",") if c.strip()]
    if not extra and "growth_score_v2_z" in df.columns:
        extra = ["growth_score_v2_z"]
    all_cols = [c for c in RAW_EXPOSURES if c in df.columns] + z_cols + [
        c for c in extra if c in df.columns
    ]

    print(f"[diag] rows={len(df)} dates={df['score_date'].nunique()} "
          f"symbols={df['symbol'].nunique()} range={df['score_date'].min()}..{df['score_date'].max()}")

    # Power: the binding constraint is independent observations per horizon.
    n_dates = df["score_date"].nunique()
    eff = {h: max(1, int(round(h / max(1, args.sampling_days)))) for h in horizons}
    print(
        f"[diag] 功效提示: {n_dates} 个月频日期; 重叠校正后每 horizon 独立观测约 "
        + ", ".join(f"h{h}→{n_dates // eff[h]}" for h in horizons)
    )

    cov = coverage_block(df)
    report: Dict[str, Any] = {"coverage": cov, "horizons": horizons, "rows": int(len(df))}

    ic = ic_block(df, all_cols, horizons, args.sampling_days)
    report["ic"] = ic
    print_ic_table(ic, all_cols, horizons)

    # Decompose upstream of the report so the JSON carries it too.
    quad = {}
    deciles = {}
    for h in horizons:
        quad[str(h)] = quadratic_block(df, z_cols, h, args.sampling_days)
        deciles[str(h)] = decile_block(df, z_cols, h, args.sampling_days)
    report["quadratic"] = quad
    report["deciles"] = deciles

    for h in horizons:
        print_decile_table(deciles[str(h)], h)
        print_quadratic_table(quad[str(h)], h)

    # conditional IC on the composite + a couple of representative exposures
    wanted = [c.strip() for c in args.focus.split(",")] if args.focus else [
        "growth_score_v2_z",
        "z_rev_yoy",
        "z_profit_yoy",
        "z_roe_level",
    ]
    focus = [c for c in wanted if c in df.columns]
    try:
        from shared.data_access import open_research_data_access_for_cli
        from analysis.research.score_ic_by_year_regime import load_regime_by_date

        db_manager = open_research_data_access_for_cli()
        score_dates = sorted(df["score_date"].unique())
        regime_by_date, regime_meta = load_regime_by_date(
            db_manager, args.universe, min(score_dates), score_dates
        )
        report["regime_meta"] = regime_meta
    except Exception as exc:  # regime labeling is a nice-to-have, not the verdict
        print(f"[diag] ⚠️ regime 标注失败，跳过条件 IC: {exc}")
        regime_by_date = {}
    if regime_by_date:
        cond = {str(h): conditional_ic(df, focus, h, regime_by_date, args.sampling_days) for h in horizons}
        report["conditional_ic"] = cond
        for h in horizons:
            print_conditional(cond[str(h)], h)

    report["redundancy"] = redundancy_block(df)

    gate = verdict(ic, quad, horizons)
    report["gate0"] = gate
    print("\n==== Gate 0 判据 ====")
    if gate["shaped_exposures"]:
        for line in gate["shaped_exposures"]:
            print(f"  ✓ {line}")
    print(f"  -> {gate['note']}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
        print(f"\n[diag] written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
