#!/usr/bin/env python3
"""TopN 子集内 RankIC —— 分数进组合篮子之后的残余排序信息。

[[repos/stock-scoring-system/score-ic-by-year-regime]] 证明「IC 稳 ≠ 能收割」：
全截面 rank IC 覆盖的是宽面单调性，组合研究实际只持有 top_n=10 的篮子。
本脚本回答另一半：**在 Top10 内部**，分数还能不能给收益排序？

口径：
* 每截面按分数取 TopN（rank asc=False, method=first，与组合研究同款选股），
  只在入选的 N 只内部算逐日 Spearman（Pearson-on-ranks），与全截面 IC 同源。
* 对比用「同一面板」现算的全截面 IC，并报告逐日差分 Δ = top_n IC − full IC，
  附非重叠 t（stride=h，h>1 时逐日 IC 重叠）。
* **功效声明**：n=10 的单日 Spearman sd ≈ 0.33，非重叠 s.e. ≈ 0.04–0.05，
  MDE80 ≈ 0.11–0.14。TopN 组内 IC 只分辨「大翻盘」，接近 0 的读数不作结论。
  报告里按实现 sd 现算 se / MDE80，不引用这里的心算值。

Example (inside quant-scorer image, workdir /app):
    python analysis/research/score_top10_subset_ic.py \
        --start-date 20200201 --end-date 20260930 --universe csi1000 \
        --horizons 1 5 10 20 30 60 --top-n 10 \
        --output reports/score_top10_ic_csi1000_2020_2026.json
"""
from __future__ import annotations

import argparse
import json
import math
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from analysis.research.score_ic_by_year_regime import (
    DEFAULT_COLUMNS,
    group_summaries,
    ic_by_date,
    prepare_panel,
    summarize_with_nonoverlap,
)

# 双侧 α=5% + 80% power：z_α/2 + z_β
Z_80 = 1.96 + 0.84


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TopN 子集内 RankIC 对比全截面 IC（stock_scores 面板）。"
    )
    parser.add_argument("--start-date", default="20200201")
    parser.add_argument("--end-date", default="20260930")
    parser.add_argument("--universe", default="csi1000")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10, 20, 30, 60])
    parser.add_argument("--top-n", type=int, default=10, help="组合研究的 top_n 档")
    parser.add_argument("--min-names-subset", type=int, default=8,
                        help="TopN 子集逐日最少有效股票数（< 8 该日跳过）")
    parser.add_argument("--full-min-names", type=int, default=20,
                        help="全截面 IC 逐日最少股票数（与主报告一致）")
    parser.add_argument("--columns", nargs="+", default=list(DEFAULT_COLUMNS))
    parser.add_argument("--output", help="JSON 输出路径")
    return parser.parse_args()


def build_report(
    dataset: pd.DataFrame,
    horizons: Sequence[int],
    columns: Sequence[str],
    *,
    top_n: int,
    min_names_subset: int,
    full_min_names: int,
) -> Dict[str, Any]:
    frame = dataset.copy()
    frame["score_date"] = frame["score_date"].astype(str)

    report: Dict[str, Any] = {}
    for column in columns:
        if column not in frame.columns:
            report[column] = {"error": "column missing from panel"}
            continue

        # TopN 入选 mask 与 horizon 无关，每列只算一次。
        rank_desc = frame.groupby("score_date", sort=False)[column].rank(
            ascending=False, method="first"
        )
        mask = rank_desc <= top_n

        per_horizon: Dict[str, Any] = {}
        for horizon in horizons:
            ret_col = f"forward_return_{horizon}d"
            if ret_col not in frame.columns:
                continue

            full = ic_by_date(frame, column, ret_col, full_min_names)
            subset = frame.loc[mask, ["score_date", column, ret_col]]
            top = ic_by_date(subset, column, ret_col, min_names_subset)

            block: Dict[str, Any] = {
                "full": summarize_with_nonoverlap(full, horizon) if not full.empty else None,
                "top": None,
                "diff": None,
                "mde": None,
            }
            if not full.empty and not top.empty:
                merged = full.merge(top, on="score_date", suffixes=("_full", "_top"))
                diff = pd.DataFrame(
                    {
                        "score_date": merged["score_date"],
                        "rank_ic": merged["rank_ic_top"] - merged["rank_ic_full"],
                        # summarize_ic 会读 count（avg_names）；用 TopN 侧的口径。
                        "count": merged["count_top"],
                    }
                )
                top_groups = group_summaries(
                    top, horizon, by_year=True, by_regime=False
                )
                per_date_sd = (
                    float(top["rank_ic"].std(ddof=1)) if len(top) > 1 else None
                )
                diff_summary = summarize_with_nonoverlap(diff, horizon)
                n_no = (
                    diff_summary.get("nonoverlap_observations")
                    if diff_summary else None
                )
                se = (
                    per_date_sd / math.sqrt(n_no)
                    if per_date_sd is not None and n_no
                    else None
                )
                block.update(
                    {
                        "top": {
                            "overall": summarize_with_nonoverlap(top, horizon),
                            "by_year": top_groups.get("by_year", []),
                        },
                        "diff": diff_summary,
                        "paired_observations": int(len(diff)),
                        "mde": {
                            "per_date_sd": per_date_sd,
                            "nonoverlap_n": n_no,
                            "se_nonoverlap": se,
                            "mde80": Z_80 * se if se is not None else None,
                        },
                    }
                )
            per_horizon[f"h{horizon}"] = block
        report[column] = per_horizon
    return report


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _year_signs(by_year: list) -> str:
    if not by_year:
        return "n/a"
    positive = 0
    for row in by_year:
        mean = row.get("ic_mean")
        if mean is not None and mean > 0:
            positive += 1
    return f"{positive}/{len(by_year)}"


def print_report(
    report: Dict[str, Any],
    columns: Sequence[str],
    horizons: Sequence[int],
    show_horizons: Sequence[int],
) -> None:
    for horizon in show_horizons:
        key = f"h{horizon}"
        print(f"\n==== TopN 子集 vs 全截面 RankIC @ h={horizon} ====")
        print(
            f"{'column':<30}{'fullIC':>9}{'f_no-t':>7}{'topIC':>9}{'t_no-t':>7}"
            f"{'delta':>9}{'d_no-t':>7}{'pairs':>6}{'signs(f->t)':>14}"
        )
        for column in columns:
            blocks = report.get(column) or {}
            if not isinstance(blocks, dict) or "error" in blocks:
                continue
            block = blocks.get(key)
            if not block or not block.get("top"):
                continue
            full = block["full"]
            top = block["top"]["overall"]
            diff = block["diff"]
            print(
                f"{column:<30}{_fmt(full['ic_mean']):>9}{_fmt(full['nonoverlap_t_stat'], 2):>7}"
                f"{_fmt(top['ic_mean']):>9}{_fmt(top['nonoverlap_t_stat'], 2):>7}"
                f"{_fmt(diff['ic_mean']):>9}{_fmt(diff['nonoverlap_t_stat'], 2):>7}"
                f"{block['paired_observations']:>6}"
                f"{_year_signs(block['top']['by_year']):>14}"
            )

    print("\n==== 功效声明（TopN 子集 per-date IC）@ 所有 horizon ====")
    print(f"{'column':<30}{'h':>4}{'sd':>9}{'N_no':>7}{'se':>9}{'MDE80':>9}")
    for column in columns:
        blocks = report.get(column) or {}
        if not isinstance(blocks, dict) or "error" in blocks:
            continue
        for horizon in horizons:
            block = (blocks or {}).get(f"h{horizon}") or {}
            mde = block.get("mde")
            if not mde:
                continue
            print(
                f"{column:<30}{horizon:>4}{_fmt(mde['per_date_sd'], 3):>9}"
                f"{mde['nonoverlap_n']:>7}{_fmt(mde['se_nonoverlap'], 4):>9}"
                f"{_fmt(mde['mde80'], 4):>9}"
            )

    # TopN 逐年明细（h=20 优先，其余进 JSON）
    focus_h = show_horizons[0] if show_horizons else horizons[-1]
    key = f"h{focus_h}"
    print(f"\n==== TopN 组内 IC 逐年 @ h={focus_h} ====")
    for column in columns:
        blocks = report.get(column) or {}
        if not isinstance(blocks, dict) or "error" in blocks:
            continue
        block = (blocks or {}).get(key) or {}
        by_year = (block.get("top") or {}).get("by_year") or []
        if not by_year:
            continue
        cells = "  ".join(
            f"{row['year'][2:]}:{_fmt(row['ic_mean'], 3)}(n={row['observations']})"
            for row in by_year
        )
        print(f"{column:<30}{cells}")


def main() -> int:
    args = _parse_args()
    columns = [str(c) for c in args.columns]
    horizons = [int(h) for h in args.horizons]

    # prepare_panel 吃 argparse.Namespace；这里固定与主报告相同的面板口径。
    panel_args = Namespace(
        universe=args.universe,
        start_date=args.start_date,
        end_date=args.end_date,
        horizons=horizons,
        use_panel_cache=True,
        growth_cycle_weights=True,
        score_spec=None,
    )
    db_manager, dataset, _ = prepare_panel(panel_args)
    print(
        f"[panel] rows={len(dataset)} dates={dataset['score_date'].nunique()} "
        f"columns={len(dataset.columns)}",
        flush=True,
    )

    report = {
        "config": {
            "universe": args.universe,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "horizons": horizons,
            "top_n": args.top_n,
            "min_names_subset": args.min_names_subset,
            "full_min_names": args.full_min_names,
            "columns": columns,
            "rows": int(len(dataset)),
        },
        "columns": build_report(
            dataset,
            horizons,
            columns,
            top_n=args.top_n,
            min_names_subset=args.min_names_subset,
            full_min_names=args.full_min_names,
        ),
    }

    # 终端突出组合研究相关的两个档；全部 horizon 留在 JSON。
    show = [h for h in (20, 30) if h in horizons]
    print_report(report["columns"], columns, horizons, show)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        print(f"\n[report] written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())