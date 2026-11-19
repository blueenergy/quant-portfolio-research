#!/usr/bin/env python3
"""MVP: champion-style grid × entry_stock_filters for csi2000 / csi1000.

Worker-equivalent params (submit to portfolio research jobs):

  entry_stock_filters: ["off", "distrib03", "distrib05", "distrib08", "hipos97", "hipos97_distrib"]
  top_n_values: [10]
  rebalance_interval_days / horizons: [20]
  growth_cycle_weights: ["60:40"]
  active_caps: [0.3]
  trailing_stop_pcts: [0]
  path_gates: ["off"]
  regime_modes: ["off"]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from analysis.sweep import portfolio_sweep_report as sweep
from analysis.sweep.entry_stock_filter import (
    ENTRY_STOCK_FILTER_OFF,
    apply_entry_stock_filter,
    build_factor_panel_for_dataset,
    normalize_entry_stock_filters,
)
from analysis.sweep.growth_panel_baseline import prepare_panel
from shared.industry_universe import attach_sw_l1
from analysis.core.benchmark_prices import load_index_benchmark_prices
from shared.universe_benchmark import default_index_benchmark_symbol


def _load_env() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _parse_args() -> argparse.Namespace:
    _load_env()
    p = argparse.ArgumentParser(description="Entry stock filter MVP sweep (csi2000/csi1000)")
    p.add_argument("--start-date", default="20230831")
    p.add_argument("--end-date", default="20261009")
    p.add_argument(
        "--universes",
        default="csi2000,csi1000",
        help="Comma-separated universe index codes",
    )
    p.add_argument(
        "--entry-filters",
        default="off,distrib03,distrib05,distrib08,hipos97,hipos97_distrib",
    )
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--rebalance-days", type=int, default=20)
    p.add_argument("--growth-cycle-weights", default="60:40")
    p.add_argument("--industry-cap", type=float, default=0.3)
    p.add_argument("--transaction-cost", type=float, default=0.001)
    p.add_argument("--focus-score-date", default="20260629")
    p.add_argument("--out-dir", default="reports/portfolio-research/entry-filter-mvp")
    p.add_argument("--no-panel-cache", action="store_true")
    p.add_argument("--falsify-workers", type=int, default=8)
    p.add_argument("--mongo-db", default=os.environ.get("MONGO_DB", "finance"))
    return p.parse_args()


def _cli_ns(ns: argparse.Namespace, universe: str) -> argparse.Namespace:
    class A:
        start_date = ns.start_date
        end_date = ns.end_date
        universe = universe
        score_column = "composite_growth_cycle_score"
        growth_cycle_weights = ns.growth_cycle_weights
        top_n = ns.top_n
        rebalance_days = ns.rebalance_days
        construction_mode = "industry_capped"
        industry_cap = ns.industry_cap
        transaction_cost = ns.transaction_cost
        use_panel_cache = not ns.no_panel_cache

    return A()


def _focus_row(rows: List[Dict[str, Any]], score_date: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for row in rows:
        ef = str(row.get("entry_stock_filter") or "off")
        periods = row.get("periods") or []
        for period in periods:
            if str(period.get("score_date")) == score_date:
                out[ef] = {
                    "period_return": period.get("period_return"),
                    "n_positions": period.get("n_positions"),
                }
                break
    return out


def _run_universe(ns: argparse.Namespace, universe: str) -> Dict[str, Any]:
    args = _cli_ns(ns, universe)
    db_manager, base_dataset, _price_cache, score_sweeps, _horizons = prepare_panel(args)
    if base_dataset is None or base_dataset.empty:
        raise RuntimeError(f"empty dataset for {universe}")
    base_dataset = attach_sw_l1(base_dataset, db_manager, as_of_column="score_date")

    from analysis.sweep.score_columns import add_weighted_score_columns

    base_dataset = add_weighted_score_columns(base_dataset, score_sweeps)
    score_column = score_sweeps[0].score_column
    index_benchmark = default_index_benchmark_symbol([universe])
    if not index_prices:
        index_prices = load_index_benchmark_prices(db_manager, index_benchmark, args.start_date)

    entry_filters = normalize_entry_stock_filters(
        [v.strip() for v in ns.entry_filters.split(",") if v.strip()]
    )
    factor_panel = None
    if any(v != ENTRY_STOCK_FILTER_OFF for v in entry_filters):
        factor_panel = build_factor_panel_for_dataset(
            db_manager,
            base_dataset,
            args.start_date,
            args.end_date,
            max_workers=ns.falsify_workers,
        )

    all_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for entry_filter in entry_filters:
        dataset = apply_entry_stock_filter(
            db_manager,
            base_dataset,
            score_column,
            entry_filter,
            args.start_date,
            args.end_date,
            factor_panel=factor_panel,
            max_workers=ns.falsify_workers,
        )
        part = sweep.run_parameter_sweep(
            dataset=dataset,
            score_column=score_column,
            top_n_values=[args.top_n],
            horizon=args.rebalance_days,
            rebalance_intervals=[args.rebalance_days],
            variants=sweep.build_sweep_variants_for_request([args.industry_cap], []),
            transaction_cost=args.transaction_cost,
            index_benchmark_symbol=index_benchmark,
            index_price_history=index_prices,
            score_sweeps=score_sweeps,
            vol_variants=sweep.build_volatility_variants(None, None),
            path_gates=["off"],
            regime_modes=["off"],
            trailing_stop_pct=None,
        )
        for row in part.get("rows") or []:
            row["entry_stock_filter"] = entry_filter if entry_filter != ENTRY_STOCK_FILTER_OFF else "off"
            row["universe_index"] = universe
        all_rows.extend(part.get("rows") or [])
        print(
            f"[{universe}] entry_stock_filter={entry_filter} "
            f"cum={all_rows[-1].get('cumulative_return') if all_rows else 'n/a'}",
            flush=True,
        )

    elapsed = time.time() - t0
    ranked = sorted(
        all_rows,
        key=lambda r: float(r.get("cumulative_return") or 0),
        reverse=True,
    )
    return {
        "universe": universe,
        "elapsed_sec": round(elapsed, 1),
        "rows": all_rows,
        "ranked_by_return": [
            {
                "entry_stock_filter": r.get("entry_stock_filter"),
                "cumulative_return": r.get("cumulative_return"),
                "max_drawdown": r.get("max_drawdown"),
                "risk_adjusted_score": r.get("risk_adjusted_score"),
            }
            for r in ranked
        ],
        "focus_rebalance": _focus_row(all_rows, ns.focus_score_date),
    }


def main() -> None:
    ns = _parse_args()
    os.environ["MONGO_DB"] = str(ns.mongo_db)
    universes = [u.strip() for u in ns.universes.split(",") if u.strip()]
    out_dir = Path(ns.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "start_date": ns.start_date,
        "end_date": ns.end_date,
        "entry_filters": ns.entry_filters,
        "focus_score_date": ns.focus_score_date,
        "universes": {},
    }
    for universe in universes:
        summary["universes"][universe] = _run_universe(ns, universe)

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
