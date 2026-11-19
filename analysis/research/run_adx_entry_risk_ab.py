#!/usr/bin/env python3
"""ADX entry-risk overlay A/B on growth-cycle portfolio (research only).

Runs frozen ADX tier exposure vs baseline / regime cash; writes scorecard under
``reports/portfolio-research/adx-entry-risk/``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

from analysis.research.adx_entry_risk_falsify import (
    build_entry_factor_panel,
    factors_aligned_to_dataset,
    exclusion_masks,
    run_falsification_backtest,
    summarize_periods,
)
from analysis.research.index_adx_entry_risk import (
    EXPOSURE_FULL,
    build_indicator_frame,
    exposure_on_rebalance_dates,
    exposure_schedule_t_plus_one,
    load_index_rows_for_research,
    rule_flags,
    run_diagnostics,
    write_diagnostics_json,
)
from shared.regime_label import label_as_of, label_index_regime


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ADX last-bar entry-risk A/B")
    p.add_argument("--start-date", default="20210101")
    p.add_argument("--end-date", default="20260930")
    p.add_argument("--universe-index", nargs="+", default=["csi1000"])
    p.add_argument("--benchmark", default="000852.SH")
    p.add_argument("--secondary-benchmark", default="000300.SH")
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--growth-cycle-weights", default="60:40")
    p.add_argument("--trailing-stop", type=float, default=0.15)
    p.add_argument("--transaction-cost", type=float, default=0.001)
    p.add_argument("--out-dir", default="reports/portfolio-research/adx-entry-risk")
    p.add_argument("--rf-annual", type=float, default=0.02)
    p.add_argument("--skip-secondary", action="store_true")
    p.add_argument("--no-panel-cache", action="store_true")
    p.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Skip portfolio backtest; write index diagnostics and WF gates only",
    )
    p.add_argument("--skip-falsify", action="store_true", help="Skip not_hipos97 / not_distrib05 arms")
    p.add_argument("--falsify-workers", type=int, default=8)
    return p.parse_args()


def _md_table(rows: List[Dict[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def _write_scorecard_html(path: Path, title: str, arm_rows: List[Dict[str, Any]], gates: Dict[str, Any]) -> None:
    rows_html = "".join(
        f"<tr><td>{r.get('arm','')}</td><td>{r.get('avg_exposure','')}</td>"
        f"<td>{r.get('cumulative','')}</td><td>{r.get('mdd','')}</td>"
        f"<td>{r.get('timing_edge','')}</td><td>{r.get('p_rot','')}</td></tr>"
        for r in arm_rows
    )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:sans-serif;margin:2rem}} table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:6px 10px}}</style></head>
<body><h1>{title}</h1><h2>Gates</h2><pre>{json.dumps(gates, indent=2)}</pre>
<h2>Portfolio arms</h2><table><tr><th>arm</th><th>avg exp</th><th>cum</th><th>mdd</th>
<th>timing edge</th><th>p_rot</th></tr>{rows_html}</table></body></html>"""
    path.write_text(html, encoding="utf-8")


def _evaluate_gates(arms: Dict[str, Dict[str, Any]], wf: Dict[str, Any]) -> Dict[str, Any]:
    adx = arms.get("adx tier", {})
    edge = float(adx.get("timing_edge", 0.0))
    p_rot = float(adx.get("p_rot", 1.0))
    g1 = edge > 0
    g2 = p_rot <= 0.10
    g3 = float(wf.get("win_rate", 0.0)) >= 0.60
    g4 = int(wf.get("recent_wins_2024_2026", 0)) >= 2
    passed = g1 and g2 and g3 and g4
    return {
        "timing_edge_positive": g1,
        "rotation_p_rot_lte_0_10": g2,
        "wf_win_rate_gte_60pct": g3,
        "recent_2024_2026_wins_gte_2": g4,
        "all_pass": passed,
        "verdict": "candidate for product planning" if passed else "advisory only — gates not met",
    }


def _run_diagnostics_only(args: argparse.Namespace, out_dir: Path, started: float) -> int:
    from shared.data_access import open_research_data_access_for_cli

    db = open_research_data_access_for_cli(enable_tushare=False)
    primary = run_diagnostics(
        load_index_rows_for_research(db, args.benchmark, args.start_date, args.end_date),
        symbol=args.benchmark,
    )
    write_diagnostics_json(primary, str(out_dir / f"diagnostics_{args.benchmark.replace('.', '_')}.json"))
    secondary: Dict[str, Any] = {}
    if not args.skip_secondary and args.secondary_benchmark:
        secondary = run_diagnostics(
            load_index_rows_for_research(
                db, args.secondary_benchmark, args.start_date, args.end_date
            ),
            symbol=args.secondary_benchmark,
        )
        write_diagnostics_json(
            secondary,
            str(out_dir / f"diagnostics_{args.secondary_benchmark.replace('.', '_')}.json"),
        )
    summary = {
        "params": vars(args),
        "diagnostics_primary": primary,
        "diagnostics_secondary": secondary,
        "elapsed_sec": time.perf_counter() - started,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    Path(out_dir / "scorecard.md").write_text(
        "# ADX diagnostics-only\n\n"
        + json.dumps(primary.get("walk_forward", {}), indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"[adx-ab] diagnostics-only -> {out_dir} ({summary['elapsed_sec']:.0f}s)")
    return 0


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.diagnostics_only:
        return _run_diagnostics_only(args, out_dir, started)

    from analysis.sweep.growth_panel_baseline import prepare_panel, run_baseline
    from analysis.sweep.timing_exposure_metrics import (
        exposure_arms,
        overlay_robustness,
        shift_edge_distribution,
    )

    class CliArgs:
        start_date = args.start_date
        end_date = args.end_date
        universe = args.universe_index[0]
        score_column = "composite_growth_cycle_score"
        growth_cycle_weights = args.growth_cycle_weights
        top_n = args.top_n
        rebalance_days = args.horizon
        construction_mode = "industry_capped"
        industry_cap = 0.3
        transaction_cost = args.transaction_cost
        regime_rule = None
        rf_annual = args.rf_annual
        vol_target = 0.22
        vol_lookback = 60
        out = None
        use_panel_cache = not args.no_panel_cache

    db, dataset, price_cache, score_sweeps, horizons = prepare_panel(CliArgs())
    from analysis.sweep.score_columns import add_weighted_score_columns

    dataset = add_weighted_score_columns(dataset.copy(), score_sweeps)
    _, score_column, periods, trades, index_prices, benchmark_symbol, interval = run_baseline(
        CliArgs(), db, dataset, price_cache, score_sweeps, horizons
    )
    baseline_summary = summarize_periods(periods)
    dates = [str(d) for d in periods["score_date"]]
    net = [float(v) for v in periods["portfolio_return_net"]]

    index_rows = load_index_rows_for_research(db, args.benchmark, args.start_date, args.end_date)
    diag_primary = run_diagnostics(index_rows, symbol=args.benchmark)
    write_diagnostics_json(diag_primary, str(out_dir / f"diagnostics_{args.benchmark.replace('.', '_')}.json"))

    frame = rule_flags(build_indicator_frame(index_rows))
    adx_exp = exposure_schedule_t_plus_one(frame)
    labels = label_index_regime(index_rows)
    regime_switch = [1.0 if label_as_of(labels, d) == "bull" else 0.0 for d in dates]
    adx_on_dates = exposure_on_rebalance_dates(adx_exp, dates)
    combined = [
        min(r, a) for r, a in zip(regime_switch, adx_on_dates)
    ]

    arms_paths: Dict[str, List[float]] = {
        "always full": [EXPOSURE_FULL] * len(dates),
        "regime bull/cash": regime_switch,
        "adx tier": adx_on_dates,
        "regime and adx": combined,
    }
    arms_metrics: Dict[str, Dict[str, Any]] = {}
    for name, path in arms_paths.items():
        metrics = exposure_arms(net, path, rf_annual=args.rf_annual, interval_days=interval)
        rotations = shift_edge_distribution(net, path, rf_annual=args.rf_annual, interval_days=interval)
        beats = sum(1 for edge in rotations if edge >= metrics["timing_edge"])
        p_rot = (beats + 1) / (len(rotations) + 1)
        metrics["p_rot"] = p_rot
        metrics["rotation_beats"] = beats
        metrics["rotation_n"] = len(rotations)
        arms_metrics[name] = metrics

    falsify_summaries: Dict[str, Any] = {}
    if not args.skip_falsify:
        symbols = dataset["symbol"].astype(str).unique().tolist()
        print(f"[adx-ab] building entry factors for {len(symbols)} symbols ...", flush=True)
        factors = build_entry_factor_panel(
            db,
            symbols,
            args.start_date,
            args.end_date,
            max_workers=int(args.falsify_workers),
        )
        masks = exclusion_masks(factors_aligned_to_dataset(dataset, factors))
        for label, col in (
            ("falsify not_hipos97", "exclude_hipos97"),
            ("falsify not_distrib05", "exclude_distrib05"),
        ):
            f_periods, f_sum = run_falsification_backtest(
                dataset,
                score_column,
                masks,
                col,
                db=db,
                price_cache=price_cache,
                index_prices=index_prices,
                top_n=args.top_n,
                horizon=args.horizon,
                transaction_cost=args.transaction_cost,
                benchmark_symbol=benchmark_symbol,
            )
            f_net = [float(v) for v in f_periods["portfolio_return_net"]]
            full_path = [EXPOSURE_FULL] * len(f_net)
            metrics = exposure_arms(f_net, full_path, rf_annual=args.rf_annual, interval_days=interval)
            falsify_summaries[label] = {**f_sum, **metrics}

    gates = _evaluate_gates(arms_metrics, diag_primary["walk_forward"])

    secondary: Dict[str, Any] = {}
    if not args.skip_secondary and args.secondary_benchmark:
        sec_rows = load_index_rows_for_research(
            db, args.secondary_benchmark, args.start_date, args.end_date
        )
        secondary = run_diagnostics(sec_rows, symbol=args.secondary_benchmark)
        write_diagnostics_json(
            secondary, str(out_dir / f"diagnostics_{args.secondary_benchmark.replace('.', '_')}.json")
        )

    summary = {
        "params": vars(args),
        "benchmark": benchmark_symbol,
        "score_column": score_column,
        "n_periods": len(dates),
        "arms": arms_metrics,
        "baseline_summary": baseline_summary,
        "falsification": falsify_summaries,
        "gates": gates,
        "diagnostics_primary": diag_primary,
        "diagnostics_secondary": secondary,
        "reproduce_cmd": (
            "PYTHONPATH=. python analysis/research/run_adx_entry_risk_ab.py "
            f"--start-date {args.start_date} --end-date {args.end_date} "
            f"--benchmark {args.benchmark} --top-n {args.top_n} --horizon {args.horizon}"
        ),
        "elapsed_sec": time.perf_counter() - started,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    arm_rows = [
        {
            "arm": name,
            "avg_exposure": f"{m['average_exposure']:.3f}",
            "cumulative": f"{m['cumulative']:.2%}",
            "mdd": f"{m['max_drawdown']:.2%}",
            "timing_edge": f"{m['timing_edge']:+.2%}",
            "p_rot": f"{m.get('p_rot', float('nan')):.3f}",
        }
        for name, m in arms_metrics.items()
    ]
    md = [
        "# ADX entry-risk A/B scorecard",
        "",
        f"- Window: `{args.start_date}` – `{args.end_date}`",
        f"- Universe: `{args.universe_index}` top_n={args.top_n} horizon={args.horizon}",
        f"- Benchmark: `{args.benchmark}`",
        "",
        "## Baseline (always full, no overlay)",
        "",
        f"- Cumulative: **{baseline_summary.get('cumulative', 0):.2%}**",
        f"- Worst period: **{baseline_summary.get('worst_period_return', 0):.2%}**",
        f"- Yearly: `{baseline_summary.get('yearly_returns', {})}`",
        "",
        "## Portfolio arms (matched-exposure control)",
        "",
        _md_table(arm_rows, ("arm", "avg_exposure", "cumulative", "mdd", "timing_edge", "p_rot")),
        "",
        "## Falsification (hard stock filters, expected to hurt)",
        "",
    ]
    if falsify_summaries:
        falsify_rows = [
            {
                "arm": name,
                "cumulative": f"{m.get('cumulative', 0):.2%}",
                "mdd": f"{m.get('max_drawdown', 0):.2%}",
                "vs_baseline_cum": f"{(m.get('cumulative') or 0) - (baseline_summary.get('cumulative') or 0):+.2%}",
            }
            for name, m in falsify_summaries.items()
        ]
        md.append(_md_table(falsify_rows, ("arm", "cumulative", "mdd", "vs_baseline_cum")))
    else:
        md.append("(skipped)")
    md.extend(
        [
            "",
            "## Gates",
            "",
            json.dumps(gates, indent=2),
            "",
            "## Index diagnostics (primary)",
            "",
            f"- Exposure mean: {diag_primary['exposure_mean']:.3f}",
            f"- Rule hits: {diag_primary['rule_hit_rates']}",
            f"- Peak lags: {diag_primary['peak_lags']}",
            "",
            "## Regime overlay LOO (reference)",
            "",
        ]
    )
    for year, kept, cumulative, edge in overlay_robustness(
        net, regime_switch, dates, rf_annual=args.rf_annual, interval_days=interval
    ):
        md.append(f"- {year}: n={kept} cum={cumulative:.2%} edge={edge:+.2%}")
    md.extend(
        [
            "",
            "## Reproduce",
            "",
            f"```bash\n{summary['reproduce_cmd']}\n```",
        ]
    )
    Path(out_dir / "scorecard.md").write_text("\n".join(md), encoding="utf-8")
    _write_scorecard_html(out_dir / "scorecard.html", "ADX entry-risk A/B", arm_rows, gates)
    print(f"[adx-ab] wrote {out_dir}/summary.json and scorecard.md ({summary['elapsed_sec']:.0f}s)")
    print(f"[adx-ab] verdict: {gates['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
