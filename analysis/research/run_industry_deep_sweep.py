#!/usr/bin/env python3
"""Run expanded industry L1 sweeps and summarize promotable parameter grids."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def load_promotable_l1_from_diagnosis(
    diagnosis_path: Path,
    *,
    horizon: int = 20,
    top_k: int = 8,
    min_excess_vs_industry: float = 0.0,
    exclude: Sequence[str] = (),
) -> List[str]:
    """Return L1 codes with positive mean Top-K excess vs their industry index."""
    payload = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    rows = (payload.get("industries_by_horizon") or {}).get(str(horizon)) or []
    excluded = {str(code).strip() for code in exclude}
    codes: List[str] = []
    for row in rows:
        if row.get("skipped"):
            continue
        code = str(row.get("l1_code") or row.get("slice_label") or "").strip()
        if not code or code in excluded:
            continue
        top_k_stats = (row.get("top_k") or {}).get(str(top_k)) or {}
        excess = top_k_stats.get("mean_excess_vs_industry_index")
        if excess is None:
            continue
        if float(excess) >= float(min_excess_vs_industry):
            codes.append(code)
    return sorted(set(codes))


@dataclass(frozen=True)
class IndustryDeepSummary:
    industry_l1: str
    row_count: int
    best_overall: Optional[Dict[str, Any]]
    best_by_path_gate: Dict[str, Dict[str, Any]]
    best_by_growth_weight: Dict[str, Dict[str, Any]]


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_score(row: Dict[str, Any]) -> float:
    score = _as_float(row.get("risk_adjusted_score"))
    return score if score is not None else float("-inf")


def _growth_weight_label(row: Dict[str, Any]) -> str:
    growth = _as_float(row.get("growth_weight"))
    cycle = _as_float(row.get("cycle_weight"))
    if growth is not None and cycle is not None:
        return f"g{int(round(growth * 100))}_c{int(round(cycle * 100))}"
    variant = str(row.get("score_variant") or "")
    if variant:
        return variant
    return "unknown"


def _row_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "score_variant": row.get("score_variant"),
        "growth_weight": row.get("growth_weight"),
        "cycle_weight": row.get("cycle_weight"),
        "path_gate": row.get("path_gate"),
        "top_n": row.get("top_n"),
        "cumulative_return": _as_float(row.get("cumulative_return")),
        "index_excess_cumulative_return": _as_float(row.get("index_excess_cumulative_return")),
        "sharpe": _as_float(row.get("sharpe")),
        "max_drawdown": _as_float(row.get("max_drawdown")),
        "risk_adjusted_score": _as_float(row.get("risk_adjusted_score")),
        "average_turnover": _as_float(row.get("average_turnover")),
        "average_selected_count": _as_float(row.get("average_selected_count")),
    }


def _best_row(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    return max(rows, key=_row_score)


def summarize_industry_deep_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    industry_l1: str,
) -> IndustryDeepSummary:
    industry_rows = [
        row
        for row in rows
        if str(row.get("industry_l1") or "").strip() == industry_l1
        or industry_l1 in (row.get("industry_l1_codes") or [])
    ]
    if not industry_rows:
        industry_rows = list(rows)

    best_by_path_gate: Dict[str, Dict[str, Any]] = {}
    for gate in sorted({str(row.get("path_gate") or "off") for row in industry_rows}):
        subset = [row for row in industry_rows if str(row.get("path_gate") or "off") == gate]
        best = _best_row(subset)
        if best is not None:
            best_by_path_gate[gate] = _row_metrics(best)

    best_by_growth_weight: Dict[str, Dict[str, Any]] = {}
    for label in sorted({_growth_weight_label(row) for row in industry_rows}):
        subset = [row for row in industry_rows if _growth_weight_label(row) == label]
        best = _best_row(subset)
        if best is not None:
            best_by_growth_weight[label] = _row_metrics(best)

    best_overall = _best_row(industry_rows)
    return IndustryDeepSummary(
        industry_l1=industry_l1,
        row_count=len(industry_rows),
        best_overall=_row_metrics(best_overall) if best_overall else None,
        best_by_path_gate=best_by_path_gate,
        best_by_growth_weight=best_by_growth_weight,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run industry L1 deep sweeps and summarize results.")
    parser.add_argument("--run-sweep", action="store_true", help="Execute portfolio_sweep_report sweeps")
    parser.add_argument(
        "--industry-l1",
        nargs="+",
        default=["801730.SI", "801150.SI"],
        help="Shenwan L1 codes to sweep (default: gate-passed 电力设备 / 医药)",
    )
    parser.add_argument("--start-date", default="20200101")
    parser.add_argument("--end-date", default="20260814")
    parser.add_argument("--universe-index", nargs="+", default=["csi1000"])
    parser.add_argument("--score-column", default="composite_growth_cycle_score")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--top-n", nargs="+", type=int, default=[5, 8, 10])
    parser.add_argument("--active-caps", nargs="+", type=float, default=[0.3])
    parser.add_argument(
        "--growth-cycle-weights",
        nargs="+",
        default=["30:70", "50:50", "70:30"],
    )
    parser.add_argument(
        "--path-gates",
        nargs="+",
        default=["off", "ret20_pos", "above_ma60"],
    )
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--json-input", help="Existing sweep JSON to summarize only")
    parser.add_argument("--output-json", help="Sweep JSON output path (single industry runs)")
    parser.add_argument("--summary-output", help="Summary JSON path for all industries")
    parser.add_argument(
        "--from-diagnosis",
        help="Pick industry codes from industry_slice_diagnosis JSON (positive industry-index Top-K excess)",
    )
    parser.add_argument("--diagnosis-horizon", type=int, default=20)
    parser.add_argument("--diagnosis-top-k", type=int, default=8)
    parser.add_argument("--min-excess-vs-industry", type=float, default=0.0)
    parser.add_argument("--exclude-industry-l1", nargs="+", default=[])
    return parser.parse_args()


def _run_sweep(
    repo_root: Path,
    *,
    output_json: Path,
    args: argparse.Namespace,
    industry_l1: str,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "analysis.sweep.portfolio_sweep_report",
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--universe-index",
        *args.universe_index,
        "--score-column",
        args.score_column,
        "--horizon",
        str(args.horizon),
        "--top-n",
        *[str(value) for value in args.top_n],
        "--active-caps",
        *[str(value) for value in args.active_caps],
        "--growth-cycle-weights",
        *args.growth_cycle_weights,
        "--path-gates",
        *args.path_gates,
        "--transaction-cost",
        str(args.transaction_cost),
        "--json-output",
        str(output_json),
        "--industry-l1",
        industry_l1,
        "--quiet",
    ]
    print("Running deep sweep:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=repo_root, check=True)


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    industry_codes = list(args.industry_l1)
    if args.from_diagnosis:
        diagnosis_path = Path(args.from_diagnosis)
        industry_codes = load_promotable_l1_from_diagnosis(
            diagnosis_path,
            horizon=args.diagnosis_horizon,
            top_k=args.diagnosis_top_k,
            min_excess_vs_industry=args.min_excess_vs_industry,
            exclude=args.exclude_industry_l1,
        )
        print(f"Diagnosis-selected L1 codes ({len(industry_codes)}): {industry_codes}", flush=True)
        if not industry_codes:
            print("No promotable L1 codes from diagnosis.", file=sys.stderr)
            return 1

    summaries: List[Dict[str, Any]] = []
    for industry_l1 in industry_codes:
        slug = industry_l1.replace(".", "_")
        sweep_json = Path(args.json_input or reports_dir / f"industry_deep_{slug}.json")
        if args.run_sweep:
            sweep_json = Path(args.output_json or reports_dir / f"industry_deep_{slug}.json")
            _run_sweep(repo_root, output_json=sweep_json, args=args, industry_l1=industry_l1)

        if not sweep_json.is_file():
            print(f"missing sweep JSON for {industry_l1}: {sweep_json}", file=sys.stderr)
            return 1

        payload = json.loads(sweep_json.read_text(encoding="utf-8"))
        rows = list(payload.get("rows") or [])
        summary = summarize_industry_deep_rows(rows, industry_l1=industry_l1)
        entry = {
            "industry_l1": summary.industry_l1,
            "row_count": summary.row_count,
            "sweep_json": str(sweep_json),
            "best_overall": summary.best_overall,
            "best_by_path_gate": summary.best_by_path_gate,
            "best_by_growth_weight": summary.best_by_growth_weight,
        }
        summaries.append(entry)
        print(json.dumps(entry, ensure_ascii=False, indent=2), flush=True)

    if args.summary_output:
        out_path = Path(args.summary_output)
        out_path.write_text(
            json.dumps(
                {
                    "industries": summaries,
                    "setup": {
                        "start_date": args.start_date,
                        "end_date": args.end_date,
                        "universe_index": args.universe_index,
                        "horizon": args.horizon,
                        "top_n": args.top_n,
                        "growth_cycle_weights": args.growth_cycle_weights,
                        "path_gates": args.path_gates,
                        "transaction_cost": args.transaction_cost,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Summary written: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
