#!/usr/bin/env python3
"""Run or evaluate Shenwan L1 industry-filter A/B sweeps and rollout gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from analysis.research.industry_ab_gate import evaluate_industry_ab


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline + industry-filter sweeps and evaluate industry A/B gate."
    )
    parser.add_argument("--json-input", help="Existing merged sweep JSON (baseline + industry rows)")
    parser.add_argument("--run-sweep", action="store_true", help="Run paired portfolio sweeps")
    parser.add_argument("--start-date", default="20200101")
    parser.add_argument("--end-date", default="20260814")
    parser.add_argument("--universe-index", nargs="+", default=["csi1000"])
    parser.add_argument("--score-column", default="composite_growth_cycle_score")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--top-n", nargs="+", type=int, default=[8, 10])
    parser.add_argument("--active-caps", nargs="+", type=float, default=[0.3])
    parser.add_argument(
        "--growth-cycle-weights",
        nargs="+",
        default=["30:70"],
        help="Fixed growth:cycle weight specs, e.g. 30:70",
    )
    parser.add_argument(
        "--industry-l1",
        required=False,
        help="Target Shenwan L1 code or Chinese name (required with --run-sweep)",
    )
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--output-json", help="Merged sweep JSON path when --run-sweep is used")
    parser.add_argument("--gate-output", help="Optional JSON path for gate decision")
    return parser.parse_args()


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("rows") or [])


def _merge_sweep_payloads(
    baseline_path: Path,
    industry_path: Path,
    *,
    industry_l1: str,
) -> Dict[str, Any]:
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    industry_payload = json.loads(industry_path.read_text(encoding="utf-8"))
    rows = list(baseline_payload.get("rows") or []) + list(industry_payload.get("rows") or [])
    return {
        "rows": rows,
        "industry_l1": industry_l1,
        "baseline_source": str(baseline_path),
        "industry_source": str(industry_path),
    }


def _run_sweep(
    repo_root: Path,
    *,
    output_json: Path,
    args: argparse.Namespace,
    industry_l1: Optional[List[str]] = None,
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
        "--transaction-cost",
        str(args.transaction_cost),
        "--json-output",
        str(output_json),
        "--quiet",
    ]
    if industry_l1:
        cmd.extend(["--industry-l1", *industry_l1])
    print("Running sweep:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=repo_root, check=True)


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    json_path = Path(args.json_input) if args.json_input else None
    industry_l1 = str(args.industry_l1 or "").strip()

    if args.run_sweep:
        if not industry_l1:
            raise SystemExit("--industry-l1 is required when using --run-sweep.")
        reports_dir = repo_root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        slug = industry_l1.replace(".", "_")
        baseline_json = reports_dir / f"industry_ab_baseline_{slug}.json"
        industry_json = reports_dir / f"industry_ab_filtered_{slug}.json"
        merged_json = Path(args.output_json or reports_dir / f"industry_ab_merged_{slug}.json")

        _run_sweep(repo_root, output_json=baseline_json, args=args, industry_l1=None)
        _run_sweep(
            repo_root,
            output_json=industry_json,
            args=args,
            industry_l1=[industry_l1],
        )
        merged = _merge_sweep_payloads(baseline_json, industry_json, industry_l1=industry_l1)
        merged_json.write_text(json.dumps(merged, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        json_path = merged_json
        print(f"Merged sweep JSON: {merged_json}", flush=True)

    if json_path is None:
        raise SystemExit("Provide --json-input or use --run-sweep.")
    if not industry_l1:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        industry_l1 = str(payload.get("industry_l1") or "").strip()
    if not industry_l1:
        raise SystemExit("industry_l1 is required (flag or merged JSON field).")

    decision = evaluate_industry_ab(_load_rows(json_path), industry_l1=industry_l1)
    out = {
        "passed": decision.passed,
        "industry_l1": industry_l1,
        "reasons": decision.reasons,
        "comparisons": decision.comparisons,
        "sweep_json": str(json_path),
    }
    text = json.dumps(out, ensure_ascii=False, indent=2, default=str)
    print(text)
    if args.gate_output:
        Path(args.gate_output).write_text(text, encoding="utf-8")
    return 0 if decision.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
