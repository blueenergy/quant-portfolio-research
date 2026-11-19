#!/usr/bin/env python3
"""Run or evaluate volatility position-scaling A/B sweeps and rollout gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from analysis.research.volatility_ab_gate import evaluate_volatility_ab, load_sweep_rows, preset_patch_for_gate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate volatility A/B sweep output and optional preset patch.")
    parser.add_argument("--json-input", help="Existing sweep JSON payload to evaluate")
    parser.add_argument("--run-sweep", action="store_true", help="Run portfolio_sweep_report with three volatility modes")
    parser.add_argument("--start-date", default="20250101")
    parser.add_argument("--end-date", default="20260501")
    parser.add_argument("--universe-index", nargs="+", default=["csi1000"])
    parser.add_argument("--score-column", default="composite_growth_cycle_score")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--top-n", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--active-caps", nargs="+", type=float, default=[0.3])
    parser.add_argument("--target-vols", nargs="+", type=float, default=[0.30, 0.40, 0.50])
    parser.add_argument("--output-json", help="Where to write sweep JSON when --run-sweep is used")
    parser.add_argument("--apply-preset-patch", action="store_true", help="Print Mongo preset patch JSON if gate passes")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    json_path = Path(args.json_input) if args.json_input else None
    if args.run_sweep:
        output_json = Path(args.output_json or repo_root / "reports" / "volatility_ab_sweep.json")
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
            "--volatility-modes",
            "off",
            "cash_drag",
            "renormalize",
            "--target-vols",
            *[str(value) for value in args.target_vols],
            "--json-output",
            str(output_json),
            "--quiet",
        ]
        print("Running volatility A/B sweep:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=repo_root, check=True)
        json_path = output_json

    if json_path is None:
        raise SystemExit("Provide --json-input or use --run-sweep.")

    decision = evaluate_volatility_ab(load_sweep_rows(json_path))
    print(json.dumps(decision.__dict__, ensure_ascii=False, indent=2, default=str))
    if args.apply_preset_patch:
        patch = preset_patch_for_gate(decision)
        if patch:
            print("\nSuggested preset params patch:")
            print(json.dumps(patch, ensure_ascii=False, indent=2))
        else:
            print("\nGate did not pass; no preset patch suggested.", file=sys.stderr)
            return 1
    return 0 if decision.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
