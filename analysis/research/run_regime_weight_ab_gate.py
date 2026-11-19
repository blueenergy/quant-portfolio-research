#!/usr/bin/env python3
"""Run or evaluate regime-switched growth/cycle weight A/B sweeps and rollout gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shared.regime_label import load_regime_rule, regime_rule_slug
from analysis.research.regime_weight_ab_gate import evaluate_regime_weight_ab, load_sweep_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate regime-switched growth/cycle weight sweep output using algorithm segments."
    )
    parser.add_argument("--json-full", help="Full-window sweep JSON payload")
    parser.add_argument("--json-bear", help="Algorithm primary bear-segment sweep JSON")
    parser.add_argument("--json-bull", help="Algorithm primary bull-segment sweep JSON")
    parser.add_argument("--diagnosis-json", help="Diagnosis JSON with regime_segments (required with --run-sweep)")
    parser.add_argument("--run-sweep", action="store_true", help="Run diagnosis + full/bear/bull segment sweeps")
    parser.add_argument("--start-date", default="20230101")
    parser.add_argument("--end-date", default="20260501")
    parser.add_argument("--universe-index", nargs="+", default=["csi1000"])
    parser.add_argument("--score-column", default="composite_growth_cycle_score")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--top-n", nargs="+", type=int, default=[20])
    parser.add_argument("--active-caps", nargs="+", type=float, default=[0.3])
    parser.add_argument("--output-dir", help="Directory for sweep JSON when --run-sweep is used")
    parser.add_argument(
        "--regime-rule",
        default=None,
        help="Preset id, JSON path, or inline JSON for injectable bull/bear conditions",
    )
    parser.add_argument("--gate-output", help="Optional JSON path for gate decision")
    return parser.parse_args()


def _rule_slug(args: argparse.Namespace) -> str:
    return regime_rule_slug(load_regime_rule(args.regime_rule).rule_id)


def _append_regime_rule_args(cmd: List[str], args: argparse.Namespace) -> None:
    if args.regime_rule:
        cmd.extend(["--regime-rule", str(args.regime_rule)])


def _run_sweep(
    repo_root: Path,
    *,
    output_json: Path,
    args: argparse.Namespace,
    start_date: str,
    end_date: str,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "analysis.sweep.portfolio_sweep_report",
        "--start-date",
        start_date,
        "--end-date",
        end_date,
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
        "20:80",
        "60:40",
        "--regime-weight",
        "--json-output",
        str(output_json),
        "--quiet",
    ]
    _append_regime_rule_args(cmd, args)
    print("Running regime sweep:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=repo_root, check=True)


def _run_diagnosis(
    repo_root: Path,
    *,
    args: argparse.Namespace,
    output_json: Path,
) -> Dict[str, Any]:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "analysis.research.regime_weight_diagnosis",
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--universe-index",
        args.universe_index[0] if len(args.universe_index) == 1 else args.universe_index[0],
        "--json-output",
        str(output_json),
    ]
    _append_regime_rule_args(cmd, args)
    print("Running regime diagnosis:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=repo_root, check=True)
    return load_diagnosis_payload(output_json)


def load_diagnosis_payload(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def segment_window(diagnosis: Dict[str, Any], label: str) -> Optional[Tuple[str, str]]:
    key = "primary_bear_segment" if label == "bear" else "primary_bull_segment"
    segment = diagnosis.get(key)
    if not segment:
        return None
    start = str(segment.get("start") or "").strip()
    end = str(segment.get("end") or "").strip()
    if not start or not end:
        return None
    return start, end


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir or repo_root / "reports")
    universe_slug = "_".join(args.universe_index)

    full_path = Path(args.json_full) if args.json_full else None
    bear_path = Path(args.json_bear) if args.json_bear else None
    bull_path = Path(args.json_bull) if args.json_bull else None
    diagnosis_path = Path(args.diagnosis_json) if args.diagnosis_json else None

    rule_slug = _rule_slug(args)

    if args.run_sweep:
        diagnosis_path = diagnosis_path or output_dir / (
            f"regime_weight_diagnosis_{universe_slug}_{rule_slug}_{args.start_date}_{args.end_date}.json"
        )
        diagnosis = _run_diagnosis(repo_root, args=args, output_json=diagnosis_path)
        if not diagnosis.get("passed"):
            print(json.dumps({"passed": False, "reasons": diagnosis.get("reasons") or ["Diagnosis stop."]}, ensure_ascii=False, indent=2))
            return 1

        bear_window = segment_window(diagnosis, "bear")
        bull_window = segment_window(diagnosis, "bull")
        if bear_window is None or bull_window is None:
            raise SystemExit("Diagnosis missing primary bear/bull segment windows.")

        full_path = output_dir / (
            f"regime_weight_sweep_{universe_slug}_{rule_slug}_full_{args.start_date}_{args.end_date}.json"
        )
        bear_start, bear_end = bear_window
        bull_start, bull_end = bull_window
        bear_path = output_dir / (
            f"regime_weight_sweep_{universe_slug}_{rule_slug}_algo_bear_{bear_start}_{bear_end}.json"
        )
        bull_path = output_dir / (
            f"regime_weight_sweep_{universe_slug}_{rule_slug}_algo_bull_{bull_start}_{bull_end}.json"
        )

        _run_sweep(repo_root, output_json=full_path, args=args, start_date=args.start_date, end_date=args.end_date)
        _run_sweep(repo_root, output_json=bear_path, args=args, start_date=bear_start, end_date=bear_end)
        _run_sweep(repo_root, output_json=bull_path, args=args, start_date=bull_start, end_date=bull_end)

    if full_path is None:
        raise SystemExit("Provide --json-full or use --run-sweep.")

    full_payload = load_sweep_payload(full_path)
    bear_rows: Optional[List[Dict[str, Any]]] = None
    bull_rows: Optional[List[Dict[str, Any]]] = None
    if bear_path and bear_path.exists():
        bear_rows = load_sweep_payload(bear_path).get("rows") or []
    if bull_path and bull_path.exists():
        bull_rows = load_sweep_payload(bull_path).get("rows") or []

    diagnosis_payload = None
    if diagnosis_path and diagnosis_path.exists():
        diagnosis_payload = load_diagnosis_payload(diagnosis_path)

    decision = evaluate_regime_weight_ab(
        full_payload.get("rows") or [],
        bear_rows=bear_rows,
        bull_rows=bull_rows,
        full_payload=full_payload,
    )
    out = {
        "passed": decision.passed,
        "reasons": decision.reasons,
        "comparisons": decision.comparisons,
        "full_window": decision.full_window,
        "bear_window": decision.bear_window,
        "bull_window": decision.bull_window,
        "diagnosis": {
            "path": str(diagnosis_path) if diagnosis_path else None,
            "regime_rule": (diagnosis_payload or {}).get("regime_rule"),
            "primary_bear_segment": (diagnosis_payload or {}).get("primary_bear_segment"),
            "primary_bull_segment": (diagnosis_payload or {}).get("primary_bull_segment"),
            "regime_segments": (diagnosis_payload or {}).get("regime_segments"),
        },
        "sources": {
            "full": str(full_path),
            "algo_bear": str(bear_path) if bear_path else None,
            "algo_bull": str(bull_path) if bull_path else None,
        },
    }
    text = json.dumps(out, ensure_ascii=False, indent=2, default=str)
    print(text)
    if args.gate_output:
        Path(args.gate_output).write_text(text, encoding="utf-8")
    return 0 if decision.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
