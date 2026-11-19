#!/usr/bin/env python3
"""Phase-0 diagnosis for index-regime growth/cycle weight switching.

Labels the benchmark index with look-ahead-free MA60/MA120 hysteresis and
reports state shares, flip rate, and algorithm-derived contiguous segments.
No calendar bull/bear priors are used.

Example:
    python -m analysis.research.regime_weight_diagnosis \
        --start-date 20230101 --end-date 20260501 \
        --universe-index csi1000 \
        --json-output reports/regime_weight_diagnosis_csi1000.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from analysis.core.benchmark_prices import load_index_benchmark_prices
from shared.regime_label import (
    CONFIRM_DAYS_DEFAULT,
    count_regime_flips,
    extract_regime_segments,
    label_index_regime,
    load_regime_rule,
    longest_regime_run,
    longest_segment_for_label,
    padded_index_start,
    regime_distribution_in_window,
)
from analysis.reporting.portfolio_comparison_report import default_index_benchmark_symbol

MAX_FLIPS_PER_YEAR = 6.0
MIN_STATE_SHARE = 0.10


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose index-regime labels for growth/cycle switching.")
    parser.add_argument("--start-date", required=True, help="Evaluation window start YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="Evaluation window end YYYYMMDD")
    parser.add_argument("--universe-index", default="csi1000", help="Universe label for default benchmark")
    parser.add_argument("--benchmark", help="Override benchmark ts_code (default from universe)")
    parser.add_argument("--confirm-days", type=int, default=CONFIRM_DAYS_DEFAULT)
    parser.add_argument(
        "--regime-rule",
        default=None,
        help="Preset id (ma120_anchor, ma60_price, ma60_symmetric), JSON path, or inline JSON",
    )
    parser.add_argument("--json-output", help="Optional JSON report path")
    return parser.parse_args()


def _share(dist: Dict[str, int], key: str) -> float:
    total = sum(dist.values())
    if total <= 0:
        return 0.0
    return float(dist.get(key, 0)) / total


def _years_in_window(start_date: str, end_date: str, trading_days: int) -> float:
    if trading_days <= 0:
        return 0.0
    start_year = int(str(start_date)[:4])
    end_year = int(str(end_date)[:4])
    span_years = max(end_year - start_year + 1, 1)
    return trading_days / (252.0 * span_years)


def evaluate_regime_diagnosis(
    labels_frame,
    *,
    start_date: str,
    end_date: str,
    confirm_days: int = CONFIRM_DAYS_DEFAULT,
) -> Dict[str, Any]:
    dist = regime_distribution_in_window(labels_frame, start_date=start_date, end_date=end_date)
    trading_days = sum(dist.values())
    flips = count_regime_flips(labels_frame, start_date=start_date, end_date=end_date)
    years = _years_in_window(start_date, end_date, trading_days)
    flips_per_year = flips / years if years > 0 else None
    longest = longest_regime_run(labels_frame, start_date=start_date, end_date=end_date)
    segments = extract_regime_segments(labels_frame, start_date=start_date, end_date=end_date)
    primary_bear = longest_segment_for_label(segments, "bear")
    primary_bull = longest_segment_for_label(segments, "bull")

    ma120_ready = labels_frame[
        (labels_frame["trade_date"] >= str(start_date))
        & (labels_frame["trade_date"] <= str(end_date))
        & labels_frame["ma120"].notna()
    ]
    ma120_coverage = len(ma120_ready) / trading_days if trading_days else 0.0

    reasons: List[str] = []
    passed = True
    if trading_days == 0:
        passed = False
        reasons.append("No index trading days in evaluation window.")
    if ma120_coverage < 0.95:
        passed = False
        reasons.append(f"MA120 coverage {ma120_coverage:.1%} below 95% in window.")
    if flips_per_year is not None and flips_per_year > MAX_FLIPS_PER_YEAR:
        passed = False
        reasons.append(
            f"Flip rate {flips_per_year:.1f}/year exceeds {MAX_FLIPS_PER_YEAR:.0f}; switcher likely churns."
        )
    if _share(dist, "bull") < MIN_STATE_SHARE:
        passed = False
        reasons.append(f"Bull share {_share(dist, 'bull'):.1%} below {MIN_STATE_SHARE:.0%}.")
    if _share(dist, "bear") < MIN_STATE_SHARE:
        passed = False
        reasons.append(f"Bear share {_share(dist, 'bear'):.1%} below {MIN_STATE_SHARE:.0%}.")
    if not primary_bear:
        passed = False
        reasons.append("No algorithm bear segment found in window.")
    if not primary_bull:
        passed = False
        reasons.append("No algorithm bull segment found in window.")
    if passed:
        reasons.append("Regime labels show usable bull/bear coverage with moderate flip rate.")

    return {
        "passed": passed,
        "verdict": "proceed" if passed else "stop",
        "reasons": reasons,
        "window": [start_date, end_date],
        "confirm_days": confirm_days,
        "distribution": dist,
        "shares": {key: _share(dist, key) for key in ("bull", "bear", "uncertain")},
        "trading_days": trading_days,
        "flips": flips,
        "flips_per_year": flips_per_year,
        "longest_runs": longest,
        "ma120_coverage": ma120_coverage,
        "regime_segments": segments,
        "primary_bear_segment": primary_bear,
        "primary_bull_segment": primary_bull,
    }


def main() -> int:
    from shared.data_access import open_research_data_access_for_cli

    args = _parse_args()
    benchmark = str(args.benchmark or default_index_benchmark_symbol(args.universe_index)).strip()
    padded_start = padded_index_start(args.start_date)

    db = open_research_data_access_for_cli(enable_tushare=False)
    index_rows = load_index_benchmark_prices(db, benchmark, padded_start)
    if not index_rows:
        print(f"❌ No index prices for {benchmark} from {padded_start}")
        return 1

    labels = label_index_regime(index_rows, confirm_days=int(args.confirm_days), regime_rule=args.regime_rule)
    rule = load_regime_rule(args.regime_rule)
    report = evaluate_regime_diagnosis(
        labels,
        start_date=args.start_date,
        end_date=args.end_date,
        confirm_days=int(args.confirm_days),
    )
    report["regime_rule"] = rule.to_dict()
    report["benchmark"] = benchmark
    report["universe_index"] = args.universe_index
    report["padded_start"] = padded_start

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"📄 Wrote {out}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
