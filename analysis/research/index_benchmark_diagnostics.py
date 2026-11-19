#!/usr/bin/env python3
"""Diagnose why periodic index benchmark returns differ from buy-and-hold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from analysis.core.benchmark_prices import load_index_benchmark_prices
from analysis.core.portfolio_backtester import PortfolioBacktester


def diagnose_index_benchmark(
    portfolio_payload: Dict[str, Any],
    index_price_history: Sequence[Dict[str, Any]],
    top_n: Optional[int] = None,
    horizon: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    periods = portfolio_payload.get("periods", [])
    if not periods:
        return {"error": "portfolio payload has no periods"}

    chosen_top_n = top_n if top_n is not None else _first_top_n(periods)
    selected_periods = [row for row in periods if int(row.get("top_n", -1)) == int(chosen_top_n)]
    selected_periods = sorted(selected_periods, key=lambda row: str(row.get("score_date", "")))
    if not selected_periods:
        return {"error": f"no periods found for top_n={chosen_top_n}"}

    chosen_horizon = horizon if horizon is not None else int(selected_periods[0].get("horizon"))
    cleaned_prices = PortfolioBacktester._clean_price_history(index_price_history)
    if not cleaned_prices:
        return {"error": "index price history is empty"}

    price_dates = [row["trade_date"] for row in cleaned_prices]
    windows: List[Dict[str, Any]] = []
    periodic_equity = 1.0
    stored_periodic_equity = 1.0

    for period in selected_periods:
        score_date = str(period.get("score_date"))
        start_idx = _first_index_on_or_after(price_dates, score_date)
        if start_idx is None:
            continue
        end_idx = start_idx + chosen_horizon
        if end_idx >= len(cleaned_prices):
            continue

        start_price = cleaned_prices[start_idx]
        end_price = cleaned_prices[end_idx]
        index_return = end_price["close"] / start_price["close"] - 1
        stored_return = _as_float(period.get("index_benchmark_return"))
        periodic_equity *= 1 + index_return
        if stored_return is not None:
            stored_periodic_equity *= 1 + stored_return

        windows.append(
            {
                "score_date": score_date,
                "start_date": start_price["trade_date"],
                "end_date": end_price["trade_date"],
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_close": start_price["close"],
                "end_close": end_price["close"],
                "index_return": index_return,
                "stored_index_benchmark_return": stored_return,
            }
        )

    if not windows:
        return {"error": "no valid benchmark windows"}

    for idx, window in enumerate(windows):
        if idx == 0:
            window["gap_from_previous_end_days"] = None
            window["overlap_with_previous_days"] = None
            window["bridge_return_from_previous_end"] = None
            continue
        previous = windows[idx - 1]
        gap = window["start_idx"] - previous["end_idx"]
        window["gap_from_previous_end_days"] = max(0, gap)
        window["overlap_with_previous_days"] = max(0, -gap)
        window["bridge_return_from_previous_end"] = (
            window["start_close"] / previous["end_close"] - 1
            if previous["end_close"] and gap >= 0
            else None
        )

    buy_hold_start = start_date or windows[0]["start_date"]
    buy_hold_end = end_date or windows[-1]["end_date"]
    buy_hold = _buy_hold_return(cleaned_prices, buy_hold_start, buy_hold_end)
    bridge_returns = [
        row["bridge_return_from_previous_end"]
        for row in windows
        if row.get("bridge_return_from_previous_end") is not None
    ]
    bridge_equity = _compound(bridge_returns)
    total_gap_days = sum(row.get("gap_from_previous_end_days") or 0 for row in windows)
    total_overlap_days = sum(row.get("overlap_with_previous_days") or 0 for row in windows)
    covered_trade_days = len({idx for row in windows for idx in range(row["start_idx"], row["end_idx"] + 1)})
    buy_hold_window_indices = _window_indices(cleaned_prices, buy_hold_start, buy_hold_end)
    buy_hold_trade_days = len(buy_hold_window_indices)

    return {
        "top_n": chosen_top_n,
        "horizon": chosen_horizon,
        "period_count": len(windows),
        "first_window_start": windows[0]["start_date"],
        "last_window_end": windows[-1]["end_date"],
        "buy_hold_start": buy_hold.get("start_date"),
        "buy_hold_end": buy_hold.get("end_date"),
        "buy_hold_return": buy_hold.get("return"),
        "periodic_compound_return": periodic_equity - 1,
        "stored_periodic_compound_return": stored_periodic_equity - 1,
        "bridge_compound_return": bridge_equity - 1 if bridge_returns else None,
        "total_gap_days": total_gap_days,
        "total_overlap_days": total_overlap_days,
        "covered_trade_days": covered_trade_days,
        "buy_hold_trade_days": buy_hold_trade_days,
        "coverage_ratio": covered_trade_days / buy_hold_trade_days if buy_hold_trade_days else None,
        "top_positive_windows": sorted(windows, key=lambda row: row["index_return"], reverse=True)[:5],
        "top_negative_windows": sorted(windows, key=lambda row: row["index_return"])[:5],
        "largest_gaps": sorted(windows[1:], key=lambda row: row.get("gap_from_previous_end_days") or 0, reverse=True)[:5],
        "largest_overlaps": sorted(windows[1:], key=lambda row: row.get("overlap_with_previous_days") or 0, reverse=True)[:5],
        "windows": windows,
    }


def render_markdown(diagnostics: Dict[str, Any]) -> str:
    if diagnostics.get("error"):
        return f"# Index Benchmark Diagnostics\n\nError: {diagnostics['error']}\n"

    lines = ["# Index Benchmark Diagnostics", ""]
    lines.append(f"- TopN: {diagnostics['top_n']}")
    lines.append(f"- Horizon: {diagnostics['horizon']} trading days")
    lines.append(f"- Periods: {diagnostics['period_count']}")
    lines.append(f"- Buy-hold window: {diagnostics['buy_hold_start']} -> {diagnostics['buy_hold_end']}")
    lines.append(f"- First/last periodic window: {diagnostics['first_window_start']} -> {diagnostics['last_window_end']}")
    lines.append(f"- Buy-hold return: {_fmt_pct(diagnostics.get('buy_hold_return'))}")
    lines.append(f"- Periodic compound return: {_fmt_pct(diagnostics.get('periodic_compound_return'))}")
    lines.append(f"- Stored periodic compound return: {_fmt_pct(diagnostics.get('stored_periodic_compound_return'))}")
    lines.append(f"- Bridge compound return between non-overlapping windows: {_fmt_pct(diagnostics.get('bridge_compound_return'))}")
    lines.append(f"- Total gap days: {diagnostics['total_gap_days']}")
    lines.append(f"- Total overlap days: {diagnostics['total_overlap_days']}")
    lines.append(f"- Covered trade days: {diagnostics['covered_trade_days']} / {diagnostics['buy_hold_trade_days']} ({_fmt_pct(diagnostics.get('coverage_ratio'))})")
    lines.append("")
    lines.append("## Largest Positive Windows")
    lines.extend(_window_table(diagnostics["top_positive_windows"]))
    lines.append("")
    lines.append("## Largest Negative Windows")
    lines.extend(_window_table(diagnostics["top_negative_windows"]))
    lines.append("")
    lines.append("## Largest Gaps")
    lines.extend(_gap_table(diagnostics["largest_gaps"]))
    lines.append("")
    lines.append("## Largest Overlaps")
    lines.extend(_gap_table(diagnostics["largest_overlaps"]))
    lines.append("")
    return "\n".join(lines)


def _window_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    lines = ["| Score Date | Start | End | Return | Start Close | End Close |", "|---|---|---|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['score_date']} | {row['start_date']} | {row['end_date']} | "
            f"{_fmt_pct(row['index_return'])} | {row['start_close']:.2f} | {row['end_close']:.2f} |"
        )
    return lines


def _gap_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    lines = ["| Score Date | Start | End | Gap Days | Overlap Days | Bridge Return | Window Return |", "|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['score_date']} | {row['start_date']} | {row['end_date']} | "
            f"{row.get('gap_from_previous_end_days') or 0} | {row.get('overlap_with_previous_days') or 0} | "
            f"{_fmt_pct(row.get('bridge_return_from_previous_end'))} | {_fmt_pct(row['index_return'])} |"
        )
    return lines


def _first_top_n(periods: Sequence[Dict[str, Any]]) -> int:
    return int(sorted({int(row["top_n"]) for row in periods if row.get("top_n") is not None})[0])


def _first_index_on_or_after(price_dates: Sequence[str], target_date: str) -> Optional[int]:
    for idx, trade_date in enumerate(price_dates):
        if trade_date >= target_date:
            return idx
    return None


def _buy_hold_return(price_history: Sequence[Dict[str, Any]], start_date: str, end_date: str) -> Dict[str, Any]:
    window = [row for row in price_history if start_date <= row["trade_date"] <= end_date]
    if len(window) < 2:
        return {"start_date": None, "end_date": None, "return": None}
    return {
        "start_date": window[0]["trade_date"],
        "end_date": window[-1]["trade_date"],
        "return": window[-1]["close"] / window[0]["close"] - 1,
    }


def _window_indices(price_history: Sequence[Dict[str, Any]], start_date: str, end_date: str) -> List[int]:
    return [idx for idx, row in enumerate(price_history) if start_date <= row["trade_date"] <= end_date]


def _compound(returns: Sequence[float]) -> float:
    equity = 1.0
    for value in returns:
        equity *= 1 + value
    return equity


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_pct(value: Any) -> str:
    numeric = _as_float(value)
    return "-" if numeric is None else f"{numeric * 100:+.2f}%"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose periodic index benchmark compounding.")
    parser.add_argument("portfolio_json", help="Portfolio backtest JSON file")
    parser.add_argument("--index-benchmark-symbol", default="000852.SH", help="Index symbol, e.g. 000852.SH")
    parser.add_argument("--top-n", type=int, help="Top-N group to inspect; defaults to the smallest top_n in JSON")
    parser.add_argument("--horizon", type=int, help="Override horizon; defaults to JSON period horizon")
    parser.add_argument("--start-date", help="Buy-hold start date; defaults to first window start")
    parser.add_argument("--end-date", help="Buy-hold end date; defaults to last window end")
    parser.add_argument("--output", help="Markdown output path")
    parser.add_argument("--json-output", help="Optional diagnostics JSON output path")
    parser.add_argument("--enable-tushare", action="store_true", help="Keep TUSHARE_TOKEN enabled when creating ResearchDataAccess")
    return parser.parse_args()


def main() -> int:
    from shared.data_access import open_research_data_access_for_cli

    args = _parse_args()
    payload = json.loads(Path(args.portfolio_json).read_text(encoding="utf-8"))
    db_manager = open_research_data_access_for_cli(enable_tushare=args.enable_tushare)
    start = args.start_date or _infer_start_date(payload)
    prices = load_index_benchmark_prices(db_manager, args.index_benchmark_symbol, start)
    diagnostics = diagnose_index_benchmark(
        payload,
        prices,
        top_n=args.top_n,
        horizon=args.horizon,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    markdown = render_markdown(diagnostics)
    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
        print(f"Markdown diagnostics: {args.output}")
    else:
        print(markdown)

    if args.json_output:
        Path(args.json_output).write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON diagnostics: {args.json_output}")
    return 0


def _infer_start_date(payload: Dict[str, Any]) -> str:
    periods = payload.get("periods", [])
    dates = sorted(str(row.get("score_date")) for row in periods if row.get("score_date"))
    return dates[0] if dates else "00000000"


if __name__ == "__main__":
    raise SystemExit(main())
