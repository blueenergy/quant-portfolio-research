#!/usr/bin/env python3
"""A/B: bull full TopN vs non-bull cash / half-TopN / half-TopN-slot.

Research CLI only. Not wired to worker, dashboard, or live.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def summarize(path: Path, title: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    print(
        f"\n=== {title}  combos={len(rows)}  "
        f"index_bh={payload.get('index_buy_hold_return')} ===",
        flush=True,
    )
    print(
        f"{'regime':<22} {'non_bull':<16} {'vol':<18} {'mode':<18} "
        f"{'cum':>8} {'mdd':>8} {'sharpe':>7} {'exs':>8} {'exp':>6} {'sel':>5}  yearly",
        flush=True,
    )
    wanted = []
    for row in rows:
        if int(row.get("top_n") or 0) != 10:
            continue
        mode = str(row.get("construction_mode") or "")
        if mode not in {"top_n", "industry_neutral"}:
            continue
        wanted.append(row)
    wanted.sort(
        key=lambda row: (
            str(row.get("regime_mode")),
            str(row.get("non_bull_action")),
            str(row.get("volatility_variant")),
            str(row.get("construction_mode")),
        )
    )
    for row in wanted:
        yearly = row.get("yearly_returns") or {}
        ytxt = " ".join(
            f"{key}:{float(value)*100:+.1f}%" for key, value in sorted(yearly.items())
        )

        def pct(key: str) -> str:
            val = row.get(key)
            return f"{float(val)*100:+.1f}%" if val is not None else "-"

        sharpe = row.get("sharpe")
        sharpe_txt = f"{float(sharpe):.2f}" if sharpe is not None else "-"
        sel = row.get("average_selected_count")
        sel_txt = f"{float(sel):.1f}" if sel is not None else "-"
        print(
            f"{str(row.get('regime_mode') or '-'):<22} "
            f"{str(row.get('non_bull_action') or '-'):<16} "
            f"{str(row.get('volatility_variant') or '-'):<18} "
            f"{str(row.get('construction_mode') or '-'):<18} "
            f"{pct('cumulative_return'):>8} {pct('max_drawdown'):>8} {sharpe_txt:>7} "
            f"{pct('index_excess_cumulative_return'):>8} {pct('average_exposure'):>6} "
            f"{sel_txt:>5}  {ytxt}",
            flush=True,
        )


def run(start: str, end: str, out_json: Path) -> None:
    cmd = [
        sys.executable,
        "analysis/sweep/portfolio_sweep_report.py",
        "--start-date",
        start,
        "--end-date",
        end,
        "--universe-index",
        "csi1000",
        "--growth-cycle-weights",
        "60:40",
        "--top-n",
        "10",
        "--horizon",
        "20",
        "--rebalance-interval-days",
        "20",
        "--trailing-stop-pct",
        "0.15",
        "--regime-modes",
        "off",
        "bull_g60_else_cash",
        "--non-bull-actions",
        "cash",
        "half_topn",
        "half_topn_slot",
        "--volatility-modes",
        "off",
        "renormalize",
        "--target-vols",
        "0.50",
        "--active-caps",
        "0.25",
        "--json-output",
        str(out_json),
        "--output",
        str(out_json.with_suffix(".html")),
    ]
    print(">>", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)
    summarize(out_json, f"{start}-{end}")


def main() -> int:
    run("20230101", "20260822", Path("/tmp/half_topn_2023.json"))
    run("20210104", "20260821", Path("/tmp/half_topn_2021.json"))
    print("\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
