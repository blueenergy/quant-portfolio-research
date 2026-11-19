#!/usr/bin/env python3
"""把 stock_scores 里某个已有因子列导成 cycle_score 形状的 overlay 文件，可选反号。

只读、不写库：用于 portfolio_sweep_report.py --cycle-score-source 的 A/B 对照。
例如把 technical_score 反号(100 - x)当作 cycle 列叠进 growth/cycle sweep，
即可在 Top-N 组合层面对比「反号 technical」与旧 cycle 的头部择股能力。

因子取值与日期归一化完全复用 ScoreBacktester 的逻辑（top-level <name>_score，
回退 scores[name]；score_date -> YYYYMMDD），保证与 sweep 数据集逐键一致。

Example:
    python analysis/precompute_factor_overlay.py \
        --start-date 20250101 --end-date 20260501 \
        --universe-index csi1000 --source-factor technical_score --reversal \
        --out reports/revtech_overlay_csi1000.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export an existing stock_scores factor as a cycle_score overlay file.")
    p.add_argument("--start-date", required=True, help="Inclusive score_date lower bound, YYYYMMDD")
    p.add_argument("--end-date", required=True, help="Inclusive score_date upper bound, YYYYMMDD")
    p.add_argument("--universe-index", nargs="+", default=["csi1000"], help="Index universe label(s), e.g. csi1000/csi2000/hs300/csi500")
    p.add_argument("--source-factor", default="technical_score", help="stock_scores 中的源因子列，如 technical_score")
    p.add_argument("--reversal", action="store_true", help="对源因子取反号(100 - x)")
    p.add_argument("--out", required=True, help="Output path (.parquet or .csv[.gz])")
    return p.parse_args()


def main() -> int:
    from analysis.core.score_backtester import BacktestConfig, ScoreBacktester
    from shared.data_access import open_research_data_access_for_cli
    from utils.dates import normalize_date

    args = _parse_args()
    start = normalize_date(args.start_date)
    end = normalize_date(args.end_date)
    name = args.source_factor[:-6] if args.source_factor.endswith("_score") else args.source_factor

    db = open_research_data_access_for_cli(enable_tushare=False)
    bt = ScoreBacktester(db, BacktestConfig())  # 仅复用 _normalize_date/_extract_score/_to_float

    universe: Set[str] = set()
    for code in args.universe_index:
        try:
            universe.update(db.get_latest_index_constituent_symbols(code) or [])
        except Exception as exc:
            print(f"⚠️ 取 {code} 成分失败: {exc}")
    if not universe:
        print(f"❌ 股票池为空: {args.universe_index}")
        return 1

    query = {"score_date": {"$gte": start, "$lte": end}}
    projection = {"_id": 0, "symbol": 1, "score_date": 1, "scores": 1, f"{name}_score": 1}
    try:
        docs = list(db.stock_scores.find(query, projection))
    except Exception as exc:
        print(f"❌ 读取 stock_scores 失败: {exc}")
        return 1

    rows: List[Dict[str, object]] = []
    n_total = n_universe = n_valid = 0
    for doc in docs:
        n_total += 1
        symbol = doc.get("symbol")
        if symbol not in universe:
            continue
        n_universe += 1
        score_date = bt._normalize_date(doc.get("score_date"))
        if not score_date:
            continue
        scores = doc.get("scores") if isinstance(doc.get("scores"), dict) else {}
        value = bt._extract_score(doc, scores, name)
        if value is None:
            continue
        n_valid += 1
        if args.reversal:
            value = 100.0 - value
        rows.append({"symbol": str(symbol), "score_date": str(score_date), "cycle_score": float(value)})

    print(
        f"源因子={args.source_factor} reversal={args.reversal} | docs={n_total} "
        f"in-universe={n_universe} valid={n_valid}",
        flush=True,
    )
    if not rows:
        print("❌ 无有效因子值，未写出文件")
        return 1

    df = (
        pd.DataFrame(rows)
        .dropna(subset=["cycle_score"])
        .drop_duplicates(["symbol", "score_date"], keep="last")
        .sort_values(["score_date", "symbol"])
        .reset_index(drop=True)
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        try:
            df.to_parquet(out_path, index=False)
        except Exception as exc:  # pragma: no cover - 缺 pyarrow 时退回 csv.gz
            fallback = out_path.with_suffix(".csv.gz")
            df.to_csv(fallback, index=False)
            print(f"⚠️ parquet 写出失败({exc})，已改写 {fallback}")
            out_path = fallback
    else:
        df.to_csv(out_path, index=False)
    print(f"📄 overlay 写入 {out_path}（rows={len(df)}, dates={df['score_date'].nunique()}, symbols={df['symbol'].nunique()}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
