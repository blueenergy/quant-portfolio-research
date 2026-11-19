#!/usr/bin/env python3
"""Phase-0 IC / ICIR for raw Tushare elg+lg money flow — not money_flow_score.

EOD ``quant_data.money_flow`` (from 20240102) is joined onto the ScoreBacktester
panel at ``score_date == trade_date``. Forward returns stay next-open, so this
does not use same-day close-to-close against same-day flow.

Primary column is turnover-scaled ``main_net_to_amt``. Raw ``main_net`` is a
size-bias control. ``elg_net_to_amt`` asks whether extra-large adds anything.
``mainflow20`` is the 20-day trailing ratio already used as a risk overlay.

Example:
    python analysis/research/moneyflow_elg_lg_ic.py \
        --start-date 20240102 --end-date 20260901 \
        --universe-index csi1000 --horizons 1 5 10 20 \
        --output reports/moneyflow_elg_lg_ic_csi1000.json
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from analysis.core.score_backtester import BacktestConfig, ScoreBacktester
from analysis.factor.ic_metrics import (
    per_date_rank_ic,
    per_year_ic,
    quantile_monotonicity,
    return_col,
    summarize_ic,
)

MIN_MONEY_FLOW_DATE = "20240102"
FLOW_WINDOW = 20
FLOW_MIN_PERIODS = 10
FLOW_LOOKBACK_CAL_DAYS = 45
IN_CHUNK = 250
COVERAGE_FLOOR = 0.80
IC_ABS_FLOOR = 0.03
ICIR_FLOOR = 0.5

FACTOR_COLUMNS = (
    "main_net",
    "main_net_to_amt",
    "elg_net_to_amt",
    "mainflow20",
)

MONEY_FLOW_PROJECTION = {
    "_id": 0,
    "symbol": 1,
    "ts_code": 1,
    "trade_date": 1,
    "buy_elg_amount": 1,
    "sell_elg_amount": 1,
    "buy_lg_amount": 1,
    "sell_lg_amount": 1,
    "buy_md_amount": 1,
    "buy_sm_amount": 1,
    "amount": 1,
}


def bare_symbol(symbol: Any) -> str:
    text = str(symbol or "").strip()
    return text.split(".")[0] if text else ""


def to_ts_symbol(symbol: Any) -> str:
    text = str(symbol or "").strip()
    if not text:
        return ""
    if "." in text:
        return text
    if text.startswith(("60", "68", "51")):
        return f"{text}.SH"
    if text.startswith(("00", "30")):
        return f"{text}.SZ"
    if text.startswith(("8", "4")):
        return f"{text}.BJ"
    return f"{text}.SZ"


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _norm_yyyymmdd(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y%m%d")
        except Exception:
            return ""
    text = str(value).replace("-", "").strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if len(text) >= 8 and text[:8].isdigit():
        return text[:8]
    return text


def money_flow_row_metrics(row: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Compute one-day elg/lg metrics from a money_flow document."""
    buy_elg = _to_float(row.get("buy_elg_amount"))
    sell_elg = _to_float(row.get("sell_elg_amount"))
    buy_lg = _to_float(row.get("buy_lg_amount"))
    sell_lg = _to_float(row.get("sell_lg_amount"))
    has_elg_lg = any(v is not None for v in (buy_elg, sell_elg, buy_lg, sell_lg))
    elg_net = ((buy_elg or 0.0) - (sell_elg or 0.0)) if has_elg_lg else None
    main_net = (
        ((buy_elg or 0.0) + (buy_lg or 0.0) - (sell_elg or 0.0) - (sell_lg or 0.0))
        if has_elg_lg
        else None
    )
    buy_md = _to_float(row.get("buy_md_amount"))
    buy_sm = _to_float(row.get("buy_sm_amount"))
    turnover = None
    buy_parts = [buy_elg, buy_lg, buy_md, buy_sm]
    if any(v is not None for v in buy_parts):
        turnover = float(sum(v or 0.0 for v in buy_parts))
    if turnover is None or turnover <= 0:
        stored = _to_float(row.get("amount"))
        if stored is not None and stored > 0:
            turnover = stored
    main_net_to_amt = None
    elg_net_to_amt = None
    if main_net is not None and turnover is not None and turnover > 0:
        main_net_to_amt = main_net / turnover
    if elg_net is not None and turnover is not None and turnover > 0:
        elg_net_to_amt = elg_net / turnover
    return {
        "elg_net": elg_net,
        "main_net": main_net,
        "turnover_wan": turnover,
        "main_net_to_amt": main_net_to_amt,
        "elg_net_to_amt": elg_net_to_amt,
    }


def money_flow_docs_to_frame(
    docs: Sequence[Dict[str, Any]],
    *,
    flow_window: int = FLOW_WINDOW,
    min_periods: int = FLOW_MIN_PERIODS,
) -> pd.DataFrame:
    """Point-in-time daily factors plus right-aligned trailing mainflow20."""
    rows: List[Dict[str, Any]] = []
    for doc in docs:
        trade_date = _norm_yyyymmdd(doc.get("trade_date"))
        symbol = bare_symbol(doc.get("symbol") or doc.get("ts_code"))
        if not trade_date or not symbol:
            continue
        metrics = money_flow_row_metrics(doc)
        rows.append({"symbol": symbol, "trade_date": trade_date, **metrics})
    if not rows:
        return pd.DataFrame(
            columns=["symbol", "trade_date", *FACTOR_COLUMNS, "elg_net", "turnover_wan"]
        )
    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(["symbol", "trade_date"], keep="last")
    frame = frame.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    grouped_net = frame.groupby("symbol")["main_net"]
    grouped_to = frame.groupby("symbol")["turnover_wan"]
    num = grouped_net.transform(lambda s: s.rolling(flow_window, min_periods=min_periods).sum())
    den = grouped_to.transform(lambda s: s.rolling(flow_window, min_periods=min_periods).sum())
    frame["mainflow20"] = num / den.where(den > 0)
    return frame


def attach_money_flow_factors(dataset: pd.DataFrame, mf_frame: pd.DataFrame) -> pd.DataFrame:
    out = dataset.copy()
    out["symbol"] = out["symbol"].map(bare_symbol)
    out["score_date"] = out["score_date"].map(_norm_yyyymmdd)
    if mf_frame is None or mf_frame.empty:
        for col in FACTOR_COLUMNS:
            if col not in out.columns:
                out[col] = np.nan
        return out
    mf = mf_frame.copy()
    mf["symbol"] = mf["symbol"].map(bare_symbol)
    mf["trade_date"] = mf["trade_date"].map(_norm_yyyymmdd)
    keep = ["symbol", "trade_date", *FACTOR_COLUMNS]
    keep = [c for c in keep if c in mf.columns]
    mf = mf[keep].rename(columns={"trade_date": "score_date"})
    return out.merge(mf, on=["symbol", "score_date"], how="left")


def coverage_stats(dataset: pd.DataFrame, *, floor: float = COVERAGE_FLOOR) -> Dict[str, Any]:
    rows = int(len(dataset))
    dates = int(dataset["score_date"].nunique()) if rows else 0
    stats: Dict[str, Any] = {
        "rows": rows,
        "distinct_dates": dates,
        "score_date_range": (
            [str(dataset["score_date"].min()), str(dataset["score_date"].max())]
            if rows
            else [None, None]
        ),
        "floor": floor,
        "columns": {},
    }
    primary_ok = False
    for col in FACTOR_COLUMNS:
        if col not in dataset.columns:
            stats["columns"][col] = {"non_null": 0, "coverage": 0.0}
            continue
        non_null = int(pd.to_numeric(dataset[col], errors="coerce").notna().sum())
        coverage = (non_null / rows) if rows else 0.0
        stats["columns"][col] = {"non_null": non_null, "coverage": coverage}
        if col == "main_net_to_amt":
            primary_ok = coverage >= floor
    stats["primary_ok"] = primary_ok
    return stats


def add_industry_neutral(dataset: pd.DataFrame, score_col: str) -> str:
    neutral_col = f"{score_col}__ind_neutral"
    if "industry" not in dataset.columns:
        dataset[neutral_col] = dataset[score_col]
        return neutral_col
    industry = dataset["industry"].fillna("__UNKNOWN__").replace("", "__UNKNOWN__")
    tmp = dataset.assign(_industry=industry)
    dataset[neutral_col] = tmp.groupby(["score_date", "_industry"], dropna=False)[score_col].transform("mean")
    dataset[neutral_col] = dataset[score_col] - dataset[neutral_col]
    return neutral_col


def add_size_neutral(dataset: pd.DataFrame, score_col: str) -> Optional[str]:
    """Per-date residual of score_col on log(total_mv). None if no market cap."""
    if "total_mv" not in dataset.columns:
        return None
    mv = pd.to_numeric(dataset["total_mv"], errors="coerce")
    if int(mv.notna().sum()) == 0:
        return None
    log_mv = np.log(mv.where(mv > 0))
    residual = pd.Series(np.nan, index=dataset.index, dtype=float)
    y_all = pd.to_numeric(dataset[score_col], errors="coerce")
    for _, idx in dataset.groupby("score_date").groups.items():
        y = y_all.loc[idx]
        x = log_mv.loc[idx]
        mask = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x)
        if int(mask.sum()) < 5:
            continue
        yy = y[mask]
        xx = x[mask]
        x_demean = xx - xx.mean()
        var = float((x_demean ** 2).sum())
        if var <= 0:
            residual.loc[yy.index] = yy - yy.mean()
            continue
        beta = float((x_demean * (yy - yy.mean())).sum() / var)
        residual.loc[yy.index] = (yy - yy.mean()) - beta * x_demean
    col = f"{score_col}__size_neutral"
    dataset[col] = residual
    return col


def evaluate(
    dataset: pd.DataFrame,
    config: BacktestConfig,
    *,
    coverage: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if dataset.empty:
        return {"error": "empty dataset: no overlapping score records and prices found"}
    available = [c for c in FACTOR_COLUMNS if c in dataset.columns]
    report: Dict[str, Any] = {
        "rows": int(len(dataset)),
        "score_date_range": [str(dataset["score_date"].min()), str(dataset["score_date"].max())],
        "distinct_dates": int(dataset["score_date"].nunique()),
        "evaluated_columns": available,
        "coverage": coverage or coverage_stats(dataset),
        "factors": {},
    }
    for score_col in available:
        ind_col = add_industry_neutral(dataset, score_col)
        size_col = add_size_neutral(dataset, score_col)
        per_horizon: Dict[str, Any] = {}
        for horizon in config.horizons:
            ret = return_col(horizon)
            if ret not in dataset.columns:
                continue
            raw_ic = per_date_rank_ic(dataset, score_col, ret, config.min_names_per_date)
            block: Dict[str, Any] = {
                "raw": summarize_ic(raw_ic),
                "industry_neutral": summarize_ic(
                    per_date_rank_ic(dataset, ind_col, ret, config.min_names_per_date)
                ),
                "per_year": per_year_ic(raw_ic),
                "quantiles": quantile_monotonicity(
                    dataset, score_col, ret, config.quantiles, config.min_names_per_date
                ),
            }
            if size_col is not None:
                block["size_neutral"] = summarize_ic(
                    per_date_rank_ic(dataset, size_col, ret, config.min_names_per_date)
                )
            per_horizon[str(horizon)] = block
        report["factors"][score_col] = per_horizon
    report["verdict"] = _verdict(report)
    return report


def _passes(summary: Optional[Dict[str, Any]]) -> bool:
    if not summary:
        return False
    ic = summary.get("ic_mean")
    icir = summary.get("ic_ir")
    if ic is None or icir is None:
        return False
    return abs(float(ic)) > IC_ABS_FLOOR and float(icir) > ICIR_FLOOR


def _verdict(report: Dict[str, Any]) -> Dict[str, Any]:
    coverage = report.get("coverage") or {}
    if not coverage.get("primary_ok"):
        return {
            "label": "insufficient_coverage",
            "reason": "main_net_to_amt coverage below floor; do not interpret IC",
        }
    factors = report.get("factors") or {}
    scaled = factors.get("main_net_to_amt") or {}
    raw_amt = factors.get("main_net") or {}
    elg = factors.get("elg_net_to_amt") or {}
    trail = factors.get("mainflow20") or {}

    def _best_neutral(block: Dict[str, Any]) -> Dict[str, Any]:
        # Prefer size-neutral, then industry, then raw — same horizon keys.
        out = {}
        for horizon, payload in block.items():
            out[horizon] = payload.get("size_neutral") or payload.get("industry_neutral") or payload.get("raw")
        return out

    scaled_n = _best_neutral(scaled)
    raw_n = _best_neutral(raw_amt)
    elg_n = _best_neutral(elg)
    trail_n = _best_neutral(trail)
    short_h = [h for h in ("1", "5") if h in scaled_n]
    scaled_short = any(_passes(scaled_n.get(h)) for h in short_h) or (
        not short_h and any(_passes(v) for v in scaled_n.values())
    )
    raw_only = (not scaled_short) and any(_passes(v) for v in raw_n.values())
    trail_only = (not scaled_short) and any(_passes(v) for v in trail_n.values())
    elg_better = False
    if scaled_short:
        for h in scaled_n:
            s = scaled_n.get(h) or {}
            e = elg_n.get(h) or {}
            s_ic, e_ic = s.get("ic_mean"), e.get("ic_mean")
            s_ir, e_ir = s.get("ic_ir"), e.get("ic_ir")
            if None not in (s_ic, e_ic, s_ir, e_ir) and abs(float(e_ic)) > abs(float(s_ic)) and float(e_ir) > float(s_ir):
                elg_better = True
                break

    if scaled_short:
        label = "scaled_has_signal"
        reason = "turnover-scaled elg+lg passes |IC|>0.03 and ICIR>0.5 after neutralization on short horizons"
    elif raw_only:
        label = "size_contaminated"
        reason = "only unscaled main_net passes; treat as size/liquidity, not institutional flow"
    elif trail_only:
        label = "persistence_not_daily_flow"
        reason = "daily scaled IC dead; mainflow20 alive — likely momentum/persistence"
    else:
        label = "no_signal"
        reason = "naive DDE elg+lg has no usable IC; do not chase TopWin L2 reconstruction"
    return {
        "label": label,
        "reason": reason,
        "elg_adds_over_elg_lg": elg_better,
        "thresholds": {"abs_ic": IC_ABS_FLOOR, "icir": ICIR_FLOOR, "icir_is_daily": True},
    }


def _print_human_summary(report: Dict[str, Any]) -> None:
    if report.get("error"):
        print(f"[elg+lg] {report['error']}")
        return
    cov = report.get("coverage") or {}
    print(
        f"[elg+lg] rows={report['rows']} dates={report['distinct_dates']} "
        f"range={report['score_date_range']} primary_ok={cov.get('primary_ok')}"
    )
    for col, meta in (cov.get("columns") or {}).items():
        print(f"  coverage {col}: {meta.get('coverage'):.2%} ({meta.get('non_null')})")
    for score_col, horizons in (report.get("factors") or {}).items():
        print(f"\n=== {score_col} ===")
        print(
            f"{'h':>4} {'IC':>8} {'ICIR':>8} {'t':>7} {'pos%':>6} "
            f"{'IC(ind)':>9} {'IC(sz)':>8} {'spread':>9} {'mono':>5}"
        )
        for h, blk in horizons.items():
            raw = blk["raw"]
            neu = blk.get("industry_neutral") or {}
            size = blk.get("size_neutral") or {}
            q = blk.get("quantiles") or {}

            def _f(v, p=4):
                return f"{v:.{p}f}" if isinstance(v, (int, float)) else "  n/a"

            print(
                f"{h:>4} {_f(raw.get('ic_mean')):>8} {_f(raw.get('ic_ir')):>8} "
                f"{_f(raw.get('t_stat'), 2):>7} {_f(raw.get('positive_ratio'), 2):>6} "
                f"{_f(neu.get('ic_mean')):>9} {_f(size.get('ic_mean')):>8} "
                f"{_f(q.get('top_minus_bottom_spread')):>9} "
                f"{str(q.get('monotonic_increasing')):>5}"
            )
    verdict = report.get("verdict") or {}
    print(f"\n[verdict] {verdict.get('label')}: {verdict.get('reason')}")


def _query_symbols(symbols: Sequence[str]) -> List[str]:
    out = []
    seen = set()
    for symbol in symbols:
        for item in (bare_symbol(symbol), to_ts_symbol(symbol)):
            if item and item not in seen:
                seen.add(item)
                out.append(item)
    return out


def _iter_find(coll, query: Dict[str, Any], projection: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        cursor = coll.find(query, projection)
    except TypeError:
        cursor = coll.find(query)
    if hasattr(cursor, "batch_size"):
        try:
            cursor = cursor.batch_size(2000)
        except (TypeError, AttributeError):
            pass
    return list(cursor)


def load_money_flow_docs(
    db: Any,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
) -> List[Dict[str, Any]]:
    coll = getattr(db, "money_flow", None)
    if coll is None:
        inner = getattr(db, "_db", None)
        coll = getattr(inner, "money_flow", None) if inner is not None else None
    if coll is None:
        raise RuntimeError("db has no money_flow collection")
    wanted = _query_symbols(symbols)
    docs: List[Dict[str, Any]] = []
    date_query = {"$gte": start_date, "$lte": end_date}
    for i in range(0, len(wanted), IN_CHUNK):
        chunk = wanted[i : i + IN_CHUNK]
        docs.extend(
            _iter_find(
                coll,
                {"symbol": {"$in": chunk}, "trade_date": date_query},
                MONEY_FLOW_PROJECTION,
            )
        )
    return docs


def load_total_mv(
    db: Any,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    coll = getattr(db, "daily_basic", None)
    if coll is None:
        inner = getattr(db, "_db", None)
        coll = getattr(inner, "daily_basic", None) if inner is not None else None
    if coll is None:
        finder = getattr(db, "find_daily_basic_by_symbol", None)
        if finder is None:
            return pd.DataFrame(columns=["symbol", "score_date", "total_mv"])
        rows: List[Dict[str, Any]] = []
        for symbol in symbols:
            for doc in finder(
                str(symbol),
                start_date,
                end_date,
                {"trade_date": 1, "total_mv": 1, "ts_code": 1, "_id": 0},
            ) or []:
                rows.append(
                    {
                        "symbol": bare_symbol(doc.get("ts_code") or symbol),
                        "score_date": _norm_yyyymmdd(doc.get("trade_date")),
                        "total_mv": _to_float(doc.get("total_mv")),
                    }
                )
        return pd.DataFrame(rows).drop_duplicates(["symbol", "score_date"], keep="last") if rows else pd.DataFrame(
            columns=["symbol", "score_date", "total_mv"]
        )
    wanted = _query_symbols(symbols)
    docs: List[Dict[str, Any]] = []
    date_query = {"$gte": start_date, "$lte": end_date}
    for i in range(0, len(wanted), IN_CHUNK):
        chunk = wanted[i : i + IN_CHUNK]
        docs.extend(
            _iter_find(
                coll,
                {"ts_code": {"$in": chunk}, "trade_date": date_query},
                {"_id": 0, "ts_code": 1, "symbol": 1, "trade_date": 1, "total_mv": 1},
            )
        )
        docs.extend(
            _iter_find(
                coll,
                {"symbol": {"$in": [bare_symbol(s) for s in chunk]}, "trade_date": date_query},
                {"_id": 0, "ts_code": 1, "symbol": 1, "trade_date": 1, "total_mv": 1},
            )
        )
    rows = []
    for doc in docs:
        symbol = bare_symbol(doc.get("symbol") or doc.get("ts_code"))
        score_date = _norm_yyyymmdd(doc.get("trade_date"))
        if not symbol or not score_date:
            continue
        rows.append({"symbol": symbol, "score_date": score_date, "total_mv": _to_float(doc.get("total_mv"))})
    if not rows:
        return pd.DataFrame(columns=["symbol", "score_date", "total_mv"])
    return pd.DataFrame(rows).drop_duplicates(["symbol", "score_date"], keep="last")


def _lookback_start(start_date: str) -> str:
    ts = pd.Timestamp(start_date)
    return (ts - pd.Timedelta(days=FLOW_LOOKBACK_CAL_DAYS)).strftime("%Y%m%d")


def attach_from_db(dataset: pd.DataFrame, db: Any) -> pd.DataFrame:
    if dataset.empty:
        return dataset
    start = str(dataset["score_date"].min())
    end = str(dataset["score_date"].max())
    symbols = sorted({bare_symbol(s) for s in dataset["symbol"].tolist() if bare_symbol(s)})
    print(
        f"[elg+lg] loading money_flow for {len(symbols)} symbols "
        f"{_lookback_start(start)}-{end}",
        flush=True,
    )
    docs = load_money_flow_docs(db, symbols, _lookback_start(start), end)
    print(f"[elg+lg] money_flow docs={len(docs)}", flush=True)
    mf_frame = money_flow_docs_to_frame(docs)
    out = attach_money_flow_factors(dataset, mf_frame)
    mv = load_total_mv(db, symbols, start, end)
    if not mv.empty:
        out = out.merge(mv, on=["symbol", "score_date"], how="left")
        print(f"[elg+lg] total_mv rows={int(out['total_mv'].notna().sum())}", flush=True)
    else:
        print("[elg+lg] total_mv unavailable; size-neutral skipped", flush=True)
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase-0 Rank IC / ICIR for raw elg+lg money flow (not money_flow_score)."
    )
    parser.add_argument("--start-date", default=MIN_MONEY_FLOW_DATE)
    parser.add_argument("--end-date", help="Inclusive score_date upper bound, YYYYMMDD")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--min-names", type=int, default=20)
    parser.add_argument("--universe-index", "--universe-index-codes", nargs="+", default=["csi1000"])
    parser.add_argument("--index-codes", nargs="+", default=[])
    parser.add_argument("--coverage-only", action="store_true", help="Stop after coverage stats")
    parser.add_argument("--coverage-floor", type=float, default=COVERAGE_FLOOR)
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument("--enable-tushare", action="store_true")
    return parser.parse_args()


def main() -> int:
    from shared.data_access import open_research_data_access_for_cli

    args = _parse_args()
    start = max(str(args.start_date or MIN_MONEY_FLOW_DATE), MIN_MONEY_FLOW_DATE)
    if args.start_date and str(args.start_date) < MIN_MONEY_FLOW_DATE:
        print(f"[elg+lg] clamping start-date {args.start_date} -> {MIN_MONEY_FLOW_DATE}", flush=True)
    config = BacktestConfig(
        horizons=tuple(args.horizons),
        quantiles=args.quantiles,
        start_date=start,
        end_date=args.end_date,
        index_codes=tuple(args.index_codes),
        universe_index_codes=tuple(args.universe_index),
        min_names_per_date=args.min_names,
    )
    db = open_research_data_access_for_cli(enable_tushare=args.enable_tushare)
    print(
        f"[elg+lg] building score panel {start}-{args.end_date or 'max'} "
        f"universe={','.join(config.universe_index_codes)}",
        flush=True,
    )
    dataset = ScoreBacktester(db, config).build_dataset()
    dataset = attach_from_db(dataset, db)
    coverage = coverage_stats(dataset, floor=args.coverage_floor)
    if args.coverage_only:
        report = {
            "rows": int(len(dataset)),
            "score_date_range": coverage["score_date_range"],
            "distinct_dates": coverage["distinct_dates"],
            "evaluated_columns": [],
            "coverage": coverage,
            "verdict": {
                "label": "coverage_only",
                "reason": "stopped before IC as requested",
            },
        }
        _print_human_summary({**report, "factors": {}})
        print(json.dumps({"coverage": coverage}, ensure_ascii=False, indent=2))
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                json.dump(report, fh, ensure_ascii=False, indent=2)
        return 0 if coverage.get("primary_ok") else 2

    report = evaluate(dataset, config, coverage=coverage)
    _print_human_summary(report)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"\n[elg+lg] written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
