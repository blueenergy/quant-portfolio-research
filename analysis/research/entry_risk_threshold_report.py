#!/usr/bin/env python3
"""Calibrate entry-risk (hipos120/mainflow20/distrib) review thresholds.

Read-only research script. It builds a historical candidate panel from score
records, attaches point-in-time high-position / money-flow features, and
compares fixed vs cross-sectional distrib thresholds against future 20d
outcomes. The output is meant to calibrate review labels, not strategy
selection.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.core.benchmark_prices import load_index_benchmark_prices
from analysis.core.entry_stock_factors import (
    attach_factors_asof as _attach_factors_asof,
    build_symbol_entry_factor_frame as _build_symbol_entry_factor_frame,
)
from analysis.core.score_backtester import BacktestConfig, ScoreBacktester
from analysis.reporting.html_report import fmt_num as _fmt_num
from analysis.reporting.html_report import fmt_pct as _fmt_pct
from analysis.reporting.html_report import html_document as _html_document

DEFAULT_ABS_THRESHOLDS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08)
DEFAULT_PERCENTILES = (0.70, 0.80, 0.85, 0.90, 0.95)
DEFAULT_HIPOS_MIN = (None, 0.90, 0.95, 0.97)
MONEYFLOW_PROJECTION = {
    "_id": 0,
    "symbol": 1,
    "ts_code": 1,
    "trade_date": 1,
    "buy_elg_amount": 1,
    "sell_elg_amount": 1,
    "buy_lg_amount": 1,
    "sell_lg_amount": 1,
    "net_mf_amount": 1,
}
PRICE_PROJECTION = {
    "_id": 0,
    "symbol": 1,
    "ts_code": 1,
    "trade_date": 1,
    "close": 1,
    "vol": 1,
    "volume": 1,
    "amount": 1,
}


@dataclass(frozen=True)
class RuleSpec:
    rule_type: str
    label: str
    threshold: float
    hipos_min: Optional[float]


def _to_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _safe_quantile(values: pd.Series, q: float) -> Optional[float]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return None
    return float(clean.quantile(q))


def _pct(value: Any) -> str:
    return _fmt_pct(value)


def _num(value: Any, digits: int = 4) -> str:
    return _fmt_num(value, digits)


def _signed_pct_points(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return "-"
    return f"{number * 100:+.2f}pp"


def _pct_number(value: Any) -> Optional[float]:
    number = _to_float(value)
    return None if number is None else number * 100


def _candidate_output_paths(output_dir: Path) -> Dict[str, Path]:
    return {
        "json": output_dir / "threshold_summary.json",
        "md": output_dir / "threshold_summary.md",
        "html": output_dir / "threshold_summary.html",
        "samples": output_dir / "threshold_samples.parquet",
    }


def _build_score_panel(args: argparse.Namespace, db: Any) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, Any]]]]:
    cache_path = Path(args.panel_cache) if args.panel_cache else None
    if cache_path and cache_path.exists() and not args.rebuild_panel:
        print(f"[panel] loading cache {cache_path}", flush=True)
        df = pd.read_parquet(cache_path)
        df["symbol"] = df["symbol"].astype(str)
        df["score_date"] = df["score_date"].astype(str)
        return df, {}

    cfg = BacktestConfig(
        horizons=(args.horizon,),
        start_date=args.start_date,
        end_date=args.end_date,
        universe_index_codes=tuple(args.universe_index),
    )
    bt = ScoreBacktester(db, cfg)
    print(
        f"[panel] building score panel {args.start_date}-{args.end_date} "
        f"universe={','.join(args.universe_index)} horizon={args.horizon}",
        flush=True,
    )
    df = bt.build_dataset()
    if df.empty:
        raise RuntimeError("score panel is empty")
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"[panel] cached -> {cache_path}", flush=True)
    return df, bt.price_cache


def _build_entry_factor_frame(
    db: Any,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    *,
    max_workers: int,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    total = len(symbols)
    if max_workers <= 1:
        for idx, symbol in enumerate(symbols, 1):
            frame = _build_symbol_entry_factor_frame(db, symbol, start_date, end_date)
            if frame is not None and not frame.empty:
                frames.append(frame)
            if idx == 1 or idx % 25 == 0 or idx == total:
                print(f"[moneyflow] {idx}/{total} symbols", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_build_symbol_entry_factor_frame, db, symbol, start_date, end_date): symbol
                for symbol in symbols
            }
            for idx, future in enumerate(as_completed(futures), 1):
                try:
                    frame = future.result()
                    if frame is not None and not frame.empty:
                        frames.append(frame)
                except Exception as exc:  # pragma: no cover - defensive logging
                    print(f"[moneyflow] failed {futures[future]}: {exc}", flush=True)
                if idx == 1 or idx % 25 == 0 or idx == total:
                    print(f"[moneyflow] {idx}/{total} symbols", flush=True)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _asof_index_return(
    index_rows: Sequence[Dict[str, Any]],
    entry_dates: Iterable[Any],
    exit_dates: Iterable[Any],
) -> List[Optional[float]]:
    frame = pd.DataFrame(index_rows)
    if frame.empty:
        return [None for _ in entry_dates]
    close_col = next((c for c in ("close", "close_price", "adj_close") if c in frame.columns), None)
    date_col = next((c for c in ("trade_date", "date") if c in frame.columns), None)
    if close_col is None or date_col is None:
        return [None for _ in entry_dates]
    frame = frame[[date_col, close_col]].copy()
    frame[date_col] = frame[date_col].astype(str)
    frame[close_col] = pd.to_numeric(frame[close_col], errors="coerce")
    frame = frame.dropna().sort_values(date_col).drop_duplicates(date_col, keep="last")
    closes = frame.set_index(date_col)[close_col].to_dict()
    ordered_dates = frame[date_col].tolist()

    def _close_asof(date_value: Any) -> Optional[float]:
        date = str(date_value or "")
        if date in closes:
            return float(closes[date])
        # Dates are YYYYMMDD strings; search backwards for nearest benchmark date.
        idx = np.searchsorted(ordered_dates, date, side="right") - 1
        if idx < 0:
            return None
        return float(closes[ordered_dates[idx]])

    returns: List[Optional[float]] = []
    for entry_date, exit_date in zip(entry_dates, exit_dates, strict=False):
        entry_close = _close_asof(entry_date)
        exit_close = _close_asof(exit_date)
        if entry_close is None or exit_close is None or entry_close <= 0:
            returns.append(None)
        else:
            returns.append(exit_close / entry_close - 1.0)
    return returns


def _future_max_drawdown(price_cache: Dict[str, List[Dict[str, Any]]], dataset: pd.DataFrame, horizon: int) -> List[Optional[float]]:
    exit_col = f"exit_date_{horizon}d"
    if not price_cache or exit_col not in dataset.columns:
        return [None] * len(dataset)
    cache: Dict[str, pd.DataFrame] = {}
    out: List[Optional[float]] = []
    for row in dataset.to_dict("records"):
        symbol = str(row.get("symbol") or "")
        entry_price = _to_float(row.get("entry_price"))
        entry_date = str(row.get("entry_date") or row.get("score_date") or "")
        exit_date = str(row.get(exit_col) or "")
        if not symbol or entry_price is None or entry_price <= 0 or not entry_date or not exit_date:
            out.append(None)
            continue
        if symbol not in cache:
            prices = pd.DataFrame(price_cache.get(symbol) or [])
            if prices.empty or "trade_date" not in prices.columns or "close" not in prices.columns:
                cache[symbol] = pd.DataFrame()
            else:
                prices = prices[["trade_date", "close"]].copy()
                prices["trade_date"] = prices["trade_date"].astype(str)
                prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
                cache[symbol] = prices.dropna().sort_values("trade_date")
        prices = cache[symbol]
        if prices.empty:
            out.append(None)
            continue
        window = prices[(prices["trade_date"] >= entry_date) & (prices["trade_date"] <= exit_date)]
        if window.empty:
            out.append(None)
        else:
            out.append(float(window["close"].min() / entry_price - 1.0))
    return out


def _prepare_candidate_samples(
    args: argparse.Namespace,
    dataset: pd.DataFrame,
    factor_frame: pd.DataFrame,
    index_rows: Sequence[Dict[str, Any]],
    price_cache: Dict[str, List[Dict[str, Any]]],
) -> pd.DataFrame:
    ret_col = f"forward_return_{args.horizon}d"
    exit_col = f"exit_date_{args.horizon}d"
    required = [args.base_column, ret_col, "score_date", "symbol"]
    missing = [col for col in required if col not in dataset.columns]
    if missing:
        raise RuntimeError(f"dataset missing required columns: {missing}")

    df = dataset.dropna(subset=[args.base_column, ret_col]).copy()
    df = _attach_factors_asof(df, factor_frame)
    needed_factor_cols = ("hipos120", "mainflow20", "distrib")
    df = df.dropna(subset=[col for col in needed_factor_cols if col in df.columns])
    df["score_date"] = df["score_date"].astype(str)
    df["symbol"] = df["symbol"].astype(str)
    df["candidate_rank"] = df.groupby("score_date")[args.base_column].rank(method="first", ascending=False).astype(int)
    df = df[df["candidate_rank"] <= max(args.top_n)].copy()
    df["topn_bucket"] = ",".join(str(v) for v in args.top_n)
    df["benchmark_return"] = _asof_index_return(index_rows, df.get("entry_date", df["score_date"]), df.get(exit_col, df["score_date"]))
    df["excess_return"] = pd.to_numeric(df[ret_col], errors="coerce") - pd.to_numeric(df["benchmark_return"], errors="coerce")
    if price_cache:
        df["future_max_drawdown"] = _future_max_drawdown(price_cache, df, args.horizon)
    else:
        df["future_max_drawdown"] = np.nan
    df["mainflow_out"] = pd.to_numeric(df["mainflow20"], errors="coerce") < 0
    return df.reset_index(drop=True)


def _segment_mask(samples: pd.DataFrame, segment: str, bear_end: str) -> pd.Series:
    if segment == "ALL":
        return pd.Series(True, index=samples.index)
    if segment == "BEAR":
        return samples["score_date"] <= bear_end
    if segment == "BULL":
        return samples["score_date"] > bear_end
    raise ValueError(segment)


def _stats_for_frame(frame: pd.DataFrame, ret_col: str) -> Dict[str, Any]:
    returns = pd.to_numeric(frame[ret_col], errors="coerce").dropna()
    excess = pd.to_numeric(frame.get("excess_return"), errors="coerce").dropna()
    mdd = pd.to_numeric(frame.get("future_max_drawdown"), errors="coerce").dropna()
    distrib = pd.to_numeric(frame.get("distrib"), errors="coerce").dropna()
    mainflow = pd.to_numeric(frame.get("mainflow20"), errors="coerce").dropna()
    return {
        "count": int(len(frame)),
        "date_count": int(frame["score_date"].nunique()) if "score_date" in frame else 0,
        "mean_return": float(returns.mean()) if len(returns) else None,
        "median_return": float(returns.median()) if len(returns) else None,
        "tail5_return": float(np.nanpercentile(returns, 5)) if len(returns) else None,
        "negative_prob": float((returns < 0).mean()) if len(returns) else None,
        "mean_excess": float(excess.mean()) if len(excess) else None,
        "underperform_prob": float((excess < 0).mean()) if len(excess) else None,
        "mean_future_mdd": float(mdd.mean()) if len(mdd) else None,
        "median_distrib": float(distrib.median()) if len(distrib) else None,
        "p85_distrib": _safe_quantile(distrib, 0.85),
        "p90_distrib": _safe_quantile(distrib, 0.90),
        "p95_distrib": _safe_quantile(distrib, 0.95),
        "median_mainflow20": float(mainflow.median()) if len(mainflow) else None,
    }


def _rule_specs(args: argparse.Namespace) -> List[RuleSpec]:
    specs: List[RuleSpec] = []
    for hipos_min in args.hipos_min:
        suffix = "" if hipos_min is None else f" + hipos>={hipos_min:.2f}"
        for threshold in args.abs_thresholds:
            specs.append(RuleSpec("absolute", f"abs>={threshold:.3f}{suffix}", float(threshold), hipos_min))
        for percentile in args.percentiles:
            specs.append(RuleSpec("percentile", f"P{int(percentile * 100)}{suffix}", float(percentile), hipos_min))
    return specs


def _apply_rule(frame: pd.DataFrame, spec: RuleSpec) -> pd.Series:
    distrib = pd.to_numeric(frame["distrib"], errors="coerce")
    hit = (pd.to_numeric(frame["mainflow20"], errors="coerce") < 0) & (distrib > 0)
    if spec.hipos_min is not None:
        hit &= pd.to_numeric(frame["hipos120"], errors="coerce") >= float(spec.hipos_min)
    if spec.rule_type == "absolute":
        hit &= distrib >= spec.threshold
        return hit.fillna(False)

    # Cross-sectional threshold is computed within the candidate pool for each
    # score date, using positive distribution-risk observations.
    thresholds = frame.assign(_risk=distrib.where(distrib > 0)).groupby("score_date")["_risk"].transform(
        lambda s: s.dropna().quantile(spec.threshold) if s.dropna().size >= 5 else np.nan
    )
    hit &= distrib >= thresholds
    return hit.fillna(False)


def _calibrate(samples: pd.DataFrame, args: argparse.Namespace) -> Dict[str, Any]:
    ret_col = f"forward_return_{args.horizon}d"
    specs = _rule_specs(args)
    segments = ("ALL", "BEAR", "BULL")
    report: Dict[str, Any] = {
        "params": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "bear_end": args.bear_end,
            "universe_index": args.universe_index,
            "benchmark": args.benchmark,
            "base_column": args.base_column,
            "horizon": args.horizon,
            "top_n": args.top_n,
            "abs_thresholds": args.abs_thresholds,
            "percentiles": args.percentiles,
            "hipos_min": args.hipos_min,
        },
        "coverage": {
            col: float(samples[col].notna().mean())
            for col in ("hipos120", "mainflow20", "distrib", "excess_return", "future_max_drawdown")
            if col in samples.columns
        },
        "segments": {},
    }

    for top_n in args.top_n:
        top_frame = samples[samples["candidate_rank"] <= top_n].copy()
        top_key = f"top{top_n}"
        report["segments"][top_key] = {}
        for segment in segments:
            frame = top_frame[_segment_mask(top_frame, segment, args.bear_end)].copy()
            if frame.empty:
                continue
            baseline = _stats_for_frame(frame, ret_col)
            rules: List[Dict[str, Any]] = []
            for spec in specs:
                hit_mask = _apply_rule(frame, spec)
                hit = frame[hit_mask].copy()
                if hit.empty:
                    stats = _stats_for_frame(hit, ret_col)
                else:
                    stats = _stats_for_frame(hit, ret_col)
                stats.update(
                    {
                        "rule_type": spec.rule_type,
                        "label": spec.label,
                        "threshold": spec.threshold,
                        "hipos_min": spec.hipos_min,
                        "hit_rate": float(len(hit) / len(frame)) if len(frame) else None,
                        "mean_return_delta_vs_pool": (
                            None
                            if stats.get("mean_return") is None or baseline.get("mean_return") is None
                            else float(stats["mean_return"] - baseline["mean_return"])
                        ),
                        "mean_excess_delta_vs_pool": (
                            None
                            if stats.get("mean_excess") is None or baseline.get("mean_excess") is None
                            else float(stats["mean_excess"] - baseline["mean_excess"])
                        ),
                    }
                )
                rules.append(stats)
            report["segments"][top_key][segment] = {"baseline": baseline, "rules": rules}
    return report


def _best_warning_rows(report: Dict[str, Any], *, top_n: int = 20, segment: str = "ALL") -> List[Dict[str, Any]]:
    block = report.get("segments", {}).get(f"top{top_n}", {}).get(segment, {})
    rules = block.get("rules") or []
    filtered = [
        row
        for row in rules
        if (row.get("count") or 0) >= 30
        and row.get("hit_rate") is not None
        and 0.02 <= row["hit_rate"] <= 0.30
        and row.get("mean_return_delta_vs_pool") is not None
    ]
    return sorted(filtered, key=lambda row: (row.get("mean_return_delta_vs_pool") or 0, row.get("tail5_return") or 0))[:8]


def _rule_lookup(report: Dict[str, Any], top_key: str, segment: str, label: str) -> Optional[Dict[str, Any]]:
    block = report.get("segments", {}).get(top_key, {}).get(segment, {})
    for row in block.get("rules") or []:
        if row.get("label") == label:
            return row
    return None


def _baseline_lookup(report: Dict[str, Any], top_key: str, segment: str) -> Dict[str, Any]:
    return report.get("segments", {}).get(top_key, {}).get(segment, {}).get("baseline", {})


def _render_key_findings_markdown(report: Dict[str, Any]) -> List[str]:
    base = _baseline_lookup(report, "top20", "ALL")
    abs02 = _rule_lookup(report, "top20", "ALL", "abs>=0.020")
    abs05 = _rule_lookup(report, "top20", "ALL", "abs>=0.050")
    abs05_bull = _rule_lookup(report, "top20", "BULL", "abs>=0.050")
    p90_h97_bull = _rule_lookup(report, "top20", "BULL", "P90 + hipos>=0.97")
    if not base or not abs02 or not abs05:
        return []
    lines = [
        "## 核心发现",
        "",
        (
            f"- Top20 候选池基线：样本 {base.get('count')}，"
            f"平均未来20日收益 {_pct(base.get('mean_return'))}，"
            f"平均超额 {_pct(base.get('mean_excess'))}。"
        ),
        (
            f"- `distrib >= 0.02` 命中 {abs02.get('count')} 个样本"
            f"（{_pct(abs02.get('hit_rate'))}），未来收益 {_pct(abs02.get('mean_return'))}，"
            f"相对候选池 {_signed_pct_points(abs02.get('mean_return_delta_vs_pool'))}。"
            " 它更像“需要复核”的宽口径信号，不足以单独定义高风险。"
        ),
        (
            f"- `distrib >= 0.05` 命中 {abs05.get('count')} 个样本"
            f"（{_pct(abs05.get('hit_rate'))}），未来收益 {_pct(abs05.get('mean_return'))}，"
            f"相对候选池 {_signed_pct_points(abs05.get('mean_return_delta_vs_pool'))}，"
            f"跑输概率 {_pct(abs05.get('underperform_prob'))}。"
            " 这是更接近“警觉/高风险”的阈值。"
        ),
    ]
    if abs05_bull:
        lines.append(
            f"- 牛市段 `distrib >= 0.05` 的相对候选池收益差为 "
            f"{_signed_pct_points(abs05_bull.get('mean_return_delta_vs_pool'))}，"
            f"跑输概率 {_pct(abs05_bull.get('underperform_prob'))}，"
            "说明强势行情里追高派发风险更值得额外标红。"
        )
    if p90_h97_bull:
        lines.append(
            f"- `hipos120 >= 0.97` 且处在当日 `distrib` P90 以上时，牛市段相对收益差为 "
            f"{_signed_pct_points(p90_h97_bull.get('mean_return_delta_vs_pool'))}。"
            " 这支持把“固定阈值 + 候选池分位”组合成后续更稳的审核口径。"
        )
    lines.extend(
        [
            "- 建议当前前端/审核链路先维持：`<0.02` 观察，`0.02-0.05` 复核，"
            "`>=0.05` 警觉；后续若要自动降仓或替换标的，应先用 P90/P95 分位口径做组合 A/B。",
            "",
        ]
    )
    return lines


def _render_key_findings_html(report: Dict[str, Any]) -> str:
    findings = _render_key_findings_markdown(report)
    bullet_lines = [line for line in findings if line.startswith("- ")]
    if not bullet_lines:
        return ""
    items = "".join(f"<li>{escape(line[2:])}</li>" for line in bullet_lines)
    return f"<h2>核心发现</h2><ul>{items}</ul>"


def _table_html(rows: Sequence[Dict[str, Any]], table_id: str) -> str:
    headers = [
        ("label", "规则"),
        ("count", "样本"),
        ("hit_rate", "命中率"),
        ("mean_return", "未来收益"),
        ("mean_excess", "超额"),
        ("mean_return_delta_vs_pool", "相对池收益差"),
        ("underperform_prob", "跑输概率"),
        ("tail5_return", "左尾5%"),
        ("mean_future_mdd", "未来MDD"),
        ("median_distrib", "median distrib"),
        ("p90_distrib", "P90 distrib"),
    ]
    body_rows = []
    for row in rows:
        cells = []
        for key, _label in headers:
            value = row.get(key)
            if key == "label":
                text = escape(str(value))
                sort = escape(str(value))
            elif key in {"count"}:
                text = str(int(value or 0))
                sort = str(value or 0)
            elif key in {"hit_rate", "mean_return", "mean_excess", "mean_return_delta_vs_pool", "underperform_prob", "tail5_return", "mean_future_mdd"}:
                text = _pct(value)
                sort = "" if value is None else str(value)
            else:
                text = _num(value)
                sort = "" if value is None else str(value)
            cls = ""
            if key in {"mean_return", "mean_excess", "mean_return_delta_vs_pool", "tail5_return", "mean_future_mdd"}:
                numeric = _to_float(value)
                if numeric is not None:
                    cls = "pos" if numeric > 0 else "neg" if numeric < 0 else ""
            cells.append(f'<td class="{cls}" data-sort="{escape(sort)}">{text}</td>')
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    head = "".join(f'<th onclick="sortTable(\'{table_id}\',{idx})">{escape(label)}</th>' for idx, (_, label) in enumerate(headers))
    return f'<div class="table-wrap"><table id="{table_id}"><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>'


def _render_markdown(report: Dict[str, Any], output_paths: Dict[str, Path]) -> str:
    params = report["params"]
    lines = [
        "# 进场风险阈值标定报告",
        "",
        f"区间：`{params['start_date']}` - `{params['end_date']}`；bear split: `{params['bear_end']}`。",
        f"Universe: `{','.join(params['universe_index'])}`；score: `{params['base_column']}`；horizon: `{params['horizon']}d`。",
        "",
        "## 结论读法",
        "",
        "- 本报告重新计算 `hipos120 / mainflow20 / distrib`，其中日线成交额按 Tushare `amount=千元` 转为万元，和 `money_flow` 单位对齐。",
        "- `distrib = hipos120 * (-mainflow20)` 是风险方向指标；只有 `mainflow20 < 0` 且 `distrib > 0` 才代表高位派发风险。",
        "- 绝对阈值 `0.02/0.05` 仍应视为初始参考，最终建议优先看候选池分位阈值是否更稳定。",
        "",
    ]
    lines.extend(_render_key_findings_markdown(report))
    for top_key, segs in report.get("segments", {}).items():
        lines.extend([f"## {top_key}", ""])
        for segment, block in segs.items():
            base = block["baseline"]
            lines.extend(
                [
                    f"### {segment}",
                    "",
                    (
                        f"候选池样本 {base['count']}，日期 {base['date_count']}；"
                        f"平均未来收益 {_pct(base.get('mean_return'))}，"
                        f"平均超额 {_pct(base.get('mean_excess'))}，"
                        f"跑输概率 {_pct(base.get('underperform_prob'))}。"
                    ),
                    "",
                    "| 规则 | 样本 | 命中率 | 未来收益 | 超额 | 相对池收益差 | 跑输概率 | 左尾5% | 未来MDD |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            rows = sorted(
                block["rules"],
                key=lambda row: (
                    row.get("rule_type") or "",
                    999 if row.get("hipos_min") is None else float(row.get("hipos_min")),
                    float(row.get("threshold") or 0),
                ),
            )
            for row in rows:
                lines.append(
                    "| {label} | {count} | {hit_rate} | {mean_return} | {mean_excess} | {delta} | {under} | {tail5} | {mdd} |".format(
                        label=row["label"],
                        count=row.get("count", 0),
                        hit_rate=_pct(row.get("hit_rate")),
                        mean_return=_pct(row.get("mean_return")),
                        mean_excess=_pct(row.get("mean_excess")),
                        delta=_signed_pct_points(row.get("mean_return_delta_vs_pool")),
                        under=_pct(row.get("underperform_prob")),
                        tail5=_pct(row.get("tail5_return")),
                        mdd=_pct(row.get("mean_future_mdd")),
                    )
                )
            lines.append("")
    lines.extend(
        [
            "## 产物",
            "",
            f"- JSON: `{output_paths['json']}`",
            f"- HTML: `{output_paths['html']}`",
            f"- Samples: `{output_paths['samples']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _render_html(report: Dict[str, Any]) -> str:
    params = report["params"]
    body = [
        '<section class="hero">',
        '<p class="eyebrow">Entry Risk Threshold Calibration</p>',
        "<h1>进场风险阈值标定报告</h1>",
        (
            f"<p class=\"subtitle\">{escape(params['start_date'])} - {escape(params['end_date'])}，"
            f"universe={escape(','.join(params['universe_index']))}，horizon={params['horizon']}d</p>"
        ),
        "</section>",
        '<div class="warning"><h2>阅读提示</h2><p>'
        "本报告用于标定审核提示阈值，不改变策略选股。"
        "<code>distrib = hipos120 * (-mainflow20)</code>，只有主力净流出时才代表风险。"
        "固定阈值 0.02/0.05 是初始参考，需结合候选池分位与未来表现验证。"
        "</p></div>",
    ]
    key_findings = _render_key_findings_html(report)
    if key_findings:
        body.append(key_findings)
    for top_key, segs in report.get("segments", {}).items():
        body.append(f"<h2>{escape(top_key)}</h2>")
        for segment, block in segs.items():
            base = block["baseline"]
            body.append(f"<h3>{escape(segment)}</h3>")
            body.append(
                '<div class="summary-card">'
                f"<div><span class=\"label\">样本</span> <span class=\"big\">{base['count']}</span></div>"
                f"<div>平均未来收益 {_pct(base.get('mean_return'))}；平均超额 {_pct(base.get('mean_excess'))}；"
                f"跑输概率 {_pct(base.get('underperform_prob'))}</div>"
                "</div>"
            )
            rows = sorted(
                block["rules"],
                key=lambda row: (
                    row.get("rule_type") or "",
                    999 if row.get("hipos_min") is None else float(row.get("hipos_min")),
                    float(row.get("threshold") or 0),
                ),
            )
            body.append(_table_html(rows, f"{top_key}_{segment}".replace(" ", "_").replace(".", "_")))
    body.append('<p class="foot">Generated by analysis/entry_risk_threshold_report.py</p>')
    return _html_document("进场风险阈值标定报告", "\n".join(body))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate entry-risk distrib thresholds.")
    parser.add_argument("--start-date", default="20230101")
    parser.add_argument("--end-date", default="20260501")
    parser.add_argument("--bear-end", default="20240920")
    parser.add_argument("--universe-index", nargs="+", default=["csi1000"])
    parser.add_argument("--benchmark", default="000852.SH")
    parser.add_argument("--base-column", default="composite_growth_cycle_score")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--top-n", nargs="+", type=int, default=[20, 50])
    parser.add_argument("--abs-thresholds", nargs="+", type=float, default=list(DEFAULT_ABS_THRESHOLDS))
    parser.add_argument("--percentiles", nargs="+", type=float, default=list(DEFAULT_PERCENTILES))
    parser.add_argument(
        "--hipos-min",
        nargs="+",
        default=["none", "0.90", "0.95", "0.97"],
        help="hipos filters; use 'none' for no hipos filter",
    )
    parser.add_argument("--output-dir", default="reports/entry-risk-threshold-calibration")
    parser.add_argument("--panel-cache", default="reports/entry-risk-threshold-calibration/score_panel.parquet")
    parser.add_argument("--samples-cache", default="reports/entry-risk-threshold-calibration/threshold_samples.parquet")
    parser.add_argument("--rebuild-panel", action="store_true")
    parser.add_argument("--rebuild-samples", action="store_true")
    parser.add_argument("--mf-workers", type=int, default=1)
    parser.add_argument("--enable-tushare", action="store_true")
    args = parser.parse_args()
    normalized_hipos = []
    for value in args.hipos_min:
        text = str(value).strip().lower()
        normalized_hipos.append(None if text in {"none", "null", "-"} else float(text))
    args.hipos_min = normalized_hipos
    args.top_n = sorted(set(int(v) for v in args.top_n))
    return args


def main() -> int:
    from shared.data_access import open_research_data_access_for_cli

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _candidate_output_paths(output_dir)

    db = open_research_data_access_for_cli(enable_tushare=args.enable_tushare)
    index_rows = load_index_benchmark_prices(db, args.benchmark, args.start_date)

    samples_path = Path(args.samples_cache)
    if samples_path.exists() and not args.rebuild_samples:
        print(f"[samples] loading cache {samples_path}", flush=True)
        samples = pd.read_parquet(samples_path)
    else:
        score_panel, price_cache = _build_score_panel(args, db)
        max_top_n = max(args.top_n)
        score_panel = score_panel.dropna(subset=[args.base_column]).copy()
        score_panel["candidate_rank"] = score_panel.groupby("score_date")[args.base_column].rank(method="first", ascending=False).astype(int)
        top_panel = score_panel[score_panel["candidate_rank"] <= max_top_n].copy()
        symbols = sorted(top_panel["symbol"].astype(str).unique())
        mf_start = (datetime.strptime(args.start_date, "%Y%m%d") - timedelta(days=260)).strftime("%Y%m%d")
        print(f"[moneyflow] building entry factors for {len(symbols)} symbols {mf_start}-{args.end_date}", flush=True)
        factor_frame = _build_entry_factor_frame(db, symbols, mf_start, args.end_date, max_workers=args.mf_workers)
        if factor_frame.empty:
            raise RuntimeError("entry factor frame is empty")
        samples = _prepare_candidate_samples(args, top_panel, factor_frame, index_rows, price_cache)
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        samples.to_parquet(samples_path, index=False)
        print(f"[samples] cached -> {samples_path}", flush=True)

    print(
        f"[samples] rows={len(samples)} dates={samples['score_date'].nunique()} "
        f"symbols={samples['symbol'].nunique()} distrib_cov={samples['distrib'].notna().mean():.1%}",
        flush=True,
    )
    report = _calibrate(samples, args)
    report["notable_rules_top20_all"] = _best_warning_rows(report, top_n=20, segment="ALL")

    with open(paths["json"], "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=str)
    paths["md"].write_text(_render_markdown(report, paths), encoding="utf-8")
    paths["html"].write_text(_render_html(report), encoding="utf-8")
    print(f"[output] json -> {paths['json']}", flush=True)
    print(f"[output] md   -> {paths['md']}", flush=True)
    print(f"[output] html -> {paths['html']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
