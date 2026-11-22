#!/usr/bin/env python3
"""Consume portfolio research jobs from quant_trading.

The worker runs production-data portfolio sweeps, persists result summaries, and
materializes a candidate portfolio strategy config that quantFinance can publish.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

def _portfolio_research_report_dir() -> Path:
    """Resolved report root shared by worker writes and quant-api reads."""
    return Path(os.getenv("PORTFOLIO_RESEARCH_REPORT_DIR", "reports/portfolio-research")).expanduser().resolve()


def _artifact_local_refs(report_dir: Path, base: Path) -> Dict[str, str]:
    """Store artifact paths relative to PORTFOLIO_RESEARCH_REPORT_DIR for API resolution."""
    html_path = base.with_suffix(".html")
    json_path = base.with_suffix(".json")
    combos_dir = report_dir / f"{base.name}_combos"
    try:
        return {
            "html_path": str(html_path.relative_to(report_dir)),
            "json_path": str(json_path.relative_to(report_dir)),
            "combos_dir": str(combos_dir.relative_to(report_dir)),
        }
    except ValueError:
        return {
            "html_path": html_path.name,
            "json_path": json_path.name,
            "combos_dir": combos_dir.name,
        }

# Run before the first project import: as a script sys.path[0] is tools/, so
# ``utils`` would not resolve yet.
from shared.regime_label import REGIME_MODE_BULL_G60_ELSE_CASH
from shared.universe_benchmark import default_index_benchmark_symbol
from utils.job_queue import (
    HeartbeatTicker,
    JobQueue,
    WorkerLockHeld,
    clean_for_mongo,
    handle_shutdown_signal,
    install_shutdown_handlers,
    set_active_job,
    utcnow,
    worker_lock,
)

log = logging.getLogger(__name__)


def resolve_score_column(params: Dict[str, Any]) -> str:
    """Ranking column for a research job.

    ``model_id`` is the single source of truth. The API also forces
    ``score_column=model_score``, but a direct job insert with a stale
    ``score_column`` must not silently backtest rule scores while the model
    column is merged into the panel.
    """
    model_id = str(params.get("model_id") or "").strip()
    if model_id:
        return "model_score"
    return str(params.get("score_column") or "composite_growth_cycle_score")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process pending portfolio research jobs.")
    parser.add_argument("--worker-id", default="quant-scorer-portfolio-research", help="Worker id written to claimed jobs")
    parser.add_argument("--limit", type=int, default=3, help="Maximum jobs to process in this run")
    parser.add_argument("--poll-interval", type=float, default=0, help="Seconds to sleep between empty polls; 0 exits when empty")
    parser.add_argument(
        "--stale-running-minutes",
        type=int,
        default=int(os.getenv("PORTFOLIO_RESEARCH_STALE_MINUTES", "5")),
        help="Requeue running jobs whose heartbeat is older than this many minutes",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=float(os.getenv("PORTFOLIO_RESEARCH_HEARTBEAT_SECONDS", "30")),
        help="Background heartbeat interval so long stages keep a fresh heartbeat_at",
    )
    parser.add_argument(
        "--lock-file",
        default=os.getenv("PORTFOLIO_RESEARCH_LOCK_FILE", "/tmp/portfolio_research_worker.lock"),
        help="Exclusive lock file to prevent overlapping worker runs",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON payload")
    parser.add_argument("--verbose-empty", action="store_true", help="Print a summary even when no jobs were processed")
    return parser.parse_args()


WORKER_TYPE = "portfolio_research"
LOCK_LABEL = "portfolio research"


def _queue(collections) -> JobQueue:
    """Bind the shared queue mechanics to this worker's collection."""
    return JobQueue(
        collections.research_jobs,
        worker_type=WORKER_TYPE,
        worker_status=getattr(collections, "worker_status", None),
    )


def _utcnow() -> datetime:
    return utcnow()


def _as_list(value: Any, default: Sequence[Any]) -> List[Any]:
    if value is None or value == "":
        return list(default)
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [value]


def _as_int_list(value: Any, default: Sequence[int]) -> List[int]:
    return [int(item) for item in _as_list(value, default)]


def _as_float_list(value: Any, default: Sequence[float]) -> List[float]:
    return [float(item) for item in _as_list(value, default)]


def _clean_for_mongo(value: Any) -> Any:
    return clean_for_mongo(value)


def _portfolio_research_job_pool() -> str:
    return str(os.getenv("PORTFOLIO_RESEARCH_JOB_POOL") or "").strip()


def _claim_job(collections, worker_id: str):
    return _queue(collections).claim(worker_id, worker_pool=_portfolio_research_job_pool())


def _requeue_stale_running_jobs(collections, stale_minutes: int) -> int:
    return _queue(collections).requeue_stale(stale_minutes)


def _set_active_job(collections, job_id: Optional[str]) -> None:
    set_active_job(_queue(collections) if job_id else None, job_id)


def _release_running_job(collections, job_id: str, message: str) -> None:
    _queue(collections).release(job_id, message)


def _handle_shutdown_signal(signum, _frame) -> None:
    handle_shutdown_signal(signum, _frame)


def _install_shutdown_handlers() -> None:
    install_shutdown_handlers()


class _HeartbeatTicker(HeartbeatTicker):
    """Ticker bound to a collection bundle rather than to a queue."""

    def __init__(self, collections, job_id: str, interval_seconds: float) -> None:
        super().__init__(_queue(collections), job_id, interval_seconds)


def _heartbeat_job(collections, job_id: str, stage: str, message: Optional[str] = None) -> None:
    _queue(collections).heartbeat(job_id, stage, message)


@contextmanager
def _worker_lock(lock_path: str):
    with worker_lock(lock_path, LOCK_LABEL):
        yield


def _timed_stage(stage: str, heartbeat: Optional[Callable[[str, Optional[str]], None]], fn: Callable[[], Any]) -> Any:
    started = time.perf_counter()
    if heartbeat:
        heartbeat(stage)
    try:
        return fn()
    finally:
        elapsed = time.perf_counter() - started
        print(f"portfolio research stage {stage} elapsed_s={elapsed:.3f}", flush=True)


def _substage_reporter(
    stage: str, heartbeat: Optional[Callable[[str, Optional[str]], None]]
) -> Callable[[str, float], None]:
    """Report one step inside a stage without moving the job's stage field.

    ``building_dataset`` can run for a quarter of an hour behind a single
    elapsed_s line, which leaves no way to tell slow Mongo from slow pandas, and
    leaves the UI showing a stage that has not changed for minutes.
    """

    def _report(name: str, elapsed: float) -> None:
        print(f"portfolio research stage {stage}.{name} elapsed_s={elapsed:.3f}", flush=True)
        if heartbeat:
            heartbeat(stage, name)

    return _report


@contextmanager
def _timed_substage(
    stage: str, name: str, heartbeat: Optional[Callable[[str, Optional[str]], None]]
):
    if heartbeat:
        heartbeat(stage, name)
    started = time.perf_counter()
    try:
        yield
    finally:
        _substage_reporter(stage, None)(name, time.perf_counter() - started)


def _record_worker_status(collections, worker_id: str, status: str, fields=None, increments=None) -> None:
    _queue(collections).record_worker_status(worker_id, status, fields, increments)


def _fail_job(collections, job_id: str, error: Exception) -> None:
    _queue(collections).fail(job_id, error)


def _complete_job(collections, job_id: str, result_id: str, artifact: Dict[str, Any], data_watermark: Dict[str, Any]) -> None:
    _queue(collections).complete(
        job_id,
        result_id=result_id,
        artifact=artifact,
        data_watermark=data_watermark,
    )


def _slug(value: Any) -> str:
    text = str(value or "").lower()
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        elif ch in {"_", "-", " "}:
            out.append("_")
    return "_".join("".join(out).split("_")) or "research"


def _best_row(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("research sweep produced no result rows")
    return dict(rows[0])


def _candidate_strategy_config(job: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    params = job.get("params") or {}
    universe = str(params.get("universe_index") or job.get("universe_index") or "csi1000")
    if row.get("score_weights"):
        score_type = "weighted_score"
    elif row.get("growth_weight") is not None and row.get("cycle_weight") is not None:
        score_type = "growth_cycle_weighted"
    else:
        score_type = "score_column"
    top_n = int(row.get("top_n") or (params.get("top_n_values") or [10])[0])
    rebalance_days = int(row.get("rebalance_interval_days") or params.get("horizon") or 20)
    variant = str(row.get("variant") or "top_n")
    score_variant = str(row.get("score_variant") or params.get("score_column") or "score")
    preset_id = "research_{universe}_{score}_{variant}_top{top_n}_{days}d_{suffix}".format(
        universe=_slug(universe),
        score=_slug(score_variant),
        variant=_slug(variant),
        top_n=top_n,
        days=rebalance_days,
        suffix=str(job.get("job_id") or uuid.uuid4())[:8],
    )
    candidate = {
        "preset_id": preset_id,
        "strategy_template_id": "growth_cycle_index_enhanced",
        "name": f"{universe.upper()} {score_variant} {variant} Top{top_n} {rebalance_days}d",
        "universe_index": universe,
        "index_benchmark_symbol": (
            params.get("index_benchmark_symbol")
            or row.get("index_benchmark_symbol")
            or default_index_benchmark_symbol(universe)
        ),
        "score_type": score_type,
        "score_column": row.get("score_column") or params.get("score_column") or "composite_growth_cycle_score",
        "growth_weight": row.get("growth_weight"),
        "cycle_weight": row.get("cycle_weight"),
        "score_weights": row.get("score_weights"),
        "top_n": top_n,
        "selection_mode": (
            row.get("selection_mode")
            or params.get("selection_mode")
            or "fixed_top_n"
        ),
        "threshold_lookback_days": int(
            row.get("threshold_lookback_days")
            if row.get("threshold_lookback_days") is not None
            else params.get("threshold_lookback_days", 10)
        ),
        "max_positions": int(
            row.get("max_positions")
            if row.get("max_positions") is not None
            else params.get("max_positions", 20)
        ),
        "rebalance_days": rebalance_days,
        "construction_mode": row.get("construction_mode") or "top_n",
        "max_industry_weight": row.get("max_industry_weight"),
        "lot_size": int(params.get("lot_size") or 100),
        "transaction_cost": float(params.get("transaction_cost") or 0.001),
        "buy_commission_rate": params.get("buy_commission_rate"),
        "sell_commission_rate": params.get("sell_commission_rate"),
        "min_commission": params.get("min_commission"),
        "stamp_tax_rate": params.get("stamp_tax_rate"),
        "transfer_fee_rate": params.get("transfer_fee_rate"),
        "execution_price": params.get("execution_price") or "next_open",
        "cash_buffer": float(params.get("cash_buffer") or 0.03),
        "initial_capital": float(params.get("initial_capital") or 1_000_000.0),
        "trailing_stop_pct": row.get("trailing_stop_pct") if row.get("trailing_stop_pct") is not None else params.get("trailing_stop_pct"),
        "candidate_pool_enabled": bool(params.get("candidate_pool_enabled", True)),
        "industry_l1": str(params.get("industry_l1") or "").strip() or None,
        "status": "draft",
        "version": 1,
    }
    if str(row.get("regime_mode") or "off") == REGIME_MODE_BULL_G60_ELSE_CASH:
        candidate["regime_cash"] = True
        rule = str(row.get("regime_rule") or params.get("regime_rule") or "").strip()
        if rule:
            candidate["regime_rule"] = rule
    return candidate


def _latest_stock_score_date(db_manager, end_date: str, universe_index: Sequence[str]) -> Optional[str]:
    from shared.scores import latest_stock_score_date

    return latest_stock_score_date(getattr(db_manager, "stock_scores", None), end_date, universe_index)


def _universe_pit_quality(db_manager, universe_index: str, start_date: str) -> str:
    """Return ``point_in_time`` when an as-of snapshot exists on/before research start."""
    try:
        from stock_data_access.index_constituents import resolve_snapshot_keys

        start = str(start_date or "").replace("-", "")[:8]
        wt, ud = resolve_snapshot_keys(db_manager.index_constituents, universe_index, as_of_date=start)
        if wt and ud and wt <= start:
            return "point_in_time"
    except Exception:
        # degrading to latest_only is the safe direction, but a lookup error and
        # a genuinely missing snapshot should not look the same to the operator
        log.warning(
            "snapshot lookup failed for %s at %s; reporting latest_only",
            universe_index,
            start_date,
            exc_info=True,
        )
    return "latest_only"


def _data_watermark(
    dataset,
    index_prices: Sequence[Dict[str, Any]],
    *,
    requested_end_date: str,
    stock_scores_max_date: Optional[str],
    universe_index: Optional[str] = None,
    requested_start_date: Optional[str] = None,
    db_manager=None,
) -> Dict[str, Any]:
    backtest_score_max = None
    backtest_score_min = None
    symbol_count = None
    if dataset is not None and not getattr(dataset, "empty", True):
        if "score_date" in dataset.columns:
            backtest_score_max = str(dataset["score_date"].max())
            backtest_score_min = str(dataset["score_date"].min())
        if "symbol" in dataset.columns:
            symbol_count = int(dataset["symbol"].nunique())
    index_trade_dates = [row.get("trade_date") for row in index_prices if row.get("trade_date")]
    watermark = {
        "requested_start_date": requested_start_date,
        "requested_end_date": requested_end_date,
        "stock_scores_max_date": stock_scores_max_date,
        "backtest_score_max_date": backtest_score_max,
        "backtest_score_min_date": backtest_score_min,
        "dataset_rows": int(len(dataset)) if dataset is not None else 0,
        "dataset_symbol_count": symbol_count,
        "index_price_max_date": max(index_trade_dates) if index_trade_dates else None,
    }
    if universe_index and requested_start_date and db_manager is not None:
        watermark["universe_pit_quality"] = _universe_pit_quality(
            db_manager,
            universe_index,
            requested_start_date,
        )
    else:
        watermark["universe_pit_quality"] = "latest_only"
    return watermark


def _trailing_stop_row_key(trailing_stop_pct: Any) -> str:
    if trailing_stop_pct is None:
        return "off"
    try:
        value = float(trailing_stop_pct)
    except (TypeError, ValueError):
        return "off"
    return "off" if value <= 0 else f"{value:g}"


def _trailing_stop_values_from_params(params: Dict[str, Any]) -> List[Optional[float]]:
    from analysis.sweep.params import normalize_trailing_stop_pcts

    trailing_stop_pcts = params.get("trailing_stop_pcts")
    if trailing_stop_pcts is not None:
        return normalize_trailing_stop_pcts(trailing_stop_pcts=trailing_stop_pcts)
    trailing_stop_raw = params.get("trailing_stop_pct")
    trailing_stop_pct = float(trailing_stop_raw) if trailing_stop_raw not in (None, "") else None
    if trailing_stop_pct is not None and trailing_stop_pct <= 0:
        trailing_stop_pct = None
    return normalize_trailing_stop_pcts(trailing_stop_pct=trailing_stop_pct)


def _generate_combo_details(
    *,
    datasets_by_trailing: Dict[str, Any],
    default_dataset,
    rows: List[Dict[str, Any]],
    index_prices: Sequence[Dict[str, Any]],
    horizon: int,
    transaction_cost: float,
    fee_kwargs: Dict[str, Any],
    index_benchmark_symbol: str,
    combo_dir: Path,
    max_rows: int,
    sweep_axes: Optional[Sequence[Dict[str, Any]]] = None,
    cash_on_dates: Optional[Sequence[str]] = None,
    regime_cash_on_dates_by_rule: Optional[Dict[str, Sequence[str]]] = None,
    regime_schedule: Optional[Sequence[Dict[str, Any]]] = None,
    intra_period_cash: bool = False,
    industry_l1_codes: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Generate per-combo trade detail (periods + trades) reusing the in-memory dataset.

    Annotates each processed row with ``combo_key``/``has_detail`` in place, writes one
    JSON per combo under ``combo_dir``, and returns a manifest. The expensive dataset
    is already built by the sweep, so each combo here is just a cheap backtest pass with
    ``include_trades=True``. Caller wraps this in try/except: failures must never break
    the core research result.
    """
    from analysis.sweep.combo_detail import annotate_combo_regime, run_single_combo
    from analysis.sweep.sweep_result_view import combo_key_from_dimensions, infer_sweep_axes

    axes = list(sweep_axes or infer_sweep_axes(rows=rows))

    trading_calendar = [r.get("trade_date") for r in index_prices]
    combo_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    for row in rows[: max(0, int(max_rows))]:
        from analysis.sweep.entry_stock_filter import dataset_cache_key

        ds_key = dataset_cache_key(
            _trailing_stop_row_key(row.get("trailing_stop_pct")),
            row.get("entry_stock_filter"),
        )
        dataset = datasets_by_trailing.get(ds_key, default_dataset)
        score_column = row.get("score_column")
        construction_mode = row.get("construction_mode")
        top_n = row.get("top_n")
        if not score_column or not construction_mode or top_n is None:
            continue
        rebalance = row.get("rebalance_interval_days") or horizon
        row_horizon = int(row.get("horizon") or rebalance or horizon)
        key = combo_key_from_dimensions(row, axes)
        try:
            detail = run_single_combo(
                dataset,
                score_column,
                construction_mode,
                int(top_n),
                row_horizon,
                rebalance_interval_days=int(rebalance),
                max_industry_weight=row.get("max_industry_weight"),
                index_prices=list(index_prices),
                trading_calendar=trading_calendar,
                index_benchmark_symbol=index_benchmark_symbol,
                transaction_cost=transaction_cost,
                volatility_mode=str(row.get("volatility_mode") or "off"),
                target_vol_annual=row.get("target_vol_annual"),
                volatility_column=str(row.get("volatility_column") or "vol_20d_annual"),
                selection_mode=str(row.get("selection_mode") or "fixed_top_n"),
                threshold_lookback_days=int(
                    row.get("threshold_lookback_days")
                    if row.get("threshold_lookback_days") is not None
                    else 10
                ),
                max_positions=int(
                    row.get("max_positions")
                    if row.get("max_positions") is not None
                    else 20
                ),
                cash_on_dates=(
                    list(
                        (regime_cash_on_dates_by_rule or {}).get(
                            str(row.get("regime_rule") or "").strip(),
                            cash_on_dates or [],
                        )
                    )
                    if str(row.get("regime_mode") or "off") == "bull_g60_else_cash"
                    else ()
                ),
                intra_period_cash=(
                    bool(intra_period_cash)
                    and str(row.get("regime_mode") or "off") == "bull_g60_else_cash"
                ),
                industry_regime_scale=(
                    float(row.get("industry_regime_scale"))
                    if row.get("industry_regime_scale") is not None
                    else 1.0
                ),
                industry_l1_codes=tuple(industry_l1_codes or ()),
                **fee_kwargs,
                meta_extra={
                    "combo_key": key,
                    "industry_l1_codes": list(industry_l1_codes or ()),
                    "score_variant": row.get("score_variant"),
                    "variant": row.get("variant"),
                    "volatility_variant": row.get("volatility_variant"),
                    "trailing_stop_pct": row.get("trailing_stop_pct"),
                    "regime_mode": row.get("regime_mode") or "off",
                    "regime_rule": row.get("regime_rule"),
                    "industry_regime_scale": row.get("industry_regime_scale"),
                },
            )
            annotate_combo_regime(detail, regime_schedule)
        except Exception as exc:  # isolate per-combo failures
            row["has_detail"] = False
            row["combo_detail_error"] = str(exc)
            continue
        (combo_dir / f"{key}.json").write_text(
            json.dumps(detail, ensure_ascii=False, default=str), encoding="utf-8"
        )
        row["combo_key"] = key
        row["has_detail"] = True
        manifest.append(
            {
                "combo_key": key,
                "score_variant": row.get("score_variant"),
                "variant": row.get("variant"),
                "construction_mode": construction_mode,
                "top_n": int(top_n),
                "selection_mode": row.get("selection_mode") or "fixed_top_n",
                "threshold_lookback_days": int(
                    row.get("threshold_lookback_days")
                    if row.get("threshold_lookback_days") is not None
                    else 10
                ),
                "max_positions": int(
                    row.get("max_positions")
                    if row.get("max_positions") is not None
                    else 20
                ),
                "rebalance_interval_days": int(rebalance),
                "trailing_stop_pct": row.get("trailing_stop_pct"),
                "n_trades": len(detail.get("trades", [])),
                "n_periods": len(detail.get("periods", [])),
                "file": f"{key}.json",
            }
        )
    return manifest


def _sweep_checkpoint(job_id: str):
    """Checkpoint store for one sweep job.

    On K8s the local checkpoint directory sits on an emptyDir, so it cannot
    survive the pod replacement it exists to protect against — an image rollout
    would otherwise restart a multi-hour sweep from zero. Use S3 wherever the
    artifact backend is already remote.
    """
    from analysis.sweep.sweep_engine import SweepCheckpoint, checkpoint_dir_for_job
    from utils.portfolio_research_s3 import S3CheckpointStore, should_use_s3_backend

    if should_use_s3_backend():
        return SweepCheckpoint(store=S3CheckpointStore(job_id))
    return SweepCheckpoint(checkpoint_dir_for_job(job_id))


def run_research_job(
    job: Dict[str, Any],
    *,
    heartbeat: Optional[Callable[[str, Optional[str]], None]] = None,
) -> Dict[str, Any]:
    """Execute one portfolio research sweep job.

    Capability notes vs CLI: see ``docs/portfolio_research_capabilities.md``.
    """
    from analysis.core.benchmark_prices import load_index_benchmark_prices
    from analysis.reporting.html_report import as_float
    from analysis.sweep import portfolio_sweep_report as sweep
    from analysis.sweep.score_columns import add_weighted_score_columns
    from shared.data_access import open_research_data_access

    params = job.get("params") or {}
    model_id = str(params.get("model_id") or "").strip()
    start_date = str(params.get("start_date") or job.get("start_date") or "20250101").replace("-", "")
    end_date = str(params.get("end_date") or job.get("end_date") or datetime.utcnow().strftime("%Y%m%d")).replace("-", "")
    universe_index = _as_list(params.get("universe_index") or job.get("universe_index"), ["csi1000"])
    score_column = resolve_score_column(params)
    raw_rebalance_intervals = params.get("rebalance_interval_days")
    parsed_rebalance_intervals = (
        None
        if raw_rebalance_intervals in (None, "")
        else _as_int_list(raw_rebalance_intervals, [])
    )
    try:
        horizons = sweep.resolve_tradable_horizons(
            params.get("horizon") or params.get("rebalance_days") or 20,
            parsed_rebalance_intervals if parsed_rebalance_intervals else None,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    horizon = max(horizons)
    rebalance_intervals = list(horizons)
    top_n_values = _as_int_list(params.get("top_n_values"), [10, 20, 50, 100])
    selection_mode = str(params.get("selection_mode") or "fixed_top_n")
    threshold_lookback_days = int(params.get("threshold_lookback_days", 10))
    max_positions = int(params.get("max_positions", 20))
    active_caps = _as_float_list(params.get("active_caps"), [0.2, 0.25, 0.3, 0.4])
    transaction_cost = float(params.get("transaction_cost") or 0.001)
    fee_kwargs = {
        "buy_commission_rate": params.get("buy_commission_rate"),
        "sell_commission_rate": params.get("sell_commission_rate"),
        "min_commission": params.get("min_commission"),
        "stamp_tax_rate": params.get("stamp_tax_rate"),
        "transfer_fee_rate": params.get("transfer_fee_rate"),
    }
    index_benchmark_symbol = str(
        params.get("index_benchmark_symbol") or default_index_benchmark_symbol(universe_index)
    )
    from analysis.core.industry_regime import attach_sw_l1_regime, sw_l1_regime_coverage
    from shared.industry_universe import (
        SW_L1_REGIME_COL,
        attach_sw_l1,
        resolve_industry_l1_codes,
        sw_l1_catalog,
    )

    raw_industry_l1 = params.get("industry_l1") or params.get("industry_l1_codes")
    industry_l1_codes: List[str] = []
    volatility_modes = params.get("volatility_modes")
    target_vols = (
        _as_float_list(params.get("target_vols"), list(sweep.DEFAULT_TARGET_VOLS))
        if params.get("target_vols") is not None
        else None
    )
    volatility_column = str(params.get("volatility_column") or "vol_20d_annual")
    trailing_stop_values = _trailing_stop_values_from_params(params)
    from analysis.sweep.params import normalize_path_gates

    path_gate_values = normalize_path_gates(params.get("path_gates"))
    from analysis.sweep.entry_stock_filter import (
        ENTRY_STOCK_FILTER_OFF,
        apply_entry_stock_filter,
        build_factor_panel_for_dataset,
        dataset_cache_key,
        normalize_entry_stock_filters,
        requires_entry_factor_panel,
    )

    entry_stock_filter_values = normalize_entry_stock_filters(params.get("entry_stock_filters"))
    from shared.regime_label import (
        REGIME_MODE_BULL_G60_ELSE_CASH,
        build_regime_cash_schedule,
        coerce_optional_bool,
        load_regime_rule,
        padded_index_start,
    )
    from analysis.sweep.params import resolve_regime_modes, resolve_regime_rules

    regime_mode_values = resolve_regime_modes(
        regime_modes=params.get("regime_modes"),
        regime_cash=params.get("regime_cash"),
    )
    intra_period_cash = coerce_optional_bool(params.get("intra_period_cash")) is True
    needs_regime_cash = REGIME_MODE_BULL_G60_ELSE_CASH in regime_mode_values
    regime_rule_values = (
        resolve_regime_rules(
            regime_rules=params.get("regime_rules"),
            regime_rule=params.get("regime_rule"),
        )
        if needs_regime_cash
        else []
    )
    regime_rule = regime_rule_values[0] if len(regime_rule_values) == 1 else None
    if not params.get("enable_tushare"):
        os.environ.pop("TUSHARE_TOKEN", None)
        print("ℹ️ 组合研究任务按设计使用本地数据(未启用 Tushare 实时接口)", flush=True)

    def _beat(stage: str, message: Optional[str] = None) -> None:
        if heartbeat:
            heartbeat(stage, message)

    db_manager = _timed_stage(
        "loading_scores",
        _beat,
        lambda: open_research_data_access(),
    )
    if raw_industry_l1:
        catalog = sw_l1_catalog(db_manager)
        values = raw_industry_l1 if isinstance(raw_industry_l1, (list, tuple)) else [raw_industry_l1]
        industry_l1_codes = resolve_industry_l1_codes(values, catalog)
        if len(industry_l1_codes) == 1 and not params.get("index_benchmark_symbol"):
            index_benchmark_symbol = industry_l1_codes[0]
    stock_scores_max_date = _latest_stock_score_date(db_manager, end_date, universe_index)
    try:
        score_sweeps = sweep.build_score_sweeps(
            score_column,
            params.get("growth_cycle_weights"),
            params.get("score_specs"),
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    sweep_score_column = score_sweeps[0].score_column
    index_prices = load_index_benchmark_prices(db_manager, index_benchmark_symbol, start_date)
    try:
        vol_variants = sweep.build_volatility_variants(volatility_modes, target_vols)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    from analysis.sweep.sweep_engine import (
        load_panel_cache,
        log_sweep_perf,
        merge_base_columns,
        merge_volatility_columns,
        panel_cache_key,
        save_panel_cache,
        save_score_panel_cache,
    )

    job_id = str(job.get("job_id") or uuid.uuid4())
    use_panel_cache = params.get("use_panel_cache", True)
    resume_checkpoint = params.get("resume_checkpoint", True)
    checkpoint = _sweep_checkpoint(job_id) if resume_checkpoint else None
    if checkpoint is not None:
        print(f"sweep checkpoint store={checkpoint.describe()} completed={checkpoint.completed_count()}", flush=True)
    cache_key = panel_cache_key(
        universe_index=universe_index,
        start_date=start_date,
        end_date=end_date,
        horizons=horizons,
        stock_scores_max_date=stock_scores_max_date,
    )

    all_rows: List[Dict[str, Any]] = []
    payload: Dict[str, Any] = {}
    datasets_by_trailing: Dict[str, Any] = {}
    last_dataset = None
    score_panel: Optional[pd.DataFrame] = None
    base_dataset: Optional[Any] = None
    backtester = None

    cached_panel = load_panel_cache(cache_key) if use_panel_cache else None
    if cached_panel is not None:
        base_dataset, price_cache, score_panel = cached_panel
        score_config = sweep.BacktestConfig(
            horizons=tuple(horizons),
            start_date=start_date,
            end_date=end_date,
            universe_index_codes=tuple(universe_index),
        )
        backtester = sweep.ScoreBacktester(db_manager, score_config)
        backtester._price_cache = price_cache  # noqa: SLF001 - shared panel reuse
        if score_panel is None or score_panel.empty:
            score_panel = backtester.prepare_score_panel()
            # Complete the entry rather than rebuild it silently: this key was
            # cached before the columnar panel existed, and without writing it
            # back the same rebuild repeats on every future run.
            if use_panel_cache and score_panel is not None and not score_panel.empty:
                save_score_panel_cache(cache_key, score_panel)
        if not base_dataset.empty:
            base_dataset = attach_sw_l1(base_dataset, db_manager, as_of_column="score_date")
            # Re-attached on every cache hit, never read back from the panel: the
            # cached key carries no schema version, so a stale panel would
            # otherwise supply an old label set with no way to notice.
            base_dataset = attach_sw_l1_regime(base_dataset, db_manager)
        log_sweep_perf("panel_cache_hit", 0.0, extra={"cache_key": cache_key})
    else:
        _beat("building_dataset", "prepare_score_panel")

        def _prepare_panel():
            nonlocal backtester, score_panel, base_dataset
            score_config = sweep.BacktestConfig(
                horizons=tuple(horizons),
                start_date=start_date,
                end_date=end_date,
                universe_index_codes=tuple(universe_index),
                trailing_stop_pct=None,
            )
            backtester = sweep.ScoreBacktester(db_manager, score_config)
            score_panel = backtester.prepare_score_panel(
                on_step=_substage_reporter("building_dataset", _beat)
            )
            with _timed_substage("building_dataset", "build_dataset_from_panel", _beat):
                base_dataset = backtester.build_dataset_from_panel(score_panel, trailing_stop_pct=None)
            if not base_dataset.empty:
                with _timed_substage("building_dataset", "add_volatility_columns", _beat):
                    base_dataset = sweep.add_volatility_columns(base_dataset, backtester.price_cache)
                with _timed_substage("building_dataset", "attach_sw_l1", _beat):
                    base_dataset = attach_sw_l1(base_dataset, db_manager, as_of_column="score_date")
                with _timed_substage("building_dataset", "attach_sw_l1_regime", _beat):
                    base_dataset = attach_sw_l1_regime(base_dataset, db_manager)
            if use_panel_cache and base_dataset is not None and not base_dataset.empty:
                with _timed_substage("building_dataset", "save_panel_cache", _beat):
                    save_panel_cache(
                        cache_key,
                        base_dataset,
                        backtester.price_cache,
                        score_panel=score_panel,
                    )
            return base_dataset

        _timed_stage("building_dataset", _beat, _prepare_panel)

    if backtester is None or score_panel is None or base_dataset is None:
        raise RuntimeError("failed to prepare portfolio research dataset panel")

    if model_id:
        from analysis.core.model_scores import merge_model_score_into_dataset

        # New jobs carry an explicit kind from the API/UI. Missing means a
        # legacy job created before P0; keep its old all-rows behaviour rather
        # than silently changing a rerun's result.
        prediction_kind = str(params.get("prediction_kind") or "").strip() or None
        try:
            base_dataset = merge_model_score_into_dataset(
                base_dataset,
                model_id,
                start_date=start_date,
                end_date=end_date,
                prediction_kind=prediction_kind,
                column="model_score",
                ffill_limit=20,
                enabled=True,
            )
            filled = int(base_dataset["model_score"].notna().sum())
            print(
                f"model score merged: model_id={model_id} "
                f"prediction_kind={prediction_kind or 'legacy'} "
                f"rows={len(base_dataset)} filled={filled}",
                flush=True,
            )
            payload["prediction_kind"] = prediction_kind
            payload["prediction_set"] = {
                "model_id": model_id,
                "kind": prediction_kind,
            }
        except Exception as exc:
            raise RuntimeError(f"model score merge failed for {model_id}: {exc}") from exc

    anchor_mode = str(params.get("rebalance_anchor") or "").strip()
    anchor_model_id = str(params.get("rebalance_anchor_model_id") or model_id).strip()
    if anchor_mode == "model_predictions" and anchor_model_id:
        from analysis.core.model_scores import load_model_score_dates

        anchor_dates = load_model_score_dates(
            anchor_model_id,
            start_date=start_date,
            end_date=end_date,
            research_only=bool(params.get("rebalance_anchor_research_only", True)),
            enabled=True,
        )
        if not anchor_dates:
            raise RuntimeError(
                f"rebalance_anchor=model_predictions found no dates for "
                f"model_id={anchor_model_id} in {start_date}..{end_date}"
            )
        before = len(base_dataset)
        base_dataset = base_dataset[
            base_dataset["score_date"].astype(str).isin(anchor_dates)
        ].reset_index(drop=True)
        print(
            f"rebalance anchor: model_id={anchor_model_id} dates={len(anchor_dates)} "
            f"rows {before}->{len(base_dataset)}",
            flush=True,
        )
        if base_dataset.empty:
            raise RuntimeError(
                f"rebalance_anchor=model_predictions left no rows for {anchor_model_id}"
            )

    if path_gate_values != ["off"]:
        from analysis.core.path_features import add_path_feature_columns

        base_dataset = add_path_feature_columns(base_dataset, backtester.price_cache)

    # Watermark fields used in the combo hash (score/index dates, PIT quality) are
    # trailing-stop invariant, so resolve them once instead of per trailing value.
    hash_watermark = _data_watermark(
        base_dataset,
        index_prices,
        requested_end_date=end_date,
        stock_scores_max_date=stock_scores_max_date,
        universe_index=universe_index[0] if universe_index else None,
        requested_start_date=start_date,
        db_manager=db_manager,
    )

    factor_panel = None
    needs_entry_stock_filters = any(
        requires_entry_factor_panel(value) for value in entry_stock_filter_values
    )
    if needs_entry_stock_filters and base_dataset is not None and not base_dataset.empty:
        factor_workers = int(
            params.get("entry_filter_workers")
            or params.get("falsify_workers")
            or 8
        )
        factor_panel = build_factor_panel_for_dataset(
            db_manager,
            base_dataset,
            start_date,
            end_date,
            max_workers=factor_workers,
        )
        print(
            "[entry_stock_filter] factor_panel "
            f"rows={len(factor_panel)} "
            f"symbols={factor_panel['symbol'].nunique() if not factor_panel.empty else 0}",
            flush=True,
        )

    entry_filter_diagnostics: List[Dict[str, Any]] = []
    cash_on_dates: List[str] = []
    regime_cash_on_dates_by_rule: Dict[str, List[str]] = {}
    regime_cash_schedule = None
    if base_dataset is not None and not base_dataset.empty:
        padded_start = padded_index_start(start_date)
        regime_index_rows = load_index_benchmark_prices(db_manager, index_benchmark_symbol, padded_start)
        if not regime_index_rows:
            if needs_regime_cash:
                raise RuntimeError(
                    f"regime_cash requires benchmark prices for {index_benchmark_symbol} from {padded_start}"
                )
        else:
            score_dates = sorted(str(d) for d in base_dataset["score_date"].dropna().unique())
            schedule_dates = score_dates
            if intra_period_cash and needs_regime_cash:
                schedule_dates = [
                    str(row.get("trade_date"))
                    for row in regime_index_rows
                    if row.get("trade_date")
                ]
            rules_to_schedule = regime_rule_values if needs_regime_cash else []
            if not rules_to_schedule:
                rules_to_schedule = [None]
            last_full_schedule = None
            for rule_id in rules_to_schedule:
                cash_on_dates_rule, full_schedule = build_regime_cash_schedule(
                    regime_index_rows,
                    schedule_dates,
                    regime_rule=rule_id,
                )
                last_full_schedule = full_schedule
                if rule_id:
                    regime_cash_on_dates_by_rule[rule_id] = list(cash_on_dates_rule)
                else:
                    cash_on_dates = list(cash_on_dates_rule)
            if needs_regime_cash and regime_rule_values:
                primary_rule = regime_rule_values[0]
                cash_on_dates = list(regime_cash_on_dates_by_rule.get(primary_rule) or [])
                full_schedule = last_full_schedule or []
            else:
                full_schedule = last_full_schedule or []
            if intra_period_cash and needs_regime_cash:
                wanted = set(score_dates)
                regime_cash_schedule = [
                    entry for entry in full_schedule if str(entry.get("score_date")) in wanted
                ]
            else:
                regime_cash_schedule = full_schedule

    for trailing_value in trailing_stop_values:
        label = trailing_value if trailing_value is not None else 0.0
        ds_key = _trailing_stop_row_key(trailing_value)
        _beat("building_dataset", f"trailing_stop_pct={label}")

        def _build_dataset_for_trailing(
            trailing_value=trailing_value,
            ds_key=ds_key,
        ):
            if trailing_value is None or float(trailing_value or 0) <= 0:
                dataset_local = base_dataset.copy()
            else:
                dataset_local = backtester.build_dataset_from_panel(score_panel, trailing_stop_pct=trailing_value)
                # Rebuilt from the score panel: it lacks model_score and the SW
                # columns attached after the base build. Copying them is what
                # makes trailing variants runnable for model-score research.
                dataset_local = merge_base_columns(base_dataset, dataset_local)
            dataset_local = add_weighted_score_columns(dataset_local, score_sweeps)
            if intra_period_cash and needs_regime_cash:
                dataset_local = backtester.add_live_cash_return_columns(
                    dataset_local,
                    cash_on_dates,
                    trailing_stop_pct=trailing_value,
                )
            return dataset_local

        dataset = _timed_stage("building_dataset", _beat, _build_dataset_for_trailing)
        last_dataset = dataset

        for entry_filter in entry_stock_filter_values:
            sweep_dataset = dataset
            if entry_filter != ENTRY_STOCK_FILTER_OFF:
                from analysis.sweep.entry_stock_filter import mask_score_column_stats

                before_stats = mask_score_column_stats(dataset, sweep_score_column)
                sweep_dataset = apply_entry_stock_filter(
                    db_manager,
                    dataset,
                    sweep_score_column,
                    entry_filter,
                    start_date,
                    end_date,
                    factor_panel=factor_panel,
                    max_workers=int(
                        params.get("entry_filter_workers")
                        or params.get("falsify_workers")
                        or 8
                    ),
                )
                after_stats = mask_score_column_stats(sweep_dataset, sweep_score_column)
                entry_filter_diagnostics.append(
                    {
                        "entry_stock_filter": entry_filter,
                        "trailing_stop_pct": label,
                        "before": before_stats,
                        "after": after_stats,
                        "new_nan_cells": int(after_stats.get("nan_count") or 0)
                        - int(before_stats.get("nan_count") or 0),
                    }
                )
                print(
                    "[entry_stock_filter] applied "
                    f"mode={entry_filter} column={sweep_score_column} "
                    f"nan_frac={after_stats.get('nan_frac')} "
                    f"new_nan={entry_filter_diagnostics[-1]['new_nan_cells']}",
                    flush=True,
                )
            cache_key = dataset_cache_key(ds_key, entry_filter)
            datasets_by_trailing[cache_key] = sweep_dataset
            last_dataset = sweep_dataset
            ef_label = entry_filter if entry_filter != ENTRY_STOCK_FILTER_OFF else "off"
            _beat(
                "running_sweep",
                f"trailing_stop_pct={label}, entry_stock_filter={ef_label}",
            )

            def _run_sweep_for_trailing(
                sweep_dataset=sweep_dataset,
                trailing_value=trailing_value,
                watermark=hash_watermark,
                cash_on_dates=cash_on_dates,
                entry_filter=entry_filter,
            ):
                ef_watermark = dict(watermark)
                if entry_filter != ENTRY_STOCK_FILTER_OFF:
                    ef_watermark["entry_stock_filter"] = entry_filter
                part_local = sweep.run_parameter_sweep(
                    dataset=sweep_dataset,
                    score_column=sweep_score_column,
                    top_n_values=top_n_values,
                    horizon=horizon,
                    rebalance_intervals=rebalance_intervals,
                    variants=sweep.build_sweep_variants_for_request(active_caps, industry_l1_codes),
                    transaction_cost=transaction_cost,
                    **fee_kwargs,
                    index_benchmark_symbol=index_benchmark_symbol,
                    index_price_history=index_prices,
                    score_sweeps=score_sweeps,
                    vol_variants=vol_variants,
                    volatility_column=volatility_column,
                    trailing_stop_pct=trailing_value,
                    selection_mode=selection_mode,
                    threshold_lookback_days=threshold_lookback_days,
                    max_positions=max_positions,
                    verbose=bool(params.get("verbose", False)),
                    watermark=ef_watermark,
                    checkpoint=checkpoint,
                    parallel_workers=params.get("sweep_parallel_workers"),
                    independent_horizons=rebalance_intervals,
                    industry_l1_codes=industry_l1_codes,
                    path_gates=path_gate_values,
                    regime_modes=regime_mode_values,
                    regime_rules=regime_rule_values if needs_regime_cash else None,
                    regime_rule=regime_rule,
                    cash_on_dates=cash_on_dates,
                    regime_cash_on_dates_by_rule=regime_cash_on_dates_by_rule or None,
                    intra_period_cash=intra_period_cash and needs_regime_cash,
                    price_cache=getattr(backtester, "price_cache", None),
                )
                for row in part_local.get("rows") or []:
                    row["entry_stock_filter"] = (
                        entry_filter if entry_filter != ENTRY_STOCK_FILTER_OFF else "off"
                    )
                return part_local

            part = _timed_stage("running_sweep", _beat, _run_sweep_for_trailing)
            all_rows.extend(part.get("rows") or [])
            payload = part

    payload["rows"] = sorted(
        all_rows,
        key=lambda row: (
            as_float(row.get("risk_adjusted_score")) is None,
            -(as_float(row.get("risk_adjusted_score")) or 0),
        ),
    )
    payload["trailing_stop_pcts"] = [value if value is not None else 0.0 for value in trailing_stop_values]
    if industry_l1_codes:
        payload["industry_l1_codes"] = industry_l1_codes
    if path_gate_values != ["off"]:
        payload["path_gates"] = path_gate_values
    if entry_stock_filter_values != [ENTRY_STOCK_FILTER_OFF]:
        payload["entry_stock_filters"] = entry_stock_filter_values
    if entry_filter_diagnostics:
        payload["entry_stock_filter_diagnostics"] = entry_filter_diagnostics
    if regime_mode_values != ["off"]:
        payload["regime_modes"] = regime_mode_values
    if regime_cash_schedule is not None:
        payload["regime_cash_schedule"] = regime_cash_schedule
        payload["cash_on_dates"] = cash_on_dates
        payload["intra_period_cash"] = bool(intra_period_cash and needs_regime_cash)
        if regime_rule_values:
            payload["regime_rules"] = list(regime_rule_values)
            if len(regime_rule_values) == 1:
                payload["regime_rule"] = regime_rule_values[0]
    if SW_L1_REGIME_COL in base_dataset.columns:
        # The per-name overlay is only as trustworthy as this mapping coverage;
        # report it next to the results instead of leaving it implicit.
        payload["industry_regime_coverage"] = sw_l1_regime_coverage(base_dataset)
    sweep_axes = sweep.infer_sweep_axes(params, payload["rows"])
    payload["stability"] = sweep.summarize_stability(payload["rows"], sweep_axes=sweep_axes)
    payload = sweep.attach_sweep_view(payload, params)
    sweep_view = payload.get("sweep_view") or {}
    dataset = last_dataset if last_dataset is not None else datasets_by_trailing.get("off")
    try:
        from analysis.factor.ic_metrics import signal_quality
        payload["signal_quality"] = signal_quality(
            dataset,
            [
                str(item.score_column)
                for item in score_sweeps
                if getattr(item, "score_column", None)
            ],
            horizons,
            min_names=int(params.get("ic_min_names") or 20),
        )
    except Exception as exc:  # diagnostics must never fail the research job
        payload["signal_quality"] = {"error": str(exc)}
        print(f"⚠️ signal quality failed: {exc}", flush=True)
    title = f"{', '.join(universe_index).upper()} Portfolio Research"

    def _render_report() -> str:
        return sweep.render_html(
            payload,
            title=title,
            start_date=start_date,
            end_date=end_date,
            index_buy_hold_return=sweep.calculate_index_buy_hold_return(index_prices, start_date, end_date),
        )

    html = _timed_stage("rendering_report", _beat, _render_report)
    result_id = str(uuid.uuid4())
    report_dir = _portfolio_research_report_dir()
    report_dir.mkdir(parents=True, exist_ok=True)
    base = report_dir / f"{job['job_id']}_{result_id[:8]}"
    html_path = base.with_suffix(".html")
    json_path = base.with_suffix(".json")
    html_path.write_text(html, encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # Per-combo trade detail (periods + trades) for the result drill-down UI.
    # Reuses the already-built dataset; isolated so failures never break the job.
    combos_manifest: List[Dict[str, Any]] = []
    if params.get("generate_combo_details", True):
        try:
            combos_manifest = _timed_stage(
                "generating_combo_details",
                _beat,
                lambda: _generate_combo_details(
                    datasets_by_trailing=datasets_by_trailing,
                    default_dataset=dataset,
                    rows=payload.get("rows") or [],
                    index_prices=index_prices,
                    horizon=horizon,
                    transaction_cost=transaction_cost,
                    fee_kwargs=fee_kwargs,
                    index_benchmark_symbol=index_benchmark_symbol,
                    combo_dir=base.parent / f"{base.name}_combos",
                    max_rows=int(params.get("combo_detail_max_rows", params.get("combo_detail_preload_top_k", 10))),
                    sweep_axes=sweep_view.get("sweep_axes"),
                    cash_on_dates=cash_on_dates,
                    regime_cash_on_dates_by_rule=regime_cash_on_dates_by_rule,
                    regime_schedule=regime_cash_schedule,
                    intra_period_cash=intra_period_cash and needs_regime_cash,
                    industry_l1_codes=industry_l1_codes,
                ),
            )
            n_failed = sum(1 for r in (payload.get("rows") or []) if r.get("has_detail") is False)
            print(f"combo detail: generated={len(combos_manifest)} failed={n_failed}", flush=True)
            if not combos_manifest:
                sample_err = next((r.get("combo_detail_error") for r in (payload.get("rows") or []) if r.get("combo_detail_error")), None)
                if sample_err:
                    print(f"⚠️ combo detail produced none; sample error: {sample_err}", flush=True)
        except Exception as exc:  # never fail the research job over detail generation
            print(f"⚠️ combo detail generation failed: {exc}", flush=True)

    # Combo generation annotates payload rows with combo_key / has_detail.
    # Persist those annotations before either local serving or S3 upload.
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    best = _best_row(payload.get("rows") or [])  # global best by risk_adjusted_score
    candidate = _candidate_strategy_config({**job, "params": {**params, "universe_index": universe_index[0]}}, best)
    watermark = _data_watermark(
        dataset,
        index_prices,
        requested_end_date=end_date,
        stock_scores_max_date=stock_scores_max_date,
        universe_index=universe_index[0] if universe_index else None,
        requested_start_date=start_date,
        db_manager=db_manager,
    )
    refs = _artifact_local_refs(report_dir, base)
    artifact = {"html_path": refs["html_path"], "json_path": refs["json_path"]}
    if combos_manifest:
        artifact["combos_dir"] = refs["combos_dir"]
        artifact["combos"] = combos_manifest

    from utils.portfolio_research_s3 import should_use_s3_backend, upload_artifact_to_s3

    if should_use_s3_backend():
        combo_path = base.parent / f"{base.name}_combos" if combos_manifest else None
        s3_fields = upload_artifact_to_s3(
            job_id=str(job["job_id"]),
            result_id=result_id,
            html_path=html_path,
            json_path=json_path,
            combo_dir=combo_path,
        )
        artifact.update(s3_fields)
    result_doc = {
        "result_id": result_id,
        "job_id": job["job_id"],
        "user_id": job.get("user_id"),
        "name": job.get("name"),
        "params": params,
        "rows": payload.get("rows", [])[:200],
        "stability": payload.get("stability", [])[:200],
        "sweep_view": sweep_view,
        "signal_quality": payload.get("signal_quality") or {},
        "row_count_total": len(payload.get("rows") or []),
        "best_row": best,
        "candidate_strategy_config": candidate,
        "artifact": artifact,
        "data_watermark": watermark,
        "created_at": _utcnow(),
    }
    _beat("writing_result")
    return {"result_id": result_id, "artifact": artifact, "data_watermark": watermark, "result_doc": _clean_for_mongo(result_doc)}


def process_one(collections, worker_id: str, heartbeat_seconds: float = 30.0):
    job = _claim_job(collections, worker_id)
    if not job:
        _record_worker_status(collections, worker_id, "idle", {"current_job_id": None, "last_message": "no pending job"})
        return None

    job_id = job["job_id"]
    _record_worker_status(
        collections,
        worker_id,
        "running",
        {"current_job_id": job_id, "last_job_id": job_id, "last_message": None},
    )
    _set_active_job(collections, job_id)
    ticker = _HeartbeatTicker(collections, job_id, heartbeat_seconds).start()
    try:
        heartbeat = lambda stage, message=None: _heartbeat_job(collections, job_id, stage, message)
        output = run_research_job(job, heartbeat=heartbeat)
        collections.research_results.replace_one(
            {"result_id": output["result_id"]},
            output["result_doc"],
            upsert=True,
        )
        _complete_job(collections, job_id, output["result_id"], output["artifact"], output["data_watermark"])
        _record_worker_status(
            collections,
            worker_id,
            "completed",
            {"current_job_id": None, "last_job_id": job_id, "last_result_id": output["result_id"], "last_message": None},
            {"completed_count": 1, "processed_count": 1},
        )
        return {"job_id": job_id, "status": "completed", "result_id": output["result_id"]}
    except Exception as exc:
        _fail_job(collections, job_id, exc)
        _record_worker_status(
            collections,
            worker_id,
            "failed",
            {"current_job_id": None, "last_job_id": job_id, "last_message": str(exc)},
            {"failed_count": 1, "processed_count": 1},
        )
        return {"job_id": job_id, "status": "failed", "error_message": str(exc)}
    finally:
        ticker.stop()
        _set_active_job(None, None)


def main() -> int:

    args = _parse_args()
    try:
        with _worker_lock(args.lock_file):
            return _run_worker(args)
    except WorkerLockHeld as exc:
        print(str(exc), flush=True)
        return 0


def _run_worker(args: argparse.Namespace) -> int:
    from research_db import ensure_research_indexes, get_research_collections

    _install_shutdown_handlers()
    collections = get_research_collections()
    ensure_research_indexes(collections)
    requeued_count = _requeue_stale_running_jobs(collections, args.stale_running_minutes)
    if requeued_count:
        _record_worker_status(
            collections,
            args.worker_id,
            "requeued_stale_jobs",
            {"current_job_id": None, "last_message": f"requeued {requeued_count} stale running research jobs"},
        )

    processed = []
    while len(processed) < args.limit:
        result = process_one(collections, args.worker_id, args.heartbeat_seconds)
        if result:
            processed.append(result)
            continue
        if args.poll_interval <= 0:
            break
        time.sleep(args.poll_interval)

    summary = {"processed": processed, "count": len(processed)}
    if processed or args.verbose_empty:
        print(f"Portfolio research jobs processed: {len(processed)}")
        for item in processed:
            print(f"  {item['status']} job_id={item['job_id']} result_id={item.get('result_id')} error={item.get('error_message')}")
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 1 if any(item["status"] == "failed" for item in processed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
