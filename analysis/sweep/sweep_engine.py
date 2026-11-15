"""Performance, caching, checkpointing, and parallel execution for portfolio sweeps."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from analysis.core.path_features import PATH_FEATURE_COLUMNS
from analysis.core.score_backtester import export_dataset_to_path
from analysis.sweep.params import (
    INDUSTRY_REGIME_SCALE_OFF,
    NON_BULL_ACTION_CASH,
    industry_regime_scale_config,
    non_bull_action_config,
)

_MIB = 1024 * 1024


def _ru_maxrss_to_mb(rss: float, platform: str) -> float:
    """Convert ru_maxrss to MiB using the operating system's documented unit."""
    if platform == "darwin":
        return rss / _MIB
    return rss / 1024


def process_peak_rss_mb() -> Optional[float]:
    """Return this process's peak RSS in MiB, or None when unavailable."""
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        return _ru_maxrss_to_mb(float(usage.ru_maxrss), sys.platform)
    except Exception:
        return None


def peak_rss_mb() -> Optional[float]:
    """Backward-compatible alias for process_peak_rss_mb."""
    return process_peak_rss_mb()


def _read_cgroup_number(path: Path) -> Optional[int]:
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw or raw == "max":
            return None
        return int(raw)
    except (OSError, ValueError):
        return None


def cgroup_memory_metrics(root: Optional[Path] = None) -> Dict[str, Optional[float]]:
    """Read aggregate cgroup-v2 memory metrics for the process tree."""
    cgroup_root = root or Path(
        os.getenv("PORTFOLIO_RESEARCH_CGROUP_ROOT", "/sys/fs/cgroup")
    )
    current_bytes = _read_cgroup_number(cgroup_root / "memory.current")
    peak_bytes = _read_cgroup_number(cgroup_root / "memory.peak")
    limit_bytes = _read_cgroup_number(cgroup_root / "memory.max")

    events: Dict[str, int] = {}
    try:
        for line in (cgroup_root / "memory.events").read_text(encoding="utf-8").splitlines():
            key, value = line.split(maxsplit=1)
            events[key] = int(value)
    except (OSError, ValueError):
        pass

    current_mb = current_bytes / _MIB if current_bytes is not None else None
    limit_mb = limit_bytes / _MIB if limit_bytes is not None else None
    utilization_pct = None
    if current_bytes is not None and limit_bytes:
        utilization_pct = current_bytes / limit_bytes * 100

    return {
        "current_mb": current_mb,
        "peak_mb": peak_bytes / _MIB if peak_bytes is not None else None,
        "limit_mb": limit_mb,
        "utilization_pct": utilization_pct,
        "max_events": float(events["max"]) if "max" in events else None,
        "oom_kill_events": float(events["oom_kill"]) if "oom_kill" in events else None,
    }


@dataclass
class SweepPerfMetrics:
    stage: str
    elapsed_s: float
    combo_count: int = 0
    combos_per_s: Optional[float] = None
    process_peak_rss_mb: Optional[float] = None
    container_current_mb: Optional[float] = None
    container_peak_mb: Optional[float] = None
    container_limit_mb: Optional[float] = None
    container_memory_pct: Optional[float] = None
    container_max_events: Optional[int] = None
    container_oom_kill_events: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None

    def to_log_line(self) -> str:
        parts = [
            f"stage={self.stage}",
            f"elapsed_s={self.elapsed_s:.3f}",
        ]
        if self.combo_count:
            parts.append(f"combos={self.combo_count}")
        if self.combos_per_s is not None:
            parts.append(f"combos_per_s={self.combos_per_s:.2f}")
        if self.process_peak_rss_mb is not None:
            parts.append(f"process_peak_rss_mb={self.process_peak_rss_mb:.1f}")
        if self.container_current_mb is not None:
            parts.append(f"container_current_mb={self.container_current_mb:.1f}")
        if self.container_peak_mb is not None:
            parts.append(f"container_peak_mb={self.container_peak_mb:.1f}")
        if self.container_limit_mb is not None:
            parts.append(f"container_limit_mb={self.container_limit_mb:.1f}")
        if self.container_memory_pct is not None:
            parts.append(f"container_memory_pct={self.container_memory_pct:.1f}")
        if self.container_max_events is not None:
            parts.append(f"container_max_events={self.container_max_events}")
        if self.container_oom_kill_events is not None:
            parts.append(f"container_oom_kill_events={self.container_oom_kill_events}")
        if self.extra:
            for key, value in self.extra.items():
                parts.append(f"{key}={value}")
        return "sweep_perf " + " ".join(parts)


def log_sweep_perf(
    stage: str,
    elapsed_s: float,
    *,
    combo_count: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> SweepPerfMetrics:
    combos_per_s = (combo_count / elapsed_s) if combo_count and elapsed_s > 0 else None
    memory = cgroup_memory_metrics()
    metrics = SweepPerfMetrics(
        stage=stage,
        elapsed_s=elapsed_s,
        combo_count=combo_count,
        combos_per_s=combos_per_s,
        process_peak_rss_mb=process_peak_rss_mb(),
        container_current_mb=memory["current_mb"],
        container_peak_mb=memory["peak_mb"],
        container_limit_mb=memory["limit_mb"],
        container_memory_pct=memory["utilization_pct"],
        container_max_events=int(memory["max_events"]) if memory["max_events"] is not None else None,
        container_oom_kill_events=int(memory["oom_kill_events"]) if memory["oom_kill_events"] is not None else None,
        extra=extra,
    )
    print(metrics.to_log_line(), flush=True)
    return metrics


def sweep_worker_count() -> int:
    raw = os.getenv("PORTFOLIO_RESEARCH_SWEEP_WORKERS")
    if raw is None or raw == "":
        return 1
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def panel_cache_dir() -> Path:
    return Path(os.getenv("PORTFOLIO_RESEARCH_PANEL_CACHE_DIR", "reports/portfolio-research/panel-cache"))


def checkpoint_dir_for_job(job_id: str) -> Path:
    base = Path(os.getenv("PORTFOLIO_RESEARCH_CHECKPOINT_DIR", "reports/portfolio-research/checkpoints"))
    return base / str(job_id)


def panel_cache_key(
    *,
    universe_index: Sequence[str],
    start_date: str,
    end_date: str,
    horizon: Optional[int] = None,
    horizons: Optional[Sequence[int]] = None,
    stock_scores_max_date: Optional[str] = None,
) -> str:
    if horizons:
        horizon_values = sorted({int(value) for value in horizons})
    elif horizon is not None:
        horizon_values = [int(horizon)]
    else:
        raise ValueError("horizon or horizons is required")
    payload = {
        # Bump when the dataset schema changes. v2 adds entry/exit raw prices
        # for lot sizing; a v1 cache would silently fall back to the hfq price
        # and reintroduce the high-adjustment-factor lot-size bug.
        "schema_version": 2,
        "universe_index": sorted(str(item) for item in universe_index),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "horizons": horizon_values,
        "stock_scores_max_date": stock_scores_max_date,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return digest[:24]


def panel_cache_paths(cache_key: str, cache_root: Optional[Path] = None) -> Tuple[Path, Path, Path]:
    root = cache_root or panel_cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    return (
        root / f"panel_{cache_key}.parquet",
        root / f"panel_{cache_key}_price_cache.pkl",
        # Was panel_<key>_score_records.pkl. Renaming it is what invalidates the
        # old caches: a pickle of the list-of-dicts form is exactly the ~9 GiB
        # this refactor removed, so it must never be loaded again.
        root / f"panel_{cache_key}_score_panel.parquet",
    )


def _read_cached_frame(parquet_path: Path) -> Optional[pd.DataFrame]:
    """Read a frame written by ``export_dataset_to_path``, CSV fallback included."""
    csv_path = parquet_path.with_suffix(".csv.gz")
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except ImportError:
            # A cache may have been produced in an environment with pyarrow and
            # read in one without it. Use the CSV fallback when one is present.
            pass
    if not csv_path.exists():
        return None
    columns = pd.read_csv(csv_path, nrows=0).columns
    string_columns = {column: str for column in ("score_date", "symbol") if column in columns}
    return pd.read_csv(csv_path, dtype=string_columns)


def legacy_score_records_path(cache_key: str, cache_root: Optional[Path] = None) -> Path:
    """The pre-refactor pickle of the whole panel as a list of dicts.

    Nothing reads it any more -- its format was the ~9 GiB the columnar panel
    removed -- but it is hundreds of MB per key and would otherwise sit in the
    cache directory forever, since renaming the live file leaves it with no
    reader, no writer and no eviction. `save_panel_cache` unlinks it.
    """
    return (cache_root or panel_cache_dir()) / f"panel_{cache_key}_score_records.pkl"


def load_panel_cache(
    cache_key: str, cache_root: Optional[Path] = None
) -> Optional[Tuple[pd.DataFrame, Dict[str, Any], Optional[pd.DataFrame]]]:
    parquet_path, pickle_path, panel_path = panel_cache_paths(cache_key, cache_root)
    dataset = _read_cached_frame(parquet_path)
    if dataset is None:
        return None
    price_cache: Dict[str, Any] = {}
    if pickle_path.exists():
        with open(pickle_path, "rb") as handle:
            price_cache = pickle.load(handle)
    return dataset, price_cache, _read_cached_frame(panel_path)


def save_panel_cache(
    cache_key: str,
    dataset: pd.DataFrame,
    price_cache: Dict[str, Any],
    *,
    score_panel: Optional[pd.DataFrame] = None,
    cache_root: Optional[Path] = None,
) -> Path:
    parquet_path, pickle_path, panel_path = panel_cache_paths(cache_key, cache_root)
    dataset_path = Path(export_dataset_to_path(dataset, parquet_path))
    _drop_stale_alternative(dataset_path, parquet_path)
    legacy_score_records_path(cache_key, cache_root).unlink(missing_ok=True)
    with open(pickle_path, "wb") as handle:
        pickle.dump(price_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if score_panel is not None:
        _drop_stale_alternative(Path(export_dataset_to_path(score_panel, panel_path)), panel_path)
    return dataset_path


def save_score_panel_cache(
    cache_key: str, score_panel: pd.DataFrame, cache_root: Optional[Path] = None
) -> Path:
    """Write the score panel for a cache entry whose other files already exist.

    The cache-hit path used to be a one-way door: `save_panel_cache` runs only on
    a miss, so a key cached before the columnar panel existed kept rebuilding it
    from Mongo on every run and never wrote the result back -- paying that cost
    forever. Those entries get completed here instead of by re-saving the whole
    entry, because on the hit path the dataset has already had `attach_sw_l1`
    applied and the price cache was just read from disk: handing those back to
    `save_panel_cache` would rewrite hundreds of MB of parquet and pickle on
    every hit and let the cache drift away from what the builder produces.
    """
    *_, panel_path = panel_cache_paths(cache_key, cache_root)
    written = Path(export_dataset_to_path(score_panel, panel_path))
    _drop_stale_alternative(written, panel_path)
    return written


def _drop_stale_alternative(written_path: Path, parquet_path: Path) -> None:
    """Keep exactly one representation of a cached frame.

    Otherwise a stale parquet file can take precedence over a newly written CSV
    fallback (or vice versa).
    """
    stale_path = parquet_path.with_suffix(".csv.gz") if written_path == parquet_path else parquet_path
    if stale_path.exists():
        stale_path.unlink()


def combo_param_hash(
    row_or_params: Dict[str, Any],
    *,
    watermark: Optional[Dict[str, Any]] = None,
    trailing_stop_pct: Optional[float] = None,
) -> str:
    """Stable hash for one outer sweep job (before top_n expansion)."""
    keys = (
        "score_variant",
        "score_column",
        "growth_weight",
        "cycle_weight",
        "variant",
        "construction_mode",
        "max_industry_weight",
        "path_gate",
        "volatility_variant",
        "volatility_mode",
        "target_vol_annual",
        "volatility_column",
        "rebalance_interval_days",
        "transaction_cost",
        "buy_commission_rate",
        "sell_commission_rate",
        "min_commission",
        "stamp_tax_rate",
        "transfer_fee_rate",
        "index_benchmark_symbol",
        "horizon",
    )
    payload: Dict[str, Any] = {key: row_or_params.get(key) for key in keys}
    for key in (
        "selection_mode",
        "threshold_lookback_days",
        "max_positions",
    ):
        if key in row_or_params:
            payload[key] = row_or_params.get(key)
    regime_mode = str(row_or_params.get("regime_mode") or "off").strip().lower()
    if regime_mode not in ("", "off"):
        payload["regime_mode"] = regime_mode
    regime_rule = str(row_or_params.get("regime_rule") or "").strip()
    if regime_rule and regime_mode == "bull_g60_else_cash":
        payload["regime_rule"] = regime_rule
    score_weights = row_or_params.get("score_weights")
    if score_weights:
        payload["score_weights"] = {
            str(key): float(value)
            for key, value in sorted(dict(score_weights).items())
        }
    if trailing_stop_pct is not None:
        payload["trailing_stop_pct"] = trailing_stop_pct
    bull_exposure = row_or_params.get("bull_exposure")
    if bull_exposure is not None:
        try:
            exposure = float(bull_exposure)
        except (TypeError, ValueError):
            exposure = 1.0
        if abs(exposure - 1.0) > 1e-12:
            payload["bull_exposure"] = exposure
    if bool(row_or_params.get("intra_period_refill")):
        payload["intra_period_refill"] = True
    crowd_exit_k = row_or_params.get("crowd_exit_k")
    if crowd_exit_k is not None:
        try:
            crowd_k = int(crowd_exit_k)
        except (TypeError, ValueError):
            crowd_k = 0
        if crowd_k > 0:
            payload["crowd_exit_k"] = crowd_k
    non_bull_action = str(row_or_params.get("non_bull_action") or NON_BULL_ACTION_CASH).strip().lower()
    if non_bull_action and non_bull_action != NON_BULL_ACTION_CASH:
        payload["non_bull_action"] = non_bull_action
    # Default 1.0 is omitted so every pre-existing job_hash keeps its value.
    industry_regime_scale = row_or_params.get("industry_regime_scale")
    if industry_regime_scale is not None:
        try:
            industry_scale = float(industry_regime_scale)
        except (TypeError, ValueError):
            industry_scale = INDUSTRY_REGIME_SCALE_OFF
        if abs(industry_scale - INDUSTRY_REGIME_SCALE_OFF) > 1e-12:
            payload["industry_regime_scale"] = industry_scale
    entry_stock_filter = str(row_or_params.get("entry_stock_filter") or "").strip().lower()
    if entry_stock_filter and entry_stock_filter not in ("", "off"):
        payload["entry_stock_filter"] = entry_stock_filter
    industry_l1_codes = row_or_params.get("industry_l1_codes")
    if industry_l1_codes:
        payload["industry_l1_codes"] = sorted(
            str(code).strip() for code in industry_l1_codes if str(code).strip()
        )
    if watermark:
        payload["watermark"] = {
            key: watermark.get(key)
            for key in (
                "stock_scores_max_date",
                "backtest_score_max_date",
                "index_price_max_date",
                "universe_pit_quality",
                "entry_stock_filter",
            )
        }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return digest[:32]


class LocalCheckpointStore:
    """Keep the completed-job index in a file under ``directory``."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory / "completed_jobs.json"

    def load(self) -> Dict[str, List[Dict[str, Any]]]:
        if not self.index_path.exists():
            return {}
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save(self, payload: Dict[str, List[Dict[str, Any]]]) -> None:
        self.index_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def describe(self) -> str:
        return str(self.index_path)


class SweepCheckpoint:
    """Persist completed outer sweep jobs so a stale worker can resume.

    ``store`` decides where the index lives. A local directory only survives as
    long as its filesystem does, which on K8s is an emptyDir that dies with the
    pod — pass a remote store when the sweep has to outlive its worker.
    """

    def __init__(self, directory: Optional[Path] = None, *, store: Optional[Any] = None):
        if store is None:
            if directory is None:
                raise ValueError("SweepCheckpoint requires either a directory or a store")
            store = LocalCheckpointStore(directory)
        self._store = store
        self.directory = getattr(store, "directory", None)
        self._completed: Dict[str, List[Dict[str, Any]]] = self._store.load()

    def is_done(self, job_hash: str) -> bool:
        return job_hash in self._completed

    def get_rows(self, job_hash: str) -> List[Dict[str, Any]]:
        return list(self._completed.get(job_hash) or [])

    def save_rows(self, job_hash: str, rows: Sequence[Dict[str, Any]]) -> None:
        self._completed[job_hash] = [dict(row) for row in rows]
        self._store.save(self._completed)

    def completed_count(self) -> int:
        return len(self._completed)

    def describe(self) -> str:
        describe = getattr(self._store, "describe", None)
        return describe() if callable(describe) else type(self._store).__name__


@dataclass(frozen=True)
class SweepOuterJob:
    job_hash: str
    score_sweep_name: str
    score_column: str
    growth_weight: Optional[float]
    cycle_weight: Optional[float]
    score_weights: Optional[Dict[str, float]]
    horizon: int
    interval: int
    variant_name: str
    construction_mode: str
    max_industry_weight: Optional[float]
    selection_mode: str
    threshold_lookback_days: int
    max_positions: int
    path_gate: str
    vol_name: str
    vol_mode: str
    target_vol: Optional[float]
    exposure_metrics: Dict[int, Dict[str, Any]]
    trailing_stop_pct: Optional[float]
    regime_mode: str = "off"
    regime_rule: Optional[str] = None
    bull_exposure: float = 1.0
    intra_period_refill: bool = False
    crowd_exit_k: int = 0
    non_bull_action: str = NON_BULL_ACTION_CASH
    industry_regime_scale: float = INDUSTRY_REGIME_SCALE_OFF


_PARALLEL_SHARED: Dict[str, Any] = {}


def _parallel_worker(job: SweepOuterJob) -> Tuple[str, List[Dict[str, Any]]]:
    from analysis.core.portfolio_backtester import PortfolioBacktestConfig, PortfolioBacktester

    dataset = _PARALLEL_SHARED["dataset"]
    index_prices = _PARALLEL_SHARED["index_prices"]
    trading_calendar = _PARALLEL_SHARED["trading_calendar"]
    top_n_values = _PARALLEL_SHARED["top_n_values"]
    horizon = int(job.horizon)
    transaction_cost = _PARALLEL_SHARED["transaction_cost"]
    fee_kwargs = _PARALLEL_SHARED["fee_kwargs"]
    index_benchmark_symbol = _PARALLEL_SHARED["index_benchmark_symbol"]
    volatility_column = _PARALLEL_SHARED["volatility_column"]
    industry_l1_codes = tuple(_PARALLEL_SHARED.get("industry_l1_codes") or ())
    cash_by_rule = _PARALLEL_SHARED.get("regime_cash_on_dates_by_rule") or {}
    default_cash = tuple(_PARALLEL_SHARED.get("cash_on_dates") or ())
    rule_key = str(job.regime_rule or "").strip()
    if rule_key and rule_key in cash_by_rule:
        cash_on_dates = tuple(cash_by_rule[rule_key] or ())
    else:
        cash_on_dates = default_cash
    intra_period_cash = bool(_PARALLEL_SHARED.get("intra_period_cash"))
    if str(job.regime_mode or "off") != "bull_g60_else_cash":
        cash_on_dates = ()
        intra_period_cash = False
        scale, keep_slots = 0.0, False
    else:
        scale, keep_slots = non_bull_action_config(job.non_bull_action)
    max_weight = job.max_industry_weight if job.max_industry_weight is not None else 0.3

    config = PortfolioBacktestConfig(
        score_column=job.score_column,
        horizon=horizon,
        top_n_values=tuple(top_n_values),
        selection_mode=job.selection_mode,
        threshold_lookback_days=job.threshold_lookback_days,
        max_positions=job.max_positions,
        rebalance_interval_days=int(job.interval),
        transaction_cost=transaction_cost,
        buy_commission_rate=fee_kwargs.get("buy_commission_rate"),
        sell_commission_rate=fee_kwargs.get("sell_commission_rate"),
        min_commission=fee_kwargs.get("min_commission"),
        stamp_tax_rate=fee_kwargs.get("stamp_tax_rate"),
        transfer_fee_rate=fee_kwargs.get("transfer_fee_rate"),
        index_benchmark_symbol=index_benchmark_symbol,
        construction_mode=job.construction_mode,
        max_industry_weight=max_weight,
        volatility_mode=job.vol_mode,
        target_vol_annual=job.target_vol,
        volatility_column=volatility_column,
        path_gate=job.path_gate,
        industry_l1_codes=industry_l1_codes,
        cash_on_dates=cash_on_dates,
        intra_period_cash=intra_period_cash,
        bull_exposure=float(job.bull_exposure),
        intra_period_refill=bool(job.intra_period_refill),
        crowd_exit_k=int(job.crowd_exit_k or 0),
        trailing_stop_pct=job.trailing_stop_pct,
        non_bull_top_n_scale=float(scale),
        non_bull_keep_parent_slots=bool(keep_slots),
        industry_regime_scale=industry_regime_scale_config(job.industry_regime_scale),
    )
    result = PortfolioBacktester(
        dataset,
        config,
        index_price_history=index_prices,
        trading_calendar=trading_calendar,
        copy_dataset=False,
        sorted_groups=_PARALLEL_SHARED.get("sorted_groups"),
        selection_cache=_PARALLEL_SHARED.get("selection_cache"),
        price_cache=_PARALLEL_SHARED.get("price_cache"),
    ).run()

    rows: List[Dict[str, Any]] = []
    risk_fn = _PARALLEL_SHARED["risk_adjusted_score_fn"]
    for summary in result.summary.to_dict(orient="records"):
        top_n = int(summary.get("top_n"))
        metrics = job.exposure_metrics.get(top_n, {})
        row_payload = {
                "score_variant": job.score_sweep_name,
                "score_column": job.score_column,
                "growth_weight": job.growth_weight,
                "cycle_weight": job.cycle_weight,
                **({"score_weights": job.score_weights} if job.score_weights else {}),
                "variant": job.variant_name,
                "construction_mode": job.construction_mode,
                "max_industry_weight": job.max_industry_weight,
                "selection_mode": job.selection_mode,
                "threshold_lookback_days": job.threshold_lookback_days,
                "max_positions": job.max_positions,
                "path_gate": job.path_gate,
                "volatility_variant": job.vol_name,
                "volatility_mode": job.vol_mode,
                "target_vol_annual": job.target_vol,
                "volatility_column": volatility_column,
                "rebalance_interval_days": int(job.interval),
                "trailing_stop_pct": job.trailing_stop_pct,
                "regime_mode": job.regime_mode or "off",
                **({"regime_rule": rule_key} if rule_key else {}),
                **summary,
                "bull_exposure": float(job.bull_exposure),
                "intra_period_refill": bool(job.intra_period_refill),
                "crowd_exit_k": int(job.crowd_exit_k or 0),
                "non_bull_action": job.non_bull_action or NON_BULL_ACTION_CASH,
                "industry_regime_scale": float(job.industry_regime_scale),
                "avg_active_share": metrics.get("avg_active_share"),
                "avg_max_abs_active_weight": metrics.get("avg_max_abs_active_weight"),
                "max_abs_active_weight": metrics.get("max_abs_active_weight"),
                "risk_adjusted_score": risk_fn(summary, metrics),
                "sweep_job_hash": job.job_hash,
            }
        if industry_l1_codes:
            row_payload["industry_l1"] = (
                industry_l1_codes[0] if len(industry_l1_codes) == 1 else list(industry_l1_codes)
            )
        rows.append(row_payload)
    return job.job_hash, rows


def run_outer_jobs(
    jobs: Sequence[SweepOuterJob],
    *,
    dataset: pd.DataFrame,
    top_n_values: Sequence[int],
    transaction_cost: float,
    fee_kwargs: Dict[str, Any],
    index_benchmark_symbol: Optional[str],
    index_price_history: Sequence[Dict[str, Any]],
    volatility_column: str,
    risk_adjusted_score_fn: Callable[[Dict[str, Any], Dict[str, Any]], Optional[float]],
    checkpoint: Optional[SweepCheckpoint] = None,
    parallel_workers: Optional[int] = None,
    verbose: bool = False,
    industry_l1_codes: Sequence[str] = (),
    cash_on_dates: Sequence[str] = (),
    regime_cash_on_dates_by_rule: Optional[Dict[str, Sequence[str]]] = None,
    intra_period_cash: bool = False,
    price_cache: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Execute outer sweep jobs, optionally in parallel with checkpoint resume."""
    global _PARALLEL_SHARED
    trading_calendar = [row.get("trade_date") for row in index_price_history]
    pending: List[SweepOuterJob] = []
    rows: List[Dict[str, Any]] = []
    for job in jobs:
        if checkpoint and checkpoint.is_done(job.job_hash):
            rows.extend(checkpoint.get_rows(job.job_hash))
            if verbose:
                print(f"sweep checkpoint hit job_hash={job.job_hash}", flush=True)
            continue
        pending.append(job)

    if not pending:
        return rows

    workers = parallel_workers if parallel_workers is not None else sweep_worker_count()
    started = time.perf_counter()

    def _persist(job_hash: str, job_rows: List[Dict[str, Any]]) -> None:
        rows.extend(job_rows)
        if checkpoint is not None:
            checkpoint.save_rows(job_hash, job_rows)

    shared_payload = {
        "dataset": dataset,
        "index_prices": list(index_price_history),
        "trading_calendar": trading_calendar,
        "top_n_values": list(top_n_values),
        "transaction_cost": transaction_cost,
        "fee_kwargs": dict(fee_kwargs),
        "index_benchmark_symbol": index_benchmark_symbol,
        "volatility_column": volatility_column,
        "risk_adjusted_score_fn": risk_adjusted_score_fn,
        # Shared across every job in this call: daily sorted groups (per score column)
        # and volatility-invariant selection plans, so ranking/selection runs once.
        "sorted_groups": {},
        "selection_cache": {},
        "industry_l1_codes": list(industry_l1_codes),
        "cash_on_dates": list(cash_on_dates),
        "regime_cash_on_dates_by_rule": {
            str(key): list(value or ())
            for key, value in (regime_cash_on_dates_by_rule or {}).items()
        },
        "intra_period_cash": bool(intra_period_cash),
        "price_cache": price_cache if price_cache is not None else {},
    }

    if workers <= 1 or len(pending) <= 1:
        _PARALLEL_SHARED.clear()
        _PARALLEL_SHARED.update(shared_payload)
        for job in pending:
            job_hash, job_rows = _parallel_worker(job)
            _persist(job_hash, job_rows)
    else:
        _PARALLEL_SHARED = shared_payload
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=min(workers, len(pending)), mp_context=ctx) as pool:
            futures = {pool.submit(_parallel_worker, job): job for job in pending}
            for future in as_completed(futures):
                job_hash, job_rows = future.result()
                _persist(job_hash, job_rows)

    log_sweep_perf(
        "running_sweep",
        time.perf_counter() - started,
        combo_count=len(pending),
        extra={"parallel_workers": workers, "resumed": len(jobs) - len(pending)},
    )
    return rows


def merge_volatility_columns(base_dataset: pd.DataFrame, target_dataset: pd.DataFrame) -> pd.DataFrame:
    """Copy volatility/path feature columns from a base panel onto a trailing variant."""
    if base_dataset.empty or target_dataset.empty:
        return target_dataset
    path_cols = set(PATH_FEATURE_COLUMNS)
    feature_cols = [
        column
        for column in base_dataset.columns
        if column.startswith("vol_") or column.startswith("atr_") or column in path_cols
    ]
    if not feature_cols:
        return target_dataset
    keys = ["symbol", "score_date"]
    merged = target_dataset.drop(columns=[col for col in feature_cols if col in target_dataset.columns], errors="ignore")
    return merged.merge(base_dataset[keys + feature_cols], on=keys, how="left")


def build_trailing_datasets(
    backtester,
    score_panel: pd.DataFrame,
    trailing_values: Sequence[Optional[float]],
    *,
    base_dataset: Optional[pd.DataFrame] = None,
    enrich_dataset: Optional[Callable[[Any, Any], Any]] = None,
) -> Dict[str, Any]:
    """Build one dataset per trailing-stop value reusing the loaded price cache."""
    datasets: Dict[str, Any] = {}
    base_panel = base_dataset
    for trailing_value in trailing_values:
        key = "off" if trailing_value is None or float(trailing_value or 0) <= 0 else f"{float(trailing_value):g}"
        if base_panel is not None and (trailing_value is None or float(trailing_value or 0) <= 0):
            dataset = base_panel.copy()
        else:
            dataset = backtester.build_dataset_from_panel(score_panel, trailing_stop_pct=trailing_value)
            if base_panel is not None:
                dataset = merge_volatility_columns(base_panel, dataset)
        if enrich_dataset is not None:
            dataset = enrich_dataset(dataset, backtester)
        datasets[key] = dataset
    return datasets
