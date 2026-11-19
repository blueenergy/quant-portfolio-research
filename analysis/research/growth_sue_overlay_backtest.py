#!/usr/bin/env python3
"""Stage 1：披露季 SUE overlay 叠进真实组合的 A/B 增量回测（离线研究，不写库、不部署）。

把 ``analysis/research/earnings_surprise_extract.py --all-dates`` 产出的 overlay 面板
（symbol / score_date / z_sue_exp_yoy / z_sue_fcst_type / sue_overlay_z）在内存里注入
``dataset["growth_score"]``，再走 ``analysis/sweep/portfolio_sweep_report.py`` 的同一套
sweep 引擎（同 variants、同成本模型、同 benchmark），回答 Stage 0 留下的问题：

    这条预期侧信息在真实组合里、扣掉换手与成本之后，还剩多少？

注入公式（只动有新闻的行；其余行原样）::

    growth_score <- clip(growth_score + λ · clip(z, ±z_clip) · std_date(growth_score), 0, 100)

λ=0 是基线，用于自检「注入路径本身不改变结果」（脚本会与 CLI 口径的
``growth_cycle_g30_c70_score`` 逐值对比）。不写库、不改打分链路、不新增生产开关：
λ 与 growth:cycle 权重只是本脚本的网格维度。

Example:
    python analysis/research/growth_sue_overlay_backtest.py \
        --overlay /app/logs/growth_sue_daily/overlay_panel_all.parquet \
        --start-date 20200228 --end-date 20260930 --horizon 30 \
        --out-dir /app/logs/growth_sue_stage1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1 SUE overlay A/B", formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--overlay", required=True, help="overlay 面板（.parquet/.csv），需含 symbol/score_date/sue_overlay_z")
    p.add_argument("--out-dir", required=True, help="输出目录（rows.csv / summary.json / headline_periods.csv）")
    p.add_argument("--start-date", default="20200228")
    p.add_argument("--end-date", default="20260930")
    p.add_argument("--universe-index", nargs="+", default=["csi1000"])
    p.add_argument("--benchmark", default="000852.SH")
    p.add_argument("--horizon", type=int, default=30, help="持有期=调仓间隔（sweep 要求两者相等）")
    p.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    p.add_argument("--growth-cycle-weights", nargs="+", default=["30:70", "50:50"])
    p.add_argument("--top-n", nargs="+", type=int, default=[10, 20, 50, 100])
    p.add_argument("--variants", nargs="+", help="限制 variant 名（默认 topn + active caps + neutral）")
    p.add_argument("--transaction-cost", type=float, default=0.001)
    p.add_argument("--z-clip", type=float, default=2.5, help="单日 z 截断（防极端值把 growth 顶飞）")
    p.add_argument("--min-news", type=int, default=20, help="单日有新闻票数低于此值不注入")
    p.add_argument("--ir-top-n", nargs="+", type=int, default=[50, 100], help="额外直跑这些 Top-N，取逐期序列算真 IR")
    p.add_argument("--skip-ir", action="store_true", help="跳过直跑 IR 段（只想要主网格时用）")
    p.add_argument("--no-panel-cache", action="store_true", help="禁用 sweep 引擎的面板 disk cache")
    return p.parse_args()


def _load_overlay(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"overlay 文件不存在: {p}")
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    if "sue_overlay_z" not in df.columns:
        # 单日 shard 没有合成列：按 extractor 的定义现算（z_exp_yoy 与 z_fcst_type 的均值）
        parts = [c for c in ("z_sue_exp_yoy", "z_sue_fcst_type") if c in df.columns]
        if not parts:
            raise SystemExit(f"overlay 缺 sue_overlay_z，也没有可回退的 z 列: {sorted(df.columns)[:12]}")
        df["sue_overlay_z"] = df[parts].mean(axis=1, skipna=True)
    need = {"symbol", "score_date", "sue_overlay_z"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"overlay 缺列: {sorted(missing)}")
    df["symbol"] = df["symbol"].astype(str)
    df["score_date"] = df["score_date"].astype(str)
    return df.drop_duplicates(["symbol", "score_date"], keep="last")


def _parse_weight(text: str) -> Tuple[float, float, str]:
    """'30:70' -> (0.3, 0.7, 'g30_c70')，与 sweep CLI 的 growth:cycle 口径一致。"""
    left, right = str(text).split(":", 1)
    growth, cycle = float(left), float(right)
    total = growth + cycle
    if total <= 0:
        raise SystemExit(f"growth:cycle 权重非法: {text}")
    growth, cycle = growth / total, cycle / total
    label = f"g{int(round(growth * 100))}_c{int(round(cycle * 100))}"
    return growth, cycle, label


def _tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _inject(
    base_growth: pd.Series,
    z: pd.Series,
    std_date: pd.Series,
    lam: float,
    mask: pd.Series,
) -> pd.Series:
    if lam == 0:
        return base_growth
    delta = float(lam) * z.fillna(0.0) * std_date
    return (base_growth + delta).clip(0, 100).where(mask, base_growth)


def _print_frame(title: str, df: pd.DataFrame) -> None:
    print(f"\n=== {title} ===", flush=True)
    if df.empty:
        print("(空)", flush=True)
        return
    print(df.to_string(index=False), flush=True)


def main() -> int:
    args = _parse_args()
    from analysis.core.benchmark_prices import load_index_benchmark_prices
    from analysis.core.portfolio_backtester import PortfolioBacktestConfig, PortfolioBacktester
    from analysis.core.score_backtester import BacktestConfig, ScoreBacktester
    from analysis.sweep.portfolio_sweep_report import (
        DEFAULT_ACTIVE_CAPS,
        add_path_feature_columns,
        add_volatility_columns,
        build_sweep_variants,
        run_parameter_sweep,
    )
    from analysis.sweep.score_columns import (
        ScoreSweep,
        add_growth_cycle_weight_columns,
        build_score_sweeps,
    )
    from shared.data_access import open_research_data_access_for_cli
    from shared.industry_universe import attach_sw_l1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    # ---- 1. 数据集：与 portfolio_sweep_report 主流程逐行一致 -------------------
    # 面板走 sweep 引擎的 disk cache（键含 universe/日期/horizons/score 水位），
    # 命中时不重复读库；缓存里存的是裸 dataset，sw_l1 与 vol/path 特征在命中后补。
    from analysis.sweep.sweep_engine import (
        load_panel_cache,
        panel_cache_dir,
        panel_cache_key,
        save_panel_cache,
    )
    from shared.scores import latest_stock_score_date

    db = open_research_data_access_for_cli(enable_tushare=False)
    horizons = (int(args.horizon),)
    stock_scores_max_date = latest_stock_score_date(
        getattr(db, "stock_scores", None), args.end_date, list(args.universe_index)
    )
    cache_key = panel_cache_key(
        universe_index=list(args.universe_index),
        start_date=args.start_date,
        end_date=args.end_date,
        horizons=horizons,
        stock_scores_max_date=stock_scores_max_date,
    )
    cached = None if args.no_panel_cache else load_panel_cache(cache_key)
    if cached is not None:
        dataset, price_cache, _records = cached
        print(f"[stage1] 面板缓存命中 {cache_key} rows={len(dataset)}", flush=True)
        if len(dataset):
            # Recompute rather than trust a cached column: panels are cached after
            # the attach, so a stale cache would silently keep pre-PIT membership.
            dataset = attach_sw_l1(dataset, db, as_of_column="score_date")
    else:
        cfg = BacktestConfig(
            horizons=horizons,
            start_date=args.start_date,
            end_date=args.end_date,
            universe_index_codes=tuple(args.universe_index),
            trailing_stop_pct=None,
        )
        backtester = ScoreBacktester(db, cfg)
        print(f"[stage1] 构建面板 {args.start_date}-{args.end_date} horizon={args.horizon} ...", flush=True)
        score_panel = backtester.prepare_score_panel()
        dataset = backtester.build_dataset_from_panel(score_panel, trailing_stop_pct=None)
        price_cache = backtester.price_cache
        if not dataset.empty and not args.no_panel_cache:
            save_panel_cache(cache_key, dataset, price_cache, score_panel=score_panel, cache_root=panel_cache_dir())
            print(f"[stage1] 面板已缓存 {cache_key}", flush=True)
    if dataset.empty:
        raise SystemExit("面板为空，检查 universe/日期区间")
    dataset = add_volatility_columns(dataset, price_cache)
    dataset = add_path_feature_columns(dataset, price_cache)
    print(
        f"[stage1] 面板就绪 rows={len(dataset)} dates={dataset['score_date'].nunique()} "
        f"symbols={dataset['symbol'].nunique()} ({time.perf_counter() - started:.0f}s)",
        flush=True,
    )

    merged = dataset.copy()
    merged["symbol"] = merged["symbol"].astype(str)
    merged["score_date"] = merged["score_date"].astype(str)

    # ---- 2. overlay 注入（内存，不落盘） --------------------------------------
    overlay = _load_overlay(args.overlay)
    before = len(merged)
    merged = merged.merge(
        overlay[["symbol", "score_date", "sue_overlay_z", "z_sue_exp_yoy", "z_sue_fcst_type"]],
        on=["symbol", "score_date"],
        how="left",
        validate="many_to_one",
    )
    if len(merged) != before:
        raise SystemExit(f"merge 改变了行数 {before} -> {len(merged)}，overlay 键不唯一")
    merged = merged.reset_index(drop=True)

    base_growth = pd.to_numeric(merged["growth_score"], errors="coerce")
    z_raw = pd.to_numeric(merged["sue_overlay_z"], errors="coerce")
    z = z_raw.clip(-float(args.z_clip), float(args.z_clip))
    news = z.notna()
    daily_news = news.groupby(merged["score_date"]).transform("sum")
    mask = news & (daily_news >= int(args.min_news))
    std_date = base_growth.groupby(merged["score_date"]).transform("std")

    covered_dates = merged.loc[news, "score_date"].nunique()
    applied_dates = merged.loc[mask, "score_date"].nunique()
    print(
        f"[stage1] overlay 覆盖 rows={int(news.sum())}/{len(merged)} ({news.mean():.1%}) "
        f"日期={covered_dates}（过 min-news={args.min_news} 门槛的 {applied_dates}）",
        flush=True,
    )
    # 附录：逐季覆盖率（判「季节 overlay」是否真只落在披露季）
    season_cov = (
        merged.assign(month=merged["score_date"].str[4:6], covered=news.astype(int))
        .groupby("month")["covered"]
        .mean()
        .sort_index()
    )
    print("[stage1] 逐月覆盖率: " + " ".join(f"{m}={v:.0%}" for m, v in season_cov.items()), flush=True)

    # ---- 3. 每个 λ × 权重组合一个 score 列 ------------------------------------
    weight_specs = [_parse_weight(text) for text in args.growth_cycle_weights]
    sweeps: List[ScoreSweep] = []
    key_frame = merged[["score_date", "growth_score", "cycle_score"]].copy()
    injection_stats: List[Dict[str, Any]] = []

    for lam in args.lambdas:
        lam_growth = _inject(base_growth, z, std_date, float(lam), mask)
        lam_sweeps = [
            ScoreSweep(
                name=f"sue_l{_tag(lam)}_{label}",
                score_column=f"sue_l{_tag(lam)}_{label}_score",
                growth_weight=growth,
                cycle_weight=cycle,
                weights={"cycle": cycle, "growth": growth},
                legacy_growth_cycle=True,
            )
            for growth, cycle, label in weight_specs
        ]
        work = key_frame.copy()
        work["growth_score"] = lam_growth
        work = add_growth_cycle_weight_columns(work, lam_sweeps)
        for sweep in lam_sweeps:
            merged[sweep.score_column] = work[sweep.score_column].to_numpy()
        sweeps.extend(lam_sweeps)
        applied = mask.to_numpy()
        injection_stats.append(
            {
                "lambda": float(lam),
                "rows_applied": int(applied.sum()),
                "dates_applied": applied_dates,
                "avg_abs_delta": float((lam_growth - base_growth).abs()[applied].mean()) if applied.any() else 0.0,
                "growth_std": float(lam_growth.std()) if lam_growth.notna().any() else None,
            }
        )
        print(
            f"[stage1] λ={lam:g}: 注入行={int(applied.sum())} 平均|Δgrowth|="
            f"{injection_stats[-1]['avg_abs_delta']:.3f} growth_std={injection_stats[-1]['growth_std']:.2f}",
            flush=True,
        )

    # 自检：λ=0 必须与 CLI 口径的 growth_cycle_<label>_score 完全一致
    first_label = weight_specs[0][2]
    check_sweeps = build_score_sweeps("composite_growth_cycle_score", [args.growth_cycle_weights[0]], None)
    check = add_growth_cycle_weight_columns(key_frame.copy(), check_sweeps)
    check_col = check_sweeps[0].score_column
    l0_col = f"sue_l0_{first_label}_score"
    diff = (check[check_col].to_numpy() - merged[l0_col].to_numpy())
    max_diff = float(pd.Series(diff).abs().max())
    print(f"[stage1] 自检 λ=0 vs CLI {check_col}: max|Δ|={max_diff:.3e}", flush=True)
    if max_diff > 1e-9:
        raise SystemExit("λ=0 基线列与 CLI 口径不一致，注入路径污染了基线 —— 停止")

    # ---- 4. 主网格 -----------------------------------------------------------
    variants = build_sweep_variants(DEFAULT_ACTIVE_CAPS)
    if args.variants:
        wanted = {str(name) for name in args.variants}
        variants = [v for v in variants if v.name in wanted]
        if not variants:
            raise SystemExit(f"--variants 没有匹配到任何变体: {sorted(wanted)}")
    index_prices = load_index_benchmark_prices(db, args.benchmark, args.start_date)
    if not index_prices:
        raise SystemExit(f"取不到 benchmark {args.benchmark} 的价格")

    print(
        f"[stage1] sweep: sweeps={len(sweeps)} variants={len(variants)} top_n={args.top_n} "
        f"benchmark={args.benchmark} cost={args.transaction_cost}",
        flush=True,
    )
    payload = run_parameter_sweep(
        dataset=merged,
        score_column=sweeps[0].score_column,
        top_n_values=list(args.top_n),
        horizon=int(args.horizon),
        rebalance_intervals=[int(args.horizon)],
        variants=variants,
        transaction_cost=float(args.transaction_cost),
        index_benchmark_symbol=args.benchmark,
        index_price_history=index_prices,
        score_sweeps=sweeps,
        verbose=True,
    )
    rows = payload["rows"]
    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(out_dir / "rows.csv", index=False)
    print(f"[stage1] sweep 行数={len(rows_df)} ({time.perf_counter() - started:.0f}s)", flush=True)

    # ---- 5. 直跑 headline 组合取逐期序列，算真 IR -----------------------------
    headline = pd.DataFrame()
    if not args.skip_ir:
        trading_calendar = [row.get("trade_date") for row in index_prices]
        periods_out: List[pd.DataFrame] = []
        headline_rows: List[Dict[str, Any]] = []
        for sweep in sweeps:
            for top_n in args.ir_top_n:
                pcfg = PortfolioBacktestConfig(
                    score_column=sweep.score_column,
                    horizon=int(args.horizon),
                    top_n_values=(int(top_n),),
                    rebalance_interval_days=int(args.horizon),
                    transaction_cost=float(args.transaction_cost),
                    index_benchmark_symbol=args.benchmark,
                    construction_mode="top_n",
                    max_industry_weight=0.3,
                )
                result = PortfolioBacktester(
                    merged,
                    pcfg,
                    index_price_history=index_prices,
                    trading_calendar=trading_calendar,
                    copy_dataset=False,
                ).run()
                periods = result.periods.copy()
                if periods.empty:
                    continue
                excess = pd.to_numeric(periods["index_excess_return"], errors="coerce").dropna()
                per_year = 250.0 / float(args.horizon)
                ir = float(excess.mean() / excess.std(ddof=1) * (per_year ** 0.5)) if len(excess) > 2 and excess.std(ddof=1) else None
                summary_row = result.summary.iloc[0].to_dict() if len(result.summary) else {}
                headline_rows.append(
                    {
                        "score_variant": sweep.name,
                        "top_n": int(top_n),
                        "periods": int(len(periods)),
                        "annualized_return": summary_row.get("annualized_return"),
                        "annualized_index_benchmark_return": summary_row.get("annualized_index_benchmark_return"),
                        "avg_excess_per_period": float(excess.mean()) if len(excess) else None,
                        "annualized_ir": ir,
                        "average_turnover": summary_row.get("average_turnover"),
                        "max_drawdown": summary_row.get("max_drawdown"),
                        "yearly_index_excess": summary_row.get("yearly_index_excess"),
                    }
                )
                periods["score_variant"] = sweep.name
                periods["top_n"] = int(top_n)
                periods_out.append(
                    periods[
                        [c for c in ("score_variant", "top_n", "score_date", "portfolio_return_net",
                                     "index_benchmark_return", "index_excess_return", "turnover")
                         if c in periods.columns]
                    ]
                )
        headline = pd.DataFrame(headline_rows)
        if periods_out:
            pd.concat(periods_out, ignore_index=True).to_csv(out_dir / "headline_periods.csv", index=False)
        print(f"[stage1] headline IR 段完成 ({time.perf_counter() - started:.0f}s)", flush=True)

    summary = {
        "config": vars(args),
        "injection": injection_stats,
        "coverage_by_month": {str(k): float(v) for k, v in season_cov.items()},
        "lambda0_selfcheck_max_abs_diff": max_diff,
        "rows": len(rows_df),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

    # ---- 6. 人眼速览 ---------------------------------------------------------
    if not rows_df.empty:
        primary = rows_df[rows_df["variant"] == "growth_cycle_topn"].copy()
        if primary.empty:
            primary = rows_df.copy()
        primary["超额年化"] = pd.to_numeric(primary["annualized_return"], errors="coerce") - pd.to_numeric(
            primary["annualized_index_benchmark_return"], errors="coerce"
        )
        view = primary[
            ["score_variant", "variant", "top_n", "annualized_return", "超额年化",
             "average_excess_return", "average_turnover", "max_drawdown", "excess_hit_rate"]
        ].sort_values(["top_n", "score_variant"])
        _print_frame("主变体：年化 / 超额年化 / 平均超额 / 换手", view.round(4))

        yearly = primary[primary["score_variant"].str.endswith(first_label)].copy()
        if not yearly.empty:
            # 两侧都必须 reset_index：yearly 是过滤后的非连续索引，
            # 直接 concat 会按索引并集对齐，印出大批 NaN 行。
            y = pd.concat(
                [
                    yearly[["score_variant", "top_n"]].reset_index(drop=True),
                    pd.DataFrame(
                        [
                            {k: (round(float(v), 4) if v is not None else None) for k, v in (row or {}).items()}
                            for row in yearly["yearly_index_excess"]
                        ]
                    ).reset_index(drop=True),
                ],
                axis=1,
            ).sort_values(["top_n", "score_variant"])
            _print_frame(f"{first_label} 分年超额（组合净值 − 指数净值）", y)

    if not headline.empty:
        _print_frame(
            "headline 直跑（真 IR = 逐期超额均值/标准差 × √(250/h)）",
            headline.drop(columns=["yearly_index_excess"]).round(4),
        )

    print(f"\n[stage1] 产物 -> {out_dir} ({(time.perf_counter() - started) / 60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
