#!/usr/bin/env python3
"""Judge whether a candidate risk factor is a genuine SELECTION edge or just a
disguised beta lever — the lesson from the volatility A/B (reports/vol_ab*.json).

Read-only. No DB writes, no scoring-pipeline changes. It:

1. Builds the standard score/return panel via ScoreBacktester (reusing its
   point-in-time price cache) for one universe over one span.
2. Constructs candidate LEFT-TAIL factors from prices, all as-of score_date
   (no lookahead): MAX (lottery), downside skew, downside deviation. Plus a
   point-in-time market beta so we can tell selection-alpha from beta-tilt.
3. Runs two diagnostics, split into bull / bear date segments:
   (A) beta-bucket double sort — inside the base-score pool, does the factor
       still separate forward returns / left tail AFTER controlling beta?
   (B) equal-weight top-N portfolio A/B (exposure always 1.0, so any Sharpe
       gap is selection, not exposure timing) comparing the base ranking vs a
       factor-tilted ranking and a factor-filter, reporting realized portfolio
       beta so a hidden low-beta tilt shows up.

Decision rule (mirrors the vol finding): a factor worth adding to the selection
layer should lift Sharpe / index-excess in BOTH regimes at ~unchanged portfolio
beta. If it only helps by lowering beta, it is the same wash as volatility.

Example:
    python analysis/candidate_factor_gate.py \
        --start-date 20230101 --end-date 20260501 --bear-end 20240920 \
        --universe-index csi1000 --benchmark 000852.SH \
        --horizon 20 --top-n 20 --factor max20 \
        --output reports/cand_gate_max20_csi1000.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_DEFAULT_MF_WORKERS = max(1, int(os.getenv("CANDIDATE_FACTOR_GATE_MF_WORKERS", "8")))
_MF_PROGRESS_INTERVAL = max(1, int(os.getenv("CANDIDATE_FACTOR_GATE_MF_PROGRESS", "25")))


# --------------------------------------------------------------------------- #
# Candidate factor construction (point-in-time, as-of score_date)
# --------------------------------------------------------------------------- #

# Price-derived factors (built from the ScoreBacktester price cache).
FACTOR_COLUMNS = ("max20", "dnskew60", "dndev20", "beta120", "ivol60", "rev20", "mdd60")

# Money-flow / volume factors (built from money_flow + raw volume_price).
# hipos120  : close / trailing-120d high, in (0,1]; ~1 = pinned at the highs.
# mainflow20: trailing-20d main-force (super-large+large) net amount / turnover.
# volratio  : vol(5d avg) / vol(20d avg); < 1 = volume drying up (量比萎缩).
# distrib   : 高位派发 interaction = hipos120 * (-mainflow20).
# distribv  : distrib amplified when volume is also drying (缩量滞涨).
MONEYFLOW_RAW = ("hipos120", "mainflow20", "volratio")
MONEYFLOW_DERIVED = ("distrib", "distribv")
MONEYFLOW_COLUMNS = MONEYFLOW_RAW + MONEYFLOW_DERIVED
ALL_FACTOR_COLUMNS = FACTOR_COLUMNS + MONEYFLOW_COLUMNS

# Sign of each factor as a RISK score: +1 => higher value is riskier (penalize
# high), -1 => higher value is safer (penalize low). beta120/hipos120 are
# controls/ingredients, not standalone candidates.
RISK_SIGN = {
    "max20": +1, "dnskew60": -1, "dndev20": +1, "ivol60": +1, "rev20": +1,
    "mdd60": +1, "mainflow20": -1, "volratio": -1, "distrib": +1, "distribv": +1,
    "hipos120": +1,
}


def _symbol_daily_frame(price_rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """close -> daily return frame indexed by trade_date (ascending)."""
    if not price_rows:
        return pd.DataFrame()
    frame = pd.DataFrame(price_rows)
    if "trade_date" not in frame.columns or "close" not in frame.columns:
        return pd.DataFrame()
    frame = frame[["trade_date", "close"]].copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["close"])
    frame = frame[frame["close"] > 0].sort_values("trade_date")
    frame = frame.drop_duplicates("trade_date", keep="last")
    frame["ret"] = frame["close"].pct_change()
    return frame.reset_index(drop=True)


def _index_return_series(index_rows: Sequence[Dict[str, Any]]) -> pd.Series:
    frame = pd.DataFrame(index_rows)
    if frame.empty:
        return pd.Series(dtype=float)
    close_col = next(
        (c for c in ("close", "close_price", "adj_close") if c in frame.columns),
        None,
    )
    date_col = next((c for c in ("trade_date", "date") if c in frame.columns), None)
    if close_col is None or date_col is None:
        return pd.Series(dtype=float)
    frame = frame[[date_col, close_col]].copy()
    frame[date_col] = frame[date_col].astype(str)
    frame[close_col] = pd.to_numeric(frame[close_col], errors="coerce")
    frame = frame.dropna().sort_values(date_col).drop_duplicates(date_col, keep="last")
    ret = frame[close_col].pct_change()
    ret.index = frame[date_col].to_numpy()
    return ret


def build_factor_frame(
    price_cache: Dict[str, List[Dict[str, Any]]],
    index_ret: pd.Series,
    *,
    max_window: int = 20,
    skew_window: int = 60,
    dndev_window: int = 20,
    beta_window: int = 120,
) -> pd.DataFrame:
    """One row per (symbol, trade_date) with rolling factor values known that day."""
    out: List[pd.DataFrame] = []
    idx = index_ret.copy()
    for symbol, rows in price_cache.items():
        f = _symbol_daily_frame(rows)
        if f.empty or len(f) < max(skew_window, beta_window) // 2:
            continue
        ret = f["ret"]
        neg = ret.where(ret < 0)
        close = f["close"]
        f["max20"] = ret.rolling(max_window, min_periods=max_window).max()
        f["dnskew60"] = ret.rolling(skew_window, min_periods=skew_window).skew()
        f["dndev20"] = neg.rolling(dndev_window, min_periods=5).std(ddof=1)
        # short-term reversal / over-extension: trailing max_window run-up
        f["rev20"] = close / close.shift(max_window) - 1.0
        # distance below trailing high (breakdown / 52w-high style), 0..1, higher=deeper
        roll_high = close.rolling(skew_window, min_periods=skew_window // 2).max()
        f["mdd60"] = 1.0 - close / roll_high
        if not idx.empty:
            aligned = idx.reindex(f["trade_date"].to_numpy())
            m = pd.Series(aligned.to_numpy(), index=f.index)
            cov = ret.rolling(beta_window, min_periods=beta_window // 2).cov(m)
            var = m.rolling(beta_window, min_periods=beta_window // 2).var()
            f["beta120"] = cov / var
            # idiosyncratic vol: total vol with the market component removed.
            # Var(resid) = Var(rs) * (1 - corr(rs, rm)^2)  ->  ivol = std_rs*sqrt(1-R^2)
            std_rs = ret.rolling(skew_window, min_periods=skew_window // 2).std(ddof=1)
            corr = ret.rolling(skew_window, min_periods=skew_window // 2).corr(m)
            f["ivol60"] = std_rs * np.sqrt((1.0 - corr**2).clip(lower=0.0)) * math.sqrt(252)
        else:
            f["beta120"] = np.nan
            f["ivol60"] = np.nan
        f["symbol"] = str(symbol)
        out.append(f[["symbol", "trade_date", *FACTOR_COLUMNS]])
    if not out:
        return pd.DataFrame(columns=["symbol", "trade_date", *FACTOR_COLUMNS])
    return pd.concat(out, ignore_index=True)


def attach_factors_asof(
    dataset: pd.DataFrame,
    factor_frame: pd.DataFrame,
    value_cols: Sequence[str] = FACTOR_COLUMNS,
) -> pd.DataFrame:
    """As-of join: each (symbol, score_date) takes the latest factor row with
    trade_date <= score_date. No lookahead."""
    if dataset.empty or factor_frame.empty:
        for col in value_cols:
            dataset[col] = np.nan
        return dataset
    df = dataset.copy()
    df["symbol"] = df["symbol"].astype(str)
    df["score_date"] = df["score_date"].astype(str)
    df["_row_id"] = np.arange(len(df))

    ff = factor_frame.copy()
    ff["symbol"] = ff["symbol"].astype(str)
    ff["trade_date"] = ff["trade_date"].astype(str)

    # merge_asof needs a numeric/datetime key: encode YYYYMMDD dates as int64.
    df["_sd"] = pd.to_numeric(df["score_date"], errors="coerce").astype("Int64")
    ff["_td"] = pd.to_numeric(ff["trade_date"], errors="coerce").astype("Int64")
    ff = ff.dropna(subset=["_td"])

    merged_parts: List[pd.DataFrame] = []
    ff_by_symbol = dict(tuple(ff.groupby("symbol", sort=False)))
    for symbol, grp in df.groupby("symbol", sort=False):
        fsym = ff_by_symbol.get(symbol)
        left = grp.dropna(subset=["_sd"]).sort_values("_sd")
        if fsym is None or fsym.empty or left.empty:
            merged_parts.append(grp)
            continue
        right = fsym.sort_values("_td")
        m = pd.merge_asof(
            left.astype({"_sd": "int64"}),
            right.drop(columns=["symbol", "trade_date"]).astype({"_td": "int64"}),
            left_on="_sd",
            right_on="_td",
            direction="backward",
        )
        merged_parts.append(m)
    merged = pd.concat(merged_parts, ignore_index=True)
    merged = merged.sort_values("_row_id").drop(
        columns=["_row_id", "_sd", "_td"], errors="ignore"
    )
    return merged.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Money-flow / volume factors (distribution-at-highs hypothesis)
# --------------------------------------------------------------------------- #

def _build_symbol_moneyflow_frame(
    db,
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    flow_window: int = 20,
    vol_fast: int = 5,
    vol_slow: int = 20,
    high_window: int = 120,
    hipos_only: bool = False,
    include_daily_basic: bool = False,
) -> Optional[pd.DataFrame]:
    """Build money-flow feature rows for one symbol (point-in-time, no lookahead)."""
    try:
        vp = db.get_volume_price_data(
            symbol, start_date=start_date, end_date=end_date, adjust="none"
        )
    except TypeError:
        vp = db.get_volume_price_data(symbol, start_date=start_date, end_date=end_date)
    if not vp:
        return None

    v = pd.DataFrame(vp)
    if "trade_date" not in v.columns or "close" not in v.columns:
        return None
    v["trade_date"] = v["trade_date"].astype(str)
    for c in ("close", "vol", "amount"):
        if c in v.columns:
            v[c] = pd.to_numeric(v[c], errors="coerce")
    v = v.dropna(subset=["close"]).sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    if v.empty:
        return None
    vol = v["vol"] if "vol" in v.columns else v.get("volume")
    amount = v["amount"] if "amount" in v.columns else (vol * v["close"] if vol is not None else None)

    f = pd.DataFrame({"trade_date": v["trade_date"].to_numpy()})
    close = v["close"].reset_index(drop=True)
    roll_high = close.rolling(high_window, min_periods=high_window // 2).max()
    f["hipos120"] = (close / roll_high).to_numpy()
    if vol is not None:
        vv = vol.reset_index(drop=True)
        f["volratio"] = (
            vv.rolling(vol_fast, min_periods=vol_fast).mean()
            / vv.rolling(vol_slow, min_periods=vol_slow).mean()
        ).to_numpy()
    else:
        f["volratio"] = np.nan

    if not hipos_only:
        mf = db.get_money_flow_data(symbol, start_date=start_date, end_date=end_date)
        if mf:
            m = pd.DataFrame(mf)
            m["trade_date"] = m["trade_date"].astype(str)
            for c in ("buy_elg_amount", "sell_elg_amount", "buy_lg_amount", "sell_lg_amount"):
                m[c] = pd.to_numeric(m.get(c), errors="coerce").fillna(0.0)
            m["main_net"] = (
                m["buy_elg_amount"] + m["buy_lg_amount"]
                - m["sell_elg_amount"] - m["sell_lg_amount"]
            )
            m = m[["trade_date", "main_net"]].dropna().drop_duplicates("trade_date", keep="last")
            merged = f.merge(m, on="trade_date", how="left")
            amt = amount.reset_index(drop=True) if amount is not None else None
            main_net = merged["main_net"]
            if amt is not None:
                amt_wan = amt / 1e4
                num = main_net.rolling(flow_window, min_periods=flow_window // 2).sum()
                den = amt_wan.rolling(flow_window, min_periods=flow_window // 2).sum()
                merged["mainflow20"] = (num / den.where(den > 0)).to_numpy()
            else:
                merged["mainflow20"] = np.nan
            f["mainflow20"] = merged["mainflow20"].to_numpy()
        else:
            f["mainflow20"] = np.nan
    else:
        f["mainflow20"] = np.nan

    if include_daily_basic:
        # financial_daily_basic (~14M rows) is indexed on ts_code, NOT symbol, so
        # get_daily_basic_data (queries by symbol) full-scans the collection. Hit
        # the ts_code index directly to keep this fast, esp. under many threads.
        try:
            dbasic = db.find_daily_basic_by_symbol(
                str(symbol),
                start_date,
                end_date,
                {"trade_date": 1, "total_mv": 1, "_id": 0},
            )
        except Exception:
            dbasic = []
        if dbasic:
            d = pd.DataFrame(dbasic)
            if "trade_date" in d.columns and "total_mv" in d.columns:
                d["trade_date"] = d["trade_date"].astype(str)
                d["total_mv"] = pd.to_numeric(d["total_mv"], errors="coerce")
                d = d[["trade_date", "total_mv"]].dropna().drop_duplicates("trade_date", keep="last")
                f = f.merge(d, on="trade_date", how="left")
            else:
                f["total_mv"] = np.nan
        else:
            f["total_mv"] = np.nan
    else:
        f["total_mv"] = np.nan

    f["symbol"] = str(symbol)
    return f[["symbol", "trade_date", *MONEYFLOW_RAW, "total_mv"]]


def build_moneyflow_frame(
    db,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    *,
    flow_window: int = 20,
    vol_fast: int = 5,
    vol_slow: int = 20,
    high_window: int = 120,
    hipos_only: bool = False,
    include_daily_basic: bool = False,
    max_workers: int = _DEFAULT_MF_WORKERS,
    progress_interval: int = _MF_PROGRESS_INTERVAL,
) -> pd.DataFrame:
    """One row per (symbol, trade_date) with point-in-time money-flow features."""
    symbol_list = [str(s) for s in symbols]
    n = len(symbol_list)
    if n == 0:
        return pd.DataFrame(columns=["symbol", "trade_date", *MONEYFLOW_RAW, "total_mv"])

    mode = "hipos-only" if hipos_only else "full"
    print(
        f"  [moneyflow] mode={mode} symbols={n} span={start_date}-{end_date} "
        f"workers={max_workers}",
        flush=True,
    )

    def _task(symbol: str) -> Tuple[str, Optional[pd.DataFrame]]:
        return symbol, _build_symbol_moneyflow_frame(
            db,
            symbol,
            start_date,
            end_date,
            flow_window=flow_window,
            vol_fast=vol_fast,
            vol_slow=vol_slow,
            high_window=high_window,
            hipos_only=hipos_only,
            include_daily_basic=include_daily_basic,
        )

    out: List[pd.DataFrame] = []
    done = 0
    workers = min(max(1, int(max_workers)), n)
    if workers == 1:
        for i, symbol in enumerate(symbol_list):
            if i % progress_interval == 0:
                print(f"  [moneyflow] {i}/{n} ...", flush=True)
            _, frame = _task(symbol)
            if frame is not None:
                out.append(frame)
            done += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_task, symbol): symbol for symbol in symbol_list}
            for future in as_completed(futures):
                _, frame = future.result()
                if frame is not None:
                    out.append(frame)
                done += 1
                if done == 1 or done % progress_interval == 0 or done == n:
                    print(f"  [moneyflow] {done}/{n} ...", flush=True)

    if not out:
        return pd.DataFrame(columns=["symbol", "trade_date", *MONEYFLOW_RAW, "total_mv"])
    return pd.concat(out, ignore_index=True)


def _factor_needs_mainflow(factor: str) -> bool:
    return factor in ("distrib", "distribv", "mainflow20", "volratio")


def _moneyflow_cache_complete(dataset: pd.DataFrame, *, need_mainflow: bool, need_daily_basic: bool) -> bool:
    if "hipos120" not in dataset.columns:
        return False
    if need_mainflow:
        if "mainflow20" not in dataset.columns:
            return False
        if int(dataset["mainflow20"].notna().sum()) == 0:
            return False
    if need_daily_basic:
        if "total_mv" not in dataset.columns:
            return False
        if int(dataset["total_mv"].notna().sum()) == 0:
            return False
    return True


def add_moneyflow_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Build the interaction (高位派发) columns from the raw money-flow features."""
    df = df.copy()
    hipos = pd.to_numeric(df.get("hipos120"), errors="coerce")
    mainflow = pd.to_numeric(df.get("mainflow20"), errors="coerce")
    volratio = pd.to_numeric(df.get("volratio"), errors="coerce")
    if not isinstance(hipos, pd.Series):
        hipos = pd.Series(hipos, index=df.index)
    if not isinstance(mainflow, pd.Series):
        mainflow = pd.Series(mainflow, index=df.index)
    if not isinstance(volratio, pd.Series):
        volratio = pd.Series(volratio, index=df.index)
    # distrib: near-high AND main-force leaving. hipos in (0,1] gates the outflow.
    df["distrib"] = hipos * (-mainflow)
    # distribv: amplify when volume is also drying up (volratio < 1).
    shrink = (1.0 - volratio).clip(lower=0.0).fillna(0.0)
    df["distribv"] = hipos * (-mainflow) * (1.0 + shrink)
    return df


# --------------------------------------------------------------------------- #
# Candidate ranking columns (selection layer)
# --------------------------------------------------------------------------- #

def _zscore_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    def _z(s: pd.Series) -> pd.Series:
        v = pd.to_numeric(s, errors="coerce")
        mu = v.mean()
        sd = v.std(ddof=0)
        if not sd or math.isnan(sd) or sd == 0:
            return pd.Series(0.0, index=s.index)
        return (v - mu) / sd
    return df.groupby("score_date")[col].transform(_z)


def add_candidate_columns(
    df: pd.DataFrame,
    base_col: str,
    factor_col: str,
    *,
    lam: float,
    pool_mult: float,
    drop_quantile: float,
    top_n: int,
    cap_gate_quantile: Optional[float] = None,
) -> Dict[str, str]:
    """Add two candidate ranking columns and return {label: column_name}.

    blend  : z(base) - lam * z(factor)   (penalize high-factor names in ranking)
    filter : keep base score, but blank out the highest-factor names inside the
             per-date base pool (two-stage: select big pool, drop worst tail).
    Higher factor value = riskier for max20 / dndev20; for dnskew60 lower
    (more negative) is riskier, so we flip its sign into a risk score first.
    """
    risk = pd.to_numeric(df[factor_col], errors="coerce") * RISK_SIGN.get(factor_col, +1)
    # Market-cap gate: exempt the largest names (>= per-date cap quantile) from
    # the risk factor entirely, so the signal only acts on small/mid caps where
    # the "high + distribution" pattern is informative.
    if cap_gate_quantile is not None and "total_mv" in df.columns:
        cap = pd.to_numeric(df["total_mv"], errors="coerce")
        cap_thresh = df.assign(_cap=cap).groupby("score_date")["_cap"].transform(
            lambda s: s.quantile(cap_gate_quantile)
        )
        exempt = cap >= cap_thresh
        risk = risk.mask(exempt.fillna(False))
    df["_risk"] = risk

    z_base = _zscore_by_date(df, base_col)
    z_risk = _zscore_by_date(df, "_risk")
    blend_col = f"cand_blend_{factor_col}"
    df[blend_col] = z_base - lam * z_risk

    pool_n = max(top_n, int(round(pool_mult * top_n)))

    def _filter(group: pd.DataFrame) -> pd.Series:
        base = pd.to_numeric(group[base_col], errors="coerce")
        out = pd.Series(np.nan, index=group.index)
        pool = base.dropna().sort_values(ascending=False).head(pool_n).index
        if len(pool) == 0:
            return out
        pr = group.loc[pool, "_risk"]
        valid_risk = pr.dropna()
        if len(valid_risk) >= 5 and drop_quantile > 0:
            thresh = valid_risk.quantile(1.0 - drop_quantile)
            keep = pool[(pr <= thresh) | (pr.isna())]
        else:
            keep = pool
        out.loc[keep] = base.loc[keep]
        return out

    filter_col = f"cand_filter_{factor_col}"
    try:
        filtered = df.groupby("score_date", group_keys=False).apply(_filter, include_groups=False)
    except TypeError:  # pandas < 2.2 has no include_groups kwarg
        filtered = df.groupby("score_date", group_keys=False).apply(_filter)
    df[filter_col] = filtered.reindex(df.index)
    df.drop(columns=["_risk"], inplace=True, errors="ignore")
    return {"blend": blend_col, "filter": filter_col}


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #

def double_sort(
    df: pd.DataFrame,
    base_col: str,
    factor_col: str,
    ret_col: str,
    *,
    top_pool_frac: float = 0.5,
    beta_buckets: int = 3,
    factor_buckets: int = 3,
) -> pd.DataFrame:
    """Inside the top-half base-score pool per date, bucket by beta then by the
    candidate factor; report forward-return mean, 5% left tail, negative prob."""
    risk_flip = RISK_SIGN.get(factor_col, +1) < 0
    rows: List[Dict[str, Any]] = []
    for score_date, grp in df.groupby("score_date", sort=True):
        g = grp[["symbol", base_col, factor_col, "beta120", ret_col]].copy()
        g = g.dropna(subset=[base_col, factor_col, ret_col])
        if len(g) < 30:
            continue
        # restrict to the high base-score names we would actually consider
        cut = g[base_col].quantile(1.0 - top_pool_frac)
        g = g[g[base_col] >= cut]
        if len(g) < 15:
            continue
        g = g.copy()
        g["_fac"] = -g[factor_col] if risk_flip else g[factor_col]
        # beta bucket (may be missing beta -> its own bucket)
        if g["beta120"].notna().sum() >= beta_buckets * 3:
            try:
                g["_bb"] = pd.qcut(g["beta120"].rank(method="first"), beta_buckets, labels=False)
            except ValueError:
                g["_bb"] = -1
        else:
            g["_bb"] = -1
        for bb, gb in g.groupby("_bb"):
            if len(gb) < factor_buckets * 2:
                continue
            try:
                gb = gb.copy()
                gb["_fb"] = pd.qcut(gb["_fac"].rank(method="first"), factor_buckets, labels=False)
            except ValueError:
                continue
            for fb, gf in gb.groupby("_fb"):
                rows.append(
                    {
                        "score_date": score_date,
                        "beta_bucket": int(bb),
                        "factor_bucket": int(fb),  # 0 = low risk, high bucket = high risk
                        "ret": gf[ret_col].mean(),
                        "n": len(gf),
                    }
                )
    daily = pd.DataFrame(rows)
    if daily.empty:
        return pd.DataFrame()
    agg = (
        daily.groupby(["beta_bucket", "factor_bucket"])["ret"]
        .agg(mean_ret="mean", tail5=lambda s: float(np.nanpercentile(s, 5)),
             neg_prob=lambda s: float((s < 0).mean()), obs="count")
        .reset_index()
    )
    return agg


def portfolio_ab(
    dataset: pd.DataFrame,
    score_cols: Dict[str, str],
    beta_lookup: pd.DataFrame,
    *,
    horizon: int,
    top_n: int,
    benchmark: str,
    index_rows: Sequence[Dict[str, Any]],
    trading_calendar: Sequence[str],
) -> List[Dict[str, Any]]:
    from analysis.core.portfolio_backtester import PortfolioBacktestConfig, PortfolioBacktester

    beta_map = beta_lookup.set_index(["symbol", "score_date"])["beta120"].to_dict()

    results: List[Dict[str, Any]] = []
    for label, col in score_cols.items():
        if col not in dataset.columns:
            continue
        cfg = PortfolioBacktestConfig(
            score_column=col,
            horizon=horizon,
            top_n_values=(top_n,),
            rebalance_interval_days=horizon,
            transaction_cost=0.001,
            index_benchmark_symbol=benchmark,
            construction_mode="top_n",
            volatility_mode="off",
            include_trades=True,
        )
        bt = PortfolioBacktester(
            dataset, cfg, index_price_history=list(index_rows), trading_calendar=list(trading_calendar)
        )
        res = bt.run()
        if res.summary.empty:
            continue
        s = res.summary.to_dict(orient="records")[0]

        # realized portfolio beta: mean beta of equal-weighted picks, averaged over periods
        port_beta = np.nan
        if not res.trades.empty:
            tr = res.trades.copy()
            tr["symbol"] = tr["symbol"].astype(str)
            tr["score_date"] = tr["score_date"].astype(str)
            tr["_beta"] = [
                beta_map.get((sy, sd)) for sy, sd in zip(tr["symbol"], tr["score_date"], strict=False)
            ]
            per_period = tr.groupby("score_date")["_beta"].mean()
            port_beta = float(per_period.mean()) if len(per_period) else np.nan

        # left tail of period returns
        tail5 = neg_prob = np.nan
        if not res.periods.empty and "portfolio_return_net" in res.periods:
            pr = pd.to_numeric(res.periods["portfolio_return_net"], errors="coerce").dropna()
            if len(pr):
                tail5 = float(np.percentile(pr, 5))
                neg_prob = float((pr < 0).mean())

        results.append(
            {
                "variant": label,
                "score_column": col,
                "periods": int(s.get("periods") or 0),
                "cum_return": s.get("cumulative_return"),
                "index_excess": s.get("index_excess_cumulative_return"),
                "sharpe": s.get("sharpe"),
                "max_drawdown": s.get("max_drawdown"),
                "hit_rate": s.get("hit_rate"),
                "portfolio_beta": port_beta,
                "period_ret_tail5": tail5,
                "period_neg_prob": neg_prob,
            }
        )
    return results


# --------------------------------------------------------------------------- #
# Regime
# --------------------------------------------------------------------------- #

def regime_distribution(index_rows: Sequence[Dict[str, Any]], score_dates: Sequence[str]) -> Dict[str, int]:
    ret = _index_return_series(index_rows)  # noqa: F841 (only need closes)
    frame = pd.DataFrame(index_rows)
    close_col = next((c for c in ("close", "close_price", "adj_close") if c in frame.columns), None)
    date_col = next((c for c in ("trade_date", "date") if c in frame.columns), None)
    if close_col is None or date_col is None:
        return {}
    frame = frame[[date_col, close_col]].copy()
    frame[date_col] = frame[date_col].astype(str)
    frame[close_col] = pd.to_numeric(frame[close_col], errors="coerce")
    frame = frame.dropna().sort_values(date_col)
    frame["ma20"] = frame[close_col].rolling(20).mean()
    frame["ma60"] = frame[close_col].rolling(60).mean()
    frame["ma120"] = frame[close_col].rolling(120).mean()
    frame = frame.set_index(date_col)
    dist = {"bull": 0, "bear": 0, "neutral": 0}
    for td in score_dates:
        sub = frame[frame.index <= td]
        if sub.empty or pd.isna(sub.iloc[-1]["ma120"]):
            dist["neutral"] += 1
            continue
        r = sub.iloc[-1]
        if r["ma20"] > r["ma60"] > r["ma120"]:
            dist["bull"] += 1
        elif r["ma20"] < r["ma60"] < r["ma120"]:
            dist["bear"] += 1
        else:
            dist["neutral"] += 1
    return dist


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Candidate risk-factor selection-vs-beta gate.")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--bear-end", required=True, help="date splitting bear (<=) from bull (>) segment, YYYYMMDD")
    p.add_argument("--universe-index", nargs="+", default=["csi1000"])
    p.add_argument("--benchmark", default="000852.SH")
    p.add_argument("--base-column", default="composite_growth_cycle_score")
    p.add_argument("--factor", choices=list(ALL_FACTOR_COLUMNS), default="max20")
    p.add_argument(
        "--with-moneyflow",
        action="store_true",
        help="Also build money-flow/volume factors (hipos120/mainflow20/volratio/"
        "distrib/distribv). Needed for --factor in that group.",
    )
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--lam", type=float, default=0.5, help="blend penalty on the risk factor")
    p.add_argument("--pool-mult", type=float, default=2.0, help="filter pool = pool_mult * top_n")
    p.add_argument("--drop-quantile", type=float, default=0.33, help="fraction of highest-risk pool names dropped")
    p.add_argument(
        "--cap-gate-quantile",
        type=float,
        default=None,
        help="Exempt names with total_mv >= this per-date quantile from the risk "
        "factor (e.g. 0.67 exempts the largest ~1/3). Implies money-flow build.",
    )
    p.add_argument("--output", help="Optional JSON output path")
    p.add_argument(
        "--panel-cache",
        help="Parquet path for the factor-attached panel; reused if present so the "
        "~9min DB build is skipped on re-runs / other factors.",
    )
    p.add_argument("--enable-tushare", action="store_true")
    p.add_argument(
        "--mf-workers",
        type=int,
        default=_DEFAULT_MF_WORKERS,
        help="Parallel workers for money-flow feature build (default from env or 8)",
    )
    return p.parse_args()


def _fmt(v: Any, spec: str = "+8.3f") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return f"{'na':>8s}"
    return format(v, spec)


def main() -> int:
    from analysis.core.benchmark_prices import load_index_benchmark_prices
    from analysis.core.score_backtester import BacktestConfig, ScoreBacktester
    from shared.data_access import open_research_data_access_for_cli

    args = _parse_args()

    db = open_research_data_access_for_cli(enable_tushare=args.enable_tushare)
    ret_col = f"forward_return_{args.horizon}d"

    # Benchmark prices are cheap (single symbol) and always needed for regime,
    # trading calendar and portfolio A/B, so load them regardless of panel cache.
    print(f"[factor] loading benchmark {args.benchmark} ...", flush=True)
    index_rows = load_index_benchmark_prices(db, args.benchmark, args.start_date)
    index_ret = _index_return_series(index_rows)

    cache_path = Path(args.panel_cache) if args.panel_cache else None
    if cache_path and cache_path.exists():
        print(f"[build] loading cached panel {cache_path} ...", flush=True)
        dataset = pd.read_parquet(cache_path)
        dataset["symbol"] = dataset["symbol"].astype(str)
        dataset["score_date"] = dataset["score_date"].astype(str)
    else:
        print(
            f"[build] dataset {args.start_date}-{args.end_date} "
            f"universe={args.universe_index} horizon={args.horizon} ...",
            flush=True,
        )
        cfg = BacktestConfig(
            horizons=(args.horizon,),
            start_date=args.start_date,
            end_date=args.end_date,
            universe_index_codes=tuple(args.universe_index),
        )
        bt = ScoreBacktester(db, cfg)
        dataset = bt.build_dataset()
        if dataset.empty:
            print("[error] empty dataset", file=sys.stderr)
            return 1
        print(
            f"[build] rows={len(dataset)} dates={dataset['score_date'].nunique()} "
            f"symbols={dataset['symbol'].nunique()}",
            flush=True,
        )
        print("[factor] building point-in-time factors ...", flush=True)
        factor_frame = build_factor_frame(bt.price_cache, index_ret)
        dataset = attach_factors_asof(dataset, factor_frame)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            h = args.horizon
            keep = [
                c
                for c in (
                    "symbol", "score_date", "name", "industry",
                    args.base_column, "growth_score", "cycle_score", ret_col,
                    "entry_price", "entry_date", "entry_price_type",
                    f"exit_price_{h}d", f"exit_date_{h}d",
                    f"exit_price_type_{h}d", f"exit_delayed_days_{h}d",
                    *ALL_FACTOR_COLUMNS, "total_mv",
                )
                if c in dataset.columns
            ]
            dataset[keep].to_parquet(cache_path, index=False)
            print(f"[build] cached panel -> {cache_path}", flush=True)

    # Optionally augment with money-flow / volume factors (distribution-at-highs).
    need_mf = args.with_moneyflow or args.factor in MONEYFLOW_COLUMNS or args.cap_gate_quantile is not None
    need_mainflow = _factor_needs_mainflow(args.factor) or args.with_moneyflow
    need_daily_basic = args.cap_gate_quantile is not None
    need_rebuild_mf = need_mf and not _moneyflow_cache_complete(
        dataset, need_mainflow=need_mainflow, need_daily_basic=need_daily_basic
    )
    if need_rebuild_mf:
        from datetime import datetime, timedelta

        mf_start = (datetime.strptime(args.start_date, "%Y%m%d") - timedelta(days=260)).strftime("%Y%m%d")
        panel_end = str(dataset["score_date"].max()) if "score_date" in dataset.columns else args.end_date
        mf_end = max(str(args.end_date), panel_end)
        symbols = sorted(dataset["symbol"].astype(str).unique())
        hipos_only = args.factor == "hipos120" and not need_mainflow and not need_daily_basic
        print(
            f"[moneyflow] building for {len(symbols)} symbols from {mf_start} to {mf_end} ...",
            flush=True,
        )
        mf_frame = build_moneyflow_frame(
            db,
            symbols,
            mf_start,
            mf_end,
            hipos_only=hipos_only,
            include_daily_basic=need_daily_basic,
            max_workers=args.mf_workers,
        )
        dataset = attach_factors_asof(dataset, mf_frame, value_cols=(*MONEYFLOW_RAW, "total_mv"))
        dataset = add_moneyflow_derived(dataset)
        if cache_path:  # persist so subsequent money-flow factors are fast
            dataset.to_parquet(cache_path, index=False)
            print(f"[moneyflow] re-cached panel with money-flow columns -> {cache_path}", flush=True)
    elif need_mf and "distrib" not in dataset.columns:
        dataset = add_moneyflow_derived(dataset)

    present = [c for c in ALL_FACTOR_COLUMNS if c in dataset.columns]
    cov = {c: float(dataset[c].notna().mean()) for c in present}
    print("[factor] coverage: " + ", ".join(f"{k}={v:.1%}" for k, v in cov.items()), flush=True)

    cand = add_candidate_columns(
        dataset,
        args.base_column,
        args.factor,
        lam=args.lam,
        pool_mult=args.pool_mult,
        drop_quantile=args.drop_quantile,
        top_n=args.top_n,
        cap_gate_quantile=args.cap_gate_quantile,
    )
    score_cols = {"base": args.base_column, **cand}

    trading_calendar = sorted({r for r in (index_ret.index.tolist())})

    segments = {
        f"BEAR {args.start_date}..{args.bear_end}": dataset[dataset["score_date"] <= args.bear_end].copy(),
        f"BULL {args.bear_end}..{args.end_date}": dataset[dataset["score_date"] > args.bear_end].copy(),
    }

    report: Dict[str, Any] = {
        "params": vars(args),
        "factor": args.factor,
        "coverage": cov,
        "segments": {},
    }

    for tag, sub in segments.items():
        if sub.empty:
            continue
        dist = regime_distribution(index_rows, sorted(sub["score_date"].unique()))
        print(f"\n================ {tag} (dates={sub['score_date'].nunique()}, MA-regime {dist}) ================", flush=True)

        # (B) portfolio A/B
        ab = portfolio_ab(
            sub, score_cols, dataset[["symbol", "score_date", "beta120"]],
            horizon=args.horizon, top_n=args.top_n, benchmark=args.benchmark,
            index_rows=index_rows, trading_calendar=trading_calendar,
        )
        print("  [Portfolio A/B, equal-weight top-N, exposure=1.0]", flush=True)
        print(
            f"  {'variant':16s} {'cum':>8s} {'idx_exc':>8s} "
            f"{'sharpe':>7s} {'maxDD':>8s} {'hit':>6s} "
            f"{'p_beta':>7s} {'tail5':>8s} {'neg_p':>6s}",
            flush=True,
        )
        for r in ab:
            print(
                f"  {r['variant']:16s} {_fmt(r['cum_return'])} {_fmt(r['index_excess'])} "
                f"{_fmt(r['sharpe'],'7.2f')} {_fmt(r['max_drawdown'])} {_fmt(r['hit_rate'],'6.2f')} "
                f"{_fmt(r['portfolio_beta'],'7.2f')} {_fmt(r['period_ret_tail5'])} {_fmt(r['period_neg_prob'],'6.2f')}",
                flush=True,
            )

        # (A) beta-bucket double sort
        ds_tbl = double_sort(sub, args.base_column, args.factor, ret_col)
        ds_records = ds_tbl.to_dict(orient="records") if not ds_tbl.empty else []
        if not ds_tbl.empty:
            print("\n  [Beta-bucket double sort inside top-half base pool] "
                  "(factor_bucket 0=low risk .. high=high risk)", flush=True)
            print(f"  {'beta_b':>6s} {'fac_b':>5s} {'mean_ret':>9s} {'tail5':>8s} {'neg_prob':>8s} {'obs':>5s}", flush=True)
            for _, r in ds_tbl.iterrows():
                print(
                    f"  {int(r['beta_bucket']):6d} {int(r['factor_bucket']):5d} "
                    f"{_fmt(r['mean_ret'],'9.4f')} {_fmt(r['tail5'],'8.4f')} {_fmt(r['neg_prob'],'8.2f')} {int(r['obs']):5d}",
                    flush=True,
                )

        report["segments"][tag] = {"regime": dist, "portfolio_ab": ab, "double_sort": ds_records}

    if args.output:
        outp = Path(args.output)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2, default=float)
        print(f"\n[done] wrote {outp}", flush=True)
    else:
        print("\n[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
