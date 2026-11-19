"""Regime-switch walk-forward engine — frozen config (pre-registered 2026-10-05).

Frozen by the vault note
`llm-wiki/projects/trading/repos/stock-scoring-system/regime-switch-wf-validation.md`.
Do NOT change constants in this module without re-opening that note.

Frozen decisions:
- Rules (7, close-only, position in {0,1}, decided at close T, applied from T+1):
  prod_default / fast_exit / confirm3 / sma60 / sma120 / sma200 / mom12m
- Costs: 7bp (stock convention, TradeFeeModel defaults) and 1bp (ETF) per flip.
- Daily factor: f_t = 1 + p_{t-1}*r_t - c*|p_{t-1} - p_{t-2}|  (r_t = close-to-close)
- Matched-exposure control: w_bar = mean(effective exposure) in window,
  b_t = 1 + w_bar*r_t (static scaled B&H, zero cost); X(W) = sum(ln f) - sum(ln b)
- WF folds: 2006..2026 (21 annual holdouts, 2026 partial Jan-Sep).
  champion(y) = argmax X(train) with tie-break Calmar -> rule id.
- Gates: G1..G4 defined in evaluate_gates().
- G3 tops: 000852.SH @ 2021-12-01, 000300.SH @ 2025-10-29 (verified from data).
- Smoke anchors (AMENDED 2026-10-05, see vault note 修订 1): prod_default on 000852
  2021-2026 with production-padded warm-up (rows from 2020-06-15) => +8.45% cum /
  -28.4% MDD (+-1pp). The original 2026-08 anchor +8.10%/-33.5% is demoted to
  "unreproducible legacy reference" (its MDD path accounting no longer exists).
- Falsification verdict: any of G1-G4 fails => no validated switch -> static exposure.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from shared.regime_label import (
    CONFIRM_DAYS_DEFAULT,
    EXIT_CONFIRM_DAYS_DEFAULT,
    label_index_regime,
)

COST_STOCK = 0.0007  # 7bp per unit turnover -- TradeFeeModel defaults
COST_ETF = 0.0001  # 1bp per flip -- ETF convention

RULE_IDS: Sequence[str] = (
    "prod_default",
    "fast_exit",
    "confirm3",
    "sma60",
    "sma120",
    "sma200",
    "mom12m",
)

# ---- Frozen WF / gate configuration -------------------------------------
FOLD_YEARS: Sequence[int] = tuple(range(2006, 2027))  # 21 annual holdouts
DEPLOY_START = "20210101"
DEPLOY_END = "20260930"
G1_RECENT_YEARS = (2024, 2025, 2026)
G1_RECENT_WINS_REQUIRED = 2
G1_ALL_WIN_RATE_REQUIRED = 0.60
G2_MDD_FLOOR = -0.20  # deploy-window equity MDD must be >= -20%
G3_MAX_LAG_DAYS = 20
G3_TOPS = (
    # (benchmark symbol, peak trade_date) -- verified from index_prices on 2026-10-05
    ("000852.SH", "20211201"),
    ("000300.SH", "20251029"),
)
G4_BOTTOM_TWO = 2  # champion holdout rank 6 or 7 of 7

FALSIFICATION_VERDICT = "no validated switch -> static exposure satellite"

# yyyymmdd trade_date -> integer year
_Y2K = 2000


def _year_of(trade_date: str) -> int:
    return int(trade_date[:4])


def position_from_labels(labels: pd.DataFrame) -> pd.Series:
    """1.0 iff label == 'bull', aligned on the labels frame's trade_date index."""
    pos = (labels["label"] == "bull").astype(float)
    return pd.Series(pos.values, index=pd.Index(labels["trade_date"], name="trade_date"))


def _k_family_position(
    index_rows: Sequence[Dict[str, Any]],
    *,
    confirm_days: int = CONFIRM_DAYS_DEFAULT,
    exit_confirm_days: int = EXIT_CONFIRM_DAYS_DEFAULT,
    non_bull_exits: bool = False,
) -> pd.Series:
    labels = label_index_regime(
        list(index_rows),
        confirm_days=confirm_days,
        exit_confirm_days=exit_confirm_days,
        non_bull_exits=non_bull_exits,
    )
    return position_from_labels(labels)


def _confirmed_cross_up(close: pd.Series, level: pd.Series) -> pd.Series:
    """1 iff close above level today AND yesterday (1-day confirm vs alternation)."""
    above = (close > level).astype(float)
    return (above * above.shift(1)).fillna(0.0)


def sma_position(close: pd.Series, window: int) -> pd.Series:
    sma = close.rolling(window, min_periods=window).mean()
    pos = _confirmed_cross_up(close, sma)
    pos[sma.isna()] = np.nan  # warm-up days carry no position at all
    return pos


def mom_position(close: pd.Series, lookback: int = 252) -> pd.Series:
    level = close.shift(lookback)
    pos = _confirmed_cross_up(close, level)
    pos[level.isna()] = np.nan
    return pos


def compute_rule_positions(
    index_rows: Sequence[Dict[str, Any]], rule_id: str
) -> pd.Series:
    """Daily position series (index=trade_date, values 0/1, NaN=warm-up)."""
    rows = list(index_rows)
    close = _close_series(rows)
    if rule_id in ("prod_default", "fast_exit", "confirm3"):
        if rule_id == "prod_default":
            pos = _k_family_position(rows)
        elif rule_id == "fast_exit":
            pos = _k_family_position(rows, non_bull_exits=True)
        else:  # confirm3: production asymmetric exit, faster entry only
            pos = _k_family_position(rows, confirm_days=3)
    elif rule_id == "sma60":
        pos = sma_position(close, 60)
    elif rule_id == "sma120":
        pos = sma_position(close, 120)
    elif rule_id == "sma200":
        pos = sma_position(close, 200)
    elif rule_id == "mom12m":
        pos = mom_position(close, 252)
    else:
        raise ValueError(f"unknown rule id: {rule_id!r}")
    return pos


def _close_series(index_rows: Sequence[Dict[str, Any]]) -> pd.Series:
    close = pd.Series(
        [float(r["close"]) for r in index_rows],
        index=pd.Index([str(r["trade_date"]) for r in index_rows], name="trade_date"),
    ).sort_index()
    return close


def daily_factors(
    index_rows: Sequence[Dict[str, Any]],
    pos: pd.Series,
    *,
    cost_rate: float,
) -> pd.DataFrame:
    """Per-day strategy factor under the frozen T+1 convention.

    f_t = 1 + p_{t-1} * r_t - cost * |p_{t-1} - p_{t-2}|, f at the first day = 1.
    NaN positions (rule warm-up) contribute nothing and are marked invalid.
    """
    rows = list(index_rows)
    close = _close_series(rows)
    pos = pos.reindex(close.index)
    r = close.pct_change()
    p_prev = pos.shift(1)
    p_prev2 = pos.shift(2)
    flip_cost = (p_prev - p_prev2).abs() * cost_rate
    factor = 1.0 + p_prev.fillna(0.0) * r - flip_cost.fillna(0.0)
    factor = factor.where(~r.isna(), 1.0)  # first day: no position, no return -> 1
    valid = (~r.isna()) & (~p_prev.isna())
    frame = pd.DataFrame(
        {
            "close": close,
            "r": r,
            "p": pos,
            "p_prev": p_prev,
            "flip_cost": flip_cost,
            "factor": factor,
            "valid": valid,
        }
    )
    frame["equity"] = np.cumprod(frame["factor"].fillna(1.0).values)
    return frame


def merged_factors(
    index_rows: Sequence[Dict[str, Any]],
    rule_positions: Dict[str, pd.Series],
    *,
    cost_rate: float,
) -> Dict[str, pd.DataFrame]:
    """One factors frame per rule over the full history (sorted by trade_date)."""
    out: Dict[str, pd.DataFrame] = {}
    for rule_id, pos in rule_positions.items():
        out[rule_id] = daily_factors(index_rows, pos, cost_rate=cost_rate)
    return out


def _mask(factors: pd.DataFrame, start: str, end: str) -> pd.Series:
    index = factors.index
    return (index >= start) & (index <= end) & factors["valid"]


def window_summary(
    factors: pd.DataFrame, start: str, end: str
) -> Dict[str, Any]:
    """Log-growth, matched-exposure control and raw numbers over [start, end]."""
    m = _mask(factors, start, end)
    sub = factors.loc[m]
    if sub.empty:
        return {
            "n_days": 0, "log_growth": 0.0, "control": 0.0, "excess": 0.0,
            "cum_return": 0.0, "mdd": 0.0, "calmar": 0.0,
            "mean_exposure": 0.0, "flips": 0, "cost_paid": 0.0,
        }
    w_bar = float(sub["p_prev"].mean())
    log_growth = float(np.log(sub["factor"]).sum())
    control = float(np.log1p(w_bar * sub["r"]).sum())
    cum_return = float(sub["factor"].prod() - 1.0)
    equity = np.cumprod(sub["factor"].to_numpy())
    mdd = float((equity / np.maximum.accumulate(equity) - 1.0).min())
    n_years = len(sub) / 252.0
    cagr = float(math.exp(log_growth / n_years) - 1.0) if n_years > 0 else 0.0
    calmar = cagr / max(abs(mdd), 1e-12)
    flips = int((sub["p_prev"].diff().abs() > 0.5).sum())
    return {
        "n_days": int(len(sub)),
        "log_growth": log_growth,
        "control": control,
        "excess": log_growth - control,
        "cum_return": cum_return,
        "mdd": mdd,
        "calmar": calmar,
        "mean_exposure": w_bar,
        "flips": flips,
        "cost_paid": float(sub["flip_cost"].sum()),
    }


def walk_forward_eval(
    index_rows: Sequence[Dict[str, Any]],
    *,
    cost_rate: float,
    fold_years: Sequence[int] = FOLD_YEARS,
) -> Dict[str, Any]:
    """Annual expanding-window rule selection. Pure functions of past closes."""
    positions = {
        rule_id: compute_rule_positions(index_rows, rule_id) for rule_id in RULE_IDS
    }
    factors = merged_factors(index_rows, positions, cost_rate=cost_rate)
    champions: Dict[int, str] = {}
    holdouts: Dict[int, Dict[str, Any]] = {}
    for y in fold_years:
        train_end = f"{y - 1}1231"
        holdout_start, holdout_end = f"{y}0101", f"{y}1231"
        train_stats = {
            r: window_summary(f, "00010101", train_end) for r, f in factors.items()
        }
        holdout_stats = {
            r: window_summary(f, holdout_start, holdout_end) for r, f in factors.items()
        }
        pickable = [
            r for r in RULE_IDS
            if train_stats[r]["n_days"] > 0 and not math.isnan(train_stats[r]["excess"])
        ]
        champion = min(
            pickable,
            key=lambda r: (
                -train_stats[r]["excess"],
                -train_stats[r]["calmar"],
                r,
            ),
        )
        hold_x = {r: holdout_stats[r]["excess"] for r in RULE_IDS}
        ranked = sorted(RULE_IDS, key=lambda r: -hold_x[r])
        rank = ranked.index(champion) + 1
        champions[y] = champion
        holdouts[y] = {
            "champion": champion,
            "champion_train_excess": train_stats[champion]["excess"],
            "champion_holdout_excess": hold_x[champion],
            "champion_holdout_rank": rank,
            "per_rule": {
                r: {
                    "holdout_excess": hold_x[r],
                    "holdout_cum_return": holdout_stats[r]["cum_return"],
                    "mean_exposure": holdout_stats[r]["mean_exposure"],
                    "flips": holdout_stats[r]["flips"],
                }
                for r in RULE_IDS
            },
        }
    return {"champions": champions, "holdouts": holdouts, "factors": factors}


def wf_deploy_path_factors(
    factors: Dict[str, pd.DataFrame],
    champions: Dict[int, str],
    *,
    start: str,
    end: str,
    cost_rate: float,
) -> pd.DataFrame:
    """Factor series of the WF path inside [start, end], incl. year-boundary
    transition costs between successive champions (positions are 0/1)."""
    index = next(iter(factors.values())).index
    window = index[(index >= start) & (index <= end)]
    rows: List[Dict[str, Any]] = []
    prev_rule: str | None = None
    prev_p: float = 0.0
    for td in window:
        y = _year_of(str(td))
        rule = champions[y]
        f = factors[rule].loc[td]
        p_new = f["p_prev"] if not pd.isna(f["p_prev"]) else 0.0
        extra = 0.0
        if prev_rule is not None and rule != prev_rule:
            extra = cost_rate * abs(p_new - prev_p)
        factor = 1.0 + p_new * f["r"] - f["flip_cost"] - extra
        if pd.isna(f["r"]):
            factor = 1.0
        rows.append(
            {
                "trade_date": td,
                "rule": rule,
                "p": f["p"],  # close-of-day position (for G3 frozen lag semantics)
                "p_prev": p_new,  # effective exposure on day t
                "factor": float(factor),
            }
        )
        prev_rule, prev_p = rule, p_new
    frame = pd.DataFrame(rows).set_index("trade_date")
    frame["equity"] = np.cumprod(frame["factor"].to_numpy())
    return frame


def top_exit_lag(
    factors: pd.DataFrame, top_date: str, *, window_end: str = "99991231"
) -> int | None:
    """Trading days from the peak to the first day the switch reads as out.

    Frozen G3 semantics: lag = (first day d >= top with p(d) == 0) - top,
    day count in trading days, top day included (a zero position decided at
    the top's own close yields lag 0). None if the path never exits.
    """
    index = factors.index
    if top_date not in index:
        return None
    start = index.get_loc(top_date)
    for td in index[start:]:
        if td > window_end:
            return None
        p = factors.at[td, "p"]
        if not pd.isna(p) and p == 0.0:
            return int(index.get_loc(td)) - start
    return None


def evaluate_gates(
    wf: Dict[str, Any],
    *,
    primary_symbol: str,
    cost_rate: float,
    fold_years: Sequence[int] = FOLD_YEARS,
    deploy_start: str = DEPLOY_START,
    deploy_end: str = DEPLOY_END,
    crosscheck_wfs: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Mechanical G1..G4 on the WF of the primary series (000852.SH @7bp).

    G3 is pre-registered with two tops: the primary top measured on the
    primary WF deploy path; the cross-check top on the matching symbol's own
    WF deploy path (``crosscheck_wfs`` maps symbol -> that symbol's wf)."""
    primary = primary_symbol
    champions = wf["champions"]
    holdouts = wf["holdouts"]
    factors = wf["factors"]

    # G1: champion holdout excess wins
    wins: Dict[int, int] = {}
    for y in fold_years:
        wins[y] = 1 if holdouts[y]["champion_holdout_excess"] > 0 else 0
    recent_wins = sum(wins.get(y, 0) for y in G1_RECENT_YEARS)
    all_wins = sum(wins.values())
    win_rate = all_wins / max(len(wins), 1)
    g1_pass = recent_wins >= G1_RECENT_WINS_REQUIRED and win_rate >= G1_ALL_WIN_RATE_REQUIRED

    # G2: deploy-window MDD of the WF path
    path = wf_deploy_path_factors(
        factors, champions, start=deploy_start, end=deploy_end, cost_rate=cost_rate
    )
    equity = path["equity"].to_numpy()
    mdd = float((equity / np.maximum.accumulate(equity) - 1.0).min())
    g2_pass = mdd >= G2_MDD_FLOOR

    # G3: top exit lag for each pre-registered top on its own symbol's path
    paths: Dict[str, pd.DataFrame] = {primary: path}
    if crosscheck_wfs:
        for sym, wf_other in crosscheck_wfs.items():
            if sym == primary or "factors" not in wf_other:
                continue
            paths[sym] = wf_deploy_path_factors(
                wf_other["factors"], wf_other["champions"],
                start=deploy_start, end=deploy_end, cost_rate=cost_rate,
            )
    lags: Dict[str, int | None] = {}
    for sym, top in G3_TOPS:
        if sym not in paths:
            continue
        lags[sym] = top_exit_lag(paths[sym], top, window_end=deploy_end)
    g3_pass = len(lags) == len(G3_TOPS) and all(
        lag is not None and lag <= G3_MAX_LAG_DAYS for lag in lags.values()
    )

    # G4: champion holdout rank in the bottom two of 7 (rank >= 6)
    bottom_count = sum(
        1 for y in fold_years if holdouts[y]["champion_holdout_rank"] >= 7 - G4_BOTTOM_TWO + 1
    )
    g4_pass = bottom_count < len(fold_years) / 2.0

    return {
        "primary_symbol": primary,
        "g1": {
            "pass": g1_pass,
            "recent_wins": recent_wins,
            "required_recent": G1_RECENT_WINS_REQUIRED,
            "win_rate": win_rate,
            "required_rate": G1_ALL_WIN_RATE_REQUIRED,
            "wins_by_year": wins,
        },
        "g2": {
            "pass": g2_pass,
            "deploy_window": [deploy_start, deploy_end],
            "mdd": mdd,
            "floor": G2_MDD_FLOOR,
        },
        "g3": {
            "pass": g3_pass,
            "tops": [{"symbol": sym, "date": top, "lag_days": lags.get(sym)}
                     for sym, top in G3_TOPS],
        },
        "g4": {
            "pass": g4_pass,
            "bottom_two_folds": bottom_count,
            "max_allowed": (len(fold_years) - 1) // 2,
            "ranks_by_year": {y: holdouts[y]["champion_holdout_rank"] for y in fold_years},
        },
        "verdict": (
            "validated switch (in-sample forward)"
            if g1_pass and g2_pass and g3_pass and g4_pass
            else FALSIFICATION_VERDICT
        ),
    }


def smoke_anchor(
    index_rows: Sequence[Dict[str, Any]],
    *,
    start: str = "20210101",
    end: str = "20260930",
    cum_anchor: float = 0.0845,
    mdd_anchor: float = -0.284,
    tol_cum: float = 0.01,
    tol_mdd: float = 0.01,
) -> Dict[str, Any]:
    """Reproduce the known prod_default numbers (000852, 2021-2026) as a smoke test.

    Returns both 7bp and 0bp numbers; caller decides STOP vs proceed per the
    pre-registered rule (+-1pp cum)."""
    out: Dict[str, Any] = {}
    for label, rate in (("cost_7bp", COST_STOCK), ("cost_0bp", 0.0)):
        pos = compute_rule_positions(index_rows, "prod_default")
        factors = daily_factors(index_rows, pos, cost_rate=rate)
        s = window_summary(factors, start, end)
        out[label] = {
            "cum_return": s["cum_return"],
            "mdd": s["mdd"],
            "mean_exposure": s["mean_exposure"],
            "flips": s["flips"],
            "cum_ok": abs(s["cum_return"] - cum_anchor) <= tol_cum,
            "mdd_ok": abs(s["mdd"] - mdd_anchor) <= tol_mdd,
        }
    ok = (
        out["cost_7bp"]["cum_ok"]
        and out["cost_7bp"]["mdd_ok"]
    )
    out["pass"] = bool(ok)
    return out