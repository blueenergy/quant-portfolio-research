"""Hand-computed acceptance tests for the frozen regime-switch engine.

Math under test (pre-registered in regime-switch-wf-validation.md):
  f_t = 1 + p_{t-1}*r_t - c*|p_{t-1} - p_{t-2}|   (T close decision, T+1 credit)
  control b_t = 1 + w_bar*r_t,  X = sum(ln f) - sum(ln b)
"""
from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from analysis.research.regime_switch_rule_engine import (
    COST_STOCK,
    daily_factors,
    sma_position,
    top_exit_lag,
    walk_forward_eval,
    wf_deploy_path_factors,
    window_summary,
)


def _rows(close_values: List[float], *, start: str = "20200106") -> List[Dict]:
    dates = pd.date_range(start, periods=len(close_values), freq="B")
    return [
        {"trade_date": d.strftime("%Y%m%d"), "close": float(c)}
        for d, c in zip(dates, close_values)
    ]


def _pos(rows: List[Dict], values: List[float]) -> pd.Series:
    return pd.Series(values, index=pd.Index([r["trade_date"] for r in rows]))


def test_t1_credit_and_flip_cost_once():
    # close: day1 100, day2 100, day3 110 (+10% credited only if in from day2 close)
    rows = _rows([100.0, 100.0, 110.0, 100.0, 100.0, 100.0])
    pos = _pos(rows, [0.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # enter day2 close
    frame = daily_factors(rows, pos, cost_rate=0.0)
    factors = frame["factor"].tolist()
    # day1: no position (p decided at day1 close applies from day2) -> r NaN -> 1
    assert factors[0] == pytest.approx(1.0)
    # day2: still flat (position applied from day3)
    assert factors[1] == pytest.approx(1.0)
    # day3: +10% credited -> 1.1
    assert factors[2] == pytest.approx(1.1)
    # flat afterwards
    assert factors[4] == pytest.approx(1.0)
    assert factors[5] == pytest.approx(1.0)

    # day3: -9.09% still held -> 1.1 * (1 - 0.090909) -> cumulative 1.0 (roundtrip)
    assert factors[3] == pytest.approx(0.9090909090909091, abs=1e-12)
    # with cost: exactly one flip (0->1 decided at day2 close, paid on day3), charged once
    frame_c = daily_factors(rows, pos, cost_rate=COST_STOCK)
    charged = frame_c[frame_c["flip_cost"] > 0]
    assert len(charged) == 1
    assert charged["flip_cost"].iloc[0] == pytest.approx(COST_STOCK)
    assert charged.index[0] == "20200108"  # day3


def test_always_in_excess_is_zero():
    rows = _rows([100.0, 101.0, 100.5, 103.0, 104.0])
    pos = _pos(rows, [1.0] * 5)
    s = window_summary(daily_factors(rows, pos, cost_rate=0.0), "20200106", "20200110")
    # p==1 with zero cost: strategy grows exactly at the index rate
    assert s["excess"] == pytest.approx(0.0, abs=1e-12)
    assert s["mean_exposure"] == pytest.approx(1.0)
    assert s["flips"] == 0
    assert s["cost_paid"] == pytest.approx(0.0)


def test_always_out_excess_is_zero():
    rows = _rows([100.0, 101.0, 100.5, 103.0, 104.0])
    pos = _pos(rows, [0.0] * 5)
    s = window_summary(daily_factors(rows, pos, cost_rate=0.0), "20200106", "20200110")
    assert s["excess"] == pytest.approx(0.0, abs=1e-12)
    assert s["log_growth"] == pytest.approx(0.0, abs=1e-12)
    assert s["cum_return"] == pytest.approx(0.0, abs=1e-12)
    assert s["mdd"] == pytest.approx(0.0, abs=1e-12)


def test_matched_exposure_control_is_scaled_bh():
    # In only on the single +10% day (entered at day2 close -> held on day3).
    # Control = static weighted B&H at the strategy's own average exposure:
    # it compares the timing choice against "same money in the index all along".
    rows = _rows([100.0, 100.0, 110.0, 110.0, 110.0])
    pos = _pos(rows, [0.0, 1.0, 0.0, 0.0, 0.0])
    s = window_summary(daily_factors(rows, pos, cost_rate=0.0), "20200106", "20200110")
    # effective exposure: day3 only (p_prev=1), 4 valid days -> w_bar = 0.25
    assert s["mean_exposure"] == pytest.approx(0.25)
    # strategy log growth = ln(1.1); control = ln(1 + 0.25*0.1) = ln(1.025)
    assert s["log_growth"] == pytest.approx(np.log(1.1))
    assert s["control"] == pytest.approx(np.log(1.025))
    assert s["excess"] == pytest.approx(
        np.log(1.1) - np.log(1.025), abs=1e-12
    )
    assert s["cum_return"] == pytest.approx(0.10)


def test_sma_confirmed_cross_hand_computed():
    # window 3: SMA(3) at day3 = (1+2+3)/3 = 2 < close 3 but day2 close vs NaN -> 0
    # day4: close 4 > SMA 3 and close 3 > SMA 2 -> first confirmed 1
    rows = _rows([1.0, 2.0, 3.0, 4.0, 3.0, 5.0])
    close = pd.Series(
        [r["close"] for r in rows],
        index=pd.Index([r["trade_date"] for r in rows]),
    ).sort_index()
    pos = sma_position(close, window=3)
    assert np.isnan(pos.iloc[0]) and np.isnan(pos.iloc[1])  # warm-up
    assert pos.iloc[2] == 0.0
    assert pos.iloc[3] == 1.0  # day4 first confirm
    assert pos.iloc[4] == pytest.approx(0.0)  # close 3 < sma 3.33 -> out immediately


def test_top_exit_lag_frozen_semantics():
    rows = _rows([100.0, 101.0, 105.0, 104.0, 103.0, 102.0, 101.0])
    # peak at day3; positions: stay in through the peak, exit at close of day5
    pos = _pos(rows, [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    factors = daily_factors(rows, pos, cost_rate=0.0)
    top = rows[2]["trade_date"]  # 20200108
    # first day d >= top with p(d)==0 is day5 -> lag = 5 - 3 = 2 trading days
    assert top == "20200108"
    assert rows[4]["trade_date"] == "20200110"
    assert top_exit_lag(factors, top) == 2
    # exit decided at the peak's own close -> lag 0
    pos_at_top = _pos(rows, [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert top_exit_lag(daily_factors(rows, pos_at_top, cost_rate=0.0), top) == 0
    # never exits -> None
    pos_hold = _pos(rows, [1.0] * 7)
    assert top_exit_lag(daily_factors(rows, pos_hold, cost_rate=0.0), top) is None
    # unknown date -> None
    assert top_exit_lag(factors, "19990101") is None


def test_walk_forward_deterministic_and_consistent():
    # ~3 years of monotone-up / down segments lets K-family and SMA rules fill in
    rng = np.random.default_rng(42)
    dates = pd.date_range("20050103", periods=780, freq="B")
    closes: List[float] = []
    level = 1000.0
    for i in range(780):
        level *= 1 + rng.normal(0.0003, 0.012)
        closes.append(level)
    rows = [
        {"trade_date": d.strftime("%Y%m%d"), "close": float(c)}
        for d, c in zip(dates, closes)
    ]
    fold_years = (2006, 2007)
    wf1 = walk_forward_eval(rows, cost_rate=COST_STOCK, fold_years=fold_years)
    wf2 = walk_forward_eval(rows, cost_rate=COST_STOCK, fold_years=fold_years)
    assert wf1["champions"] == wf2["champions"]
    for y in fold_years:
        assert wf1["holdouts"][y]["champion"] == wf2["holdouts"][y]["champion"]
        assert wf1["holdouts"][y]["champion_holdout_excess"] == pytest.approx(
            wf2["holdouts"][y]["champion_holdout_excess"]
        )
        assert 1 <= wf1["holdouts"][y]["champion_holdout_rank"] <= 7
    # per-rule factor frames agree day by day
    for rule, frame1 in wf1["factors"].items():
        frame2 = wf2["factors"][rule]
        assert frame1.index.equals(frame2.index)
        assert np.allclose(
            frame1["factor"].fillna(0).to_numpy(),
            frame2["factor"].fillna(0).to_numpy(),
        )
    # deploy-path factor series is deterministic too
    p1 = wf_deploy_path_factors(
        wf1["factors"], wf1["champions"],
        start="20070101", end="20071231", cost_rate=COST_STOCK,
    )
    p2 = wf_deploy_path_factors(
        wf2["factors"], wf2["champions"],
        start="20070101", end="20071231", cost_rate=COST_STOCK,
    )
    assert np.allclose(p1["factor"].to_numpy(), p2["factor"].to_numpy())


def test_training_uses_only_prior_years():
    # A giant late-window move must not change a champion whose training
    # window ends before it (walk-forward = expanding window over the past).
    # 255 business days from 2005-01-03 land entirely in 2005 (through ~2005-12-26),
    # then 10 business days from 2006-01-02 form fold 2006's holdout.
    d2005 = pd.bdate_range("20050103", periods=255)
    d2006 = pd.bdate_range("20060102", periods=10)
    rows_a = [
        {"trade_date": d.strftime("%Y%m%d"), "close": 100.0}
        for d in d2005.append(d2006)
    ]
    rows_b = copy.deepcopy(rows_a)
    for r in rows_b[-10:]:
        r["close"] = 500.0  # move strictly inside the 2006 holdout
    folds = (2006,)
    wf_a = walk_forward_eval(rows_a, cost_rate=0.0, fold_years=folds)
    wf_b = walk_forward_eval(rows_b, cost_rate=0.0, fold_years=folds)
    # flat series -> every rule has X=0 in the 2005 training window; the
    # tie-break picks the same id for both runs regardless of the holdout move
    assert wf_a["champions"][2006] == wf_b["champions"][2006]
    assert wf_a["champions"][2006] == "confirm3"  # lexicographic tie-break
    # the holdout consequences may differ, but the choice rule must not see them
    assert not np.isclose(
        wf_a["holdouts"][2006]["champion_holdout_excess"],
        wf_b["holdouts"][2006]["champion_holdout_excess"],
    )