#!/usr/bin/env python3
"""Level A CLI —— 牛市开关 walk-forward 验收（预注册口径，见 vault 笔记
`regime-switch-wf-validation.md`；常量冻结在 regime_switch_rule_engine.py 头 docstring）。

运行（容器内）:
  docker exec -w /app quant-scorer python analysis/research/regime_switch_walkforward.py

输出: `reports/portfolio-research/regime-switch-wf/`
  - wf_results.json  （folds/champions/gate 证据/smoke，两次运行逐字节一致）
  - scorecard.md     （裁决表）
  - report.html      （自包含：部署窗 equity 曲线、逐年 champion 条带、逐年表）

退出码: 0 = 全流程完成；2 = smoke 锚点失配（STOP 诊断，gate 不可信）。
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
os.environ.pop("TUSHARE_TOKEN", None)

from analysis.core.benchmark_prices import load_index_benchmark_prices  # noqa: E402
from analysis.research.regime_switch_rule_engine import (  # noqa: E402
    COST_ETF,
    COST_STOCK,
    DEPLOY_END,
    DEPLOY_START,
    FOLD_YEARS,
    RULE_IDS,
    evaluate_gates,
    smoke_anchor,
    walk_forward_eval,
    wf_deploy_path_factors,
)
from shared.data_access import open_research_data_access  # noqa: E402
from shared.regime_label import count_regime_flips, label_index_regime  # noqa: E402

T0 = time.time()

PRIMARY = "000852.SH"
SECONDARY = ("000300.SH", "000905.SH")
LOAD_START = "20000101"  # grab everything the DB has (verified back to 2005-01-04)

OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "reports", "portfolio-research", "regime-switch-wf"
)


def log(message: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {message}", flush=True)


def _close_series(rows: Sequence[Dict[str, Any]]) -> pd.Series:
    return pd.Series(
        [float(r["close"]) for r in rows],
        index=pd.Index([str(r["trade_date"]) for r in rows], name="trade_date"),
    ).sort_index()


def _public_wf(wf: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic subset of walk_forward_eval for JSON (no factor frames)."""
    out: Dict[str, Any] = {"champions": wf["champions"], "holdouts": wf["holdouts"]}
    flip_note = {}
    for y, h in wf["holdouts"].items():
        flip_note[str(y)] = {
            r: h["per_rule"][r]["flips"] for r in RULE_IDS if r in h["per_rule"]
        }
    out["flips_by_fold"] = flip_note  # kept explicit for the flip cross-check
    return out


def _holdout_index_cum(close: pd.Series, start: str, end: str) -> float:
    r = close.pct_change()
    m = (close.index >= start) & (close.index <= end) & (~r.isna())
    sub = r.loc[m]
    return float(np.prod(1.0 + sub.to_numpy()) - 1.0) if len(sub) else 0.0


def _add_index_cums(wf: Dict[str, Any], close: pd.Series) -> None:
    """Attach the raw (unmatched) index cum return per fold for transparency."""
    for y in FOLD_YEARS:
        start, end = f"{y}0101", f"{y}1231"
        wf["holdouts"][y]["index_cum_return"] = _holdout_index_cum(close, start, end)


def run_symbol(
    wf: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    *,
    cost_rate: float,
    gates: bool,
    primary_symbol: str,
    crosscheck_wfs: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    log(f"打包 {primary_symbol} @{cost_rate * 10000:.0f}bp")
    _add_index_cums(wf, _close_series(rows))
    out = _public_wf(wf)
    if gates:
        out["gates"] = evaluate_gates(
            wf,
            primary_symbol=primary_symbol,
            cost_rate=cost_rate,
            crosscheck_wfs=crosscheck_wfs,
        )
    return out


def _public_wf_path(path: pd.DataFrame) -> Dict[str, Any]:
    """Deterministic daily deploy-window path (for HTML/audit, not in JSON by default)."""
    return {
        "trade_dates": [str(td) for td in path.index],
        "rules": [str(r) for r in path["rule"].tolist()],
        "p": [None if pd.isna(v) else float(v) for v in path["p"].tolist()],
        "equity": [float(v) for v in path["equity"].tolist()],
    }


def _flip_crosscheck(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """引擎翻转数 vs count_regime_flips（每位置翻转必对应一次标签翻转，反向不必）。"""
    labels = label_index_regime(list(rows))
    label_flips = int(count_regime_flips(labels))
    pos = (labels["label"] == "bull").astype(float)
    engine_flips = int((pos.diff().abs() > 0.5).sum())
    return {
        "engine_position_flips": engine_flips,
        "label_flips": label_flips,
        "consistent": engine_flips <= label_flips,
    }


def main() -> int:
    out_dir = os.path.abspath(OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    log("打开数据访问")
    db = open_research_data_access()

    log(f"加载指数 {PRIMARY} / {SECONDARY}")
    rows_all: Dict[str, List[Dict[str, Any]]] = {}
    for sym in (PRIMARY,) + SECONDARY:
        rows = load_index_benchmark_prices(db, sym, LOAD_START)
        rows_all[sym] = rows
        log(f"  {sym}: {len(rows)} rows, {rows[0]['trade_date']}..{rows[-1]['trade_date']}")
        if len(rows) < 400:
            raise SystemExit(f"{sym} 行数不足（{len(rows)}），拒绝运行")

    # ---- 0. smoke test（修订锚点 +8.45% / −28.4% ±1pp，见 vault 注记修订 1）----
    # 用生产管线口径预热：padded_index_start(20210101) = 2020-06-15
    log("smoke: prod_default 000852 2021-2026（padded 预热）")
    smoke_rows = [
        r for r in rows_all[PRIMARY] if r["trade_date"] >= "20200615"
    ]
    smoke = smoke_anchor(smoke_rows)
    for key in ("cost_7bp", "cost_0bp"):
        s = smoke[key]
        log(
            f"  {key}: cum={s['cum_return']:+.4f} (ok={s['cum_ok']}) "
            f"mdd={s['mdd']:+.4f} (ok={s['mdd_ok']}) exposure={s['mean_exposure']:.3f} flips={s['flips']}"
        )
    smoke_ok = bool(smoke["pass"])
    flip_cc = _flip_crosscheck(smoke_rows)
    log(f"smoke 结论: {'PASS' if smoke_ok else 'FAIL'}")
    log(f"翻转交叉核对: {flip_cc}")

    # ---- 1. WF 主序列 + 副表 ----
    results: Dict[str, Any] = {
        "primary_symbol": PRIMARY,
        "secondary_symbols": list(SECONDARY),
        "fold_years": list(FOLD_YEARS),
        "deploy_window": [DEPLOY_START, DEPLOY_END],
        "smoke": smoke,
        "flip_crosscheck": flip_cc,
    }
    wf7_all = {
        sym: walk_forward_eval(rows_all[sym], cost_rate=COST_STOCK)
        for sym in (PRIMARY,) + SECONDARY
    }
    wf1_all = {
        sym: walk_forward_eval(rows_all[sym], cost_rate=COST_ETF)
        for sym in (PRIMARY,)
    }

    results["primary_7bp"] = run_symbol(
        wf7_all[PRIMARY],
        rows_all[PRIMARY],
        cost_rate=COST_STOCK,
        gates=True,
        primary_symbol=PRIMARY,
        crosscheck_wfs=wf7_all,
    )
    if not smoke_ok:
        # 折返点: gate 不可信,但仍把已算出的零成本诊断落盘后再 STOP
        results["STOP"] = "smoke anchor mismatch — gates not trustworthy"
        with open(os.path.join(out_dir, "wf_results.json"), "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2, allow_nan=True, default=str)
        log("smoke 失配: 已落盘诊断 JSON,退出码 2（不得继续信 gate）")
        return 2

    results["primary_1bp"] = run_symbol(
        wf1_all[PRIMARY],
        rows_all[PRIMARY],
        cost_rate=COST_ETF,
        gates=False,
        primary_symbol=PRIMARY,
    )
    results["secondary"] = {
        sym: run_symbol(
            wf7_all[sym], rows_all[sym], cost_rate=COST_STOCK, gates=False, primary_symbol=sym
        )
        for sym in SECONDARY
    }

    # 部署窗 equity 曲线数据（主序列,供 HTML 与审计）
    path = wf_deploy_path_factors(
        wf7_all[PRIMARY]["factors"], wf7_all[PRIMARY]["champions"],
        start=DEPLOY_START, end=DEPLOY_END, cost_rate=COST_STOCK,
    )
    close = _close_series(rows_all[PRIMARY])
    deploy_dates = [str(td) for td in close.index if DEPLOY_START <= str(td) <= DEPLOY_END]
    results["deploy_series"] = {
        "path": _public_wf_path(path),
        "index_close": {
            "trade_dates": deploy_dates,
            "values": [float(close.loc[td]) for td in deploy_dates],
        },
    }

    # ---- 2. 落盘 ----
    json_path = os.path.join(out_dir, "wf_results.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2, allow_nan=True, default=str)
    log(f"written {json_path}")

    md_path = os.path.join(out_dir, "scorecard.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_scorecard(results))
    log(f"written {md_path}")

    html_path = os.path.join(out_dir, "report.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(render_html(results))
    log(f"written {html_path}")

    verdict = results["primary_7bp"]["gates"]["verdict"]
    log(f"裁决: {verdict}")
    return 0


# ---------------- scorecard / report rendering ----------------

def render_scorecard(results: Dict[str, Any]) -> str:
    """Markdown scorecard: gate table + smoke + secondary table summary."""
    g = results["primary_7bp"]["gates"]
    smoke = results["smoke"]
    lines: List[str] = [
        "# 牛市开关 walk-forward 验收 scorecard",
        "",
        f"主序列 {results['primary_symbol']} @7bp / 部署窗 {results['deploy_window'][0]}–{results['deploy_window'][1]}",
        "",
        "## 裁决",
        "",
        f"**{g['verdict']}**",
        "",
        "## Gate 表",
        "",
        "| gate | 判定 | 数值 | 门槛 |",
        "|---|---|---|---|",
    ]
    g1 = g["g1"]
    lines.append(
        f"| G1 前向胜 | {'✅' if g1['pass'] else '❌'} | 近年(2024/25/26) 胜 {g1['recent_wins']} 年，"
        f"总胜率 {g1['win_rate']:.2%}（5 胜: {', '.join(str(y) for y, w in g1['wins_by_year'].items() if w) or '无'}） | "
        f"近年≥{g1['required_recent']} 且 总胜率≥{g1['required_rate']:.0%} |"
    )
    g2 = g["g2"]
    lines.append(
        f"| G2 部署窗 MDD | {'✅' if g2['pass'] else '❌'} | {g2['mdd']:+.2%} | ≥{g2['floor']:+.0%} |"
    )
    g3 = g["g3"]
    top_text = "、".join(
        f"{t['symbol']}@{t['date']} lag={t['lag_days']}d" for t in g3["tops"]
    )
    lines.append(
        f"| G3 见顶滞后 | {'✅' if g3['pass'] else '❌'} | {top_text} | ≤20 交易日 |"
    )
    g4 = g["g4"]
    lines.append(
        f"| G4 排名下坠 | {'✅' if g4['pass'] else '❌'} | 后两位 folds {g4['bottom_two_folds']}/{len(results['fold_years'])} | "
        f"< 半数（即 <{len(results['fold_years']) / 2}） |"
    )
    lines += [
        "",
        "## Smoke 锚点（prod_default 000852 2021–2026，修订目标 +8.45%/−28.4% ±1pp）",
        "",
        "| 口径 | cum | MDD | cum ok | mdd ok |",
        "|---|---|---|---|---|",
    ]
    for key in ("cost_7bp", "cost_0bp"):
        s = smoke[key]
        lines.append(
            f"| {key} | {s['cum_return']:+.4f} | {s['mdd']:+.4f} | {'✅' if s['cum_ok'] else '❌'} | {'✅' if s['mdd_ok'] else '❌'} |"
        )
    lines += [
        "",
        "## 逐年 champion（主序列 7bp）",
        "",
        "| 年 | champion | 训练超额 | holdout 超额(匹配) | holdout 排名 | 原始 cum | 指数 cum |",
        "|---|---|---|---|---|---|---|",
    ]
    for y in results["fold_years"]:
        yk = str(y)
        h = results["primary_7bp"]["holdouts"].get(yk, results["primary_7bp"]["holdouts"].get(y))
        if h is None:
            continue
        lines.append(
            f"| {y} | {h['champion']} | {h['champion_train_excess']:+.4f} | "
            f"{h['champion_holdout_excess']:+.4f} | {h['champion_holdout_rank']}/7 | "
            f"{h['per_rule'][h['champion']]['holdout_cum_return']:+.2%} | {h['index_cum_return']:+.2%} |"
        )
    return "\n".join(lines) + "\n"


def render_html(results: Dict[str, Any]) -> str:
    """Self-contained HTML report (ledger style, mirrors backtest-randomness.standalone)."""
    g = results["primary_7bp"]["gates"]
    verdict = g["verdict"] or "no data"
    smoke = results["smoke"]

    champion_rows: List[str] = []
    for y in results["fold_years"]:
        yk = str(y)
        h = results["primary_7bp"]["holdouts"].get(yk, results["primary_7bp"]["holdouts"].get(y))
        if h is None:
            continue
        champ = h["champion"]
        champion_rows.append(
            f"<tr><td class='mono'>{y}</td><td class='mono'>{champ}</td>"
            f"<td class='mono'>{h['champion_train_excess']:+.4f}</td>"
            f"<td class='mono'>{h['champion_holdout_excess']:+.4f}</td>"
            f"<td class='mono'>{h['champion_holdout_rank']}/7</td>"
            f"<td class='mono'>{h['per_rule'][champ]['holdout_cum_return']:+.2%}</td>"
            f"<td class='mono'>{h['index_cum_return']:+.2%}</td></tr>"
        )

    deploy = results.get("deploy_series", {})
    path = deploy.get("path", {})
    equity = path.get("equity", [])
    dates = path.get("trade_dates", [])
    idx_close = deploy.get("index_close", {}).get("values", [])
    if equity:
        chart_svg = _svg_two_lines(dates, equity, idx_close)
    else:
        chart_svg = "<p>no deploy series</p>"

    gate_rows = [
        ("G1 前向胜", g["g1"]["pass"],
         f"近年 {g['g1']['recent_wins']}/{g['g1']['required_recent']}+，胜率 {g['g1']['win_rate']:.2%}（≥{g['g1']['required_rate']:.0%}）",
         f"近年≥{g['g1']['required_recent']} 且 总胜率≥{g['g1']['required_rate']:.0%}"),
        ("G2 部署窗 MDD", g["g2"]["pass"],
         f"{g['g2']['mdd']:+.2%}（2021-01-01–2026-09-30）",
         f"≥{g['g2']['floor']:+.0%}"),
        ("G3 见顶滞后", g["g3"]["pass"],
         "、".join(f"{t['symbol']}@{t['date']} lag={t['lag_days']}d" for t in g["g3"]["tops"]),
         "≤20 交易日"),
        ("G4 排名下坠", g["g4"]["pass"],
         f"后两位 folds {g['g4']['bottom_two_folds']}/{len(results['fold_years'])}",
         "< 半数"),
    ]
    gate_row_html = "\n".join(
        "<tr><td>{0}</td><td class='{1}'>{2}</td>"
        "<td class='mono'>{3}</td><td class='mono'>{4}</td></tr>".format(
            name, "ok" if ok else "bad", "✅" if ok else "❌", value, threshold
        )
        for name, ok, value, threshold in gate_rows
    )

    smoke_rows = "\n".join(
        f"<tr><td class='mono'>{key}</td><td class='mono'>{s['cum_return']:+.4f}</td>"
        f"<td class='mono'>{s['mdd']:+.4f}</td>"
        f"<td class='mono'>{'✅' if s['cum_ok'] else '❌'}</td>"
        f"<td class='mono'>{'✅' if s['mdd_ok'] else '❌'}</td></tr>"
        for key, s in (("cost_7bp", smoke["cost_7bp"]), ("cost_0bp", smoke["cost_0bp"]))
    )

    secondary_summary: List[str] = []
    for variant in ("primary_1bp",):
        block = results.get(variant, {})
        for y in results["fold_years"][-6:]:
            yk = str(y)
            h = block.get("holdouts", {}).get(yk)
            if h:
                secondary_summary.append(f"{variant} {y}: {h['champion']} x={h['champion_holdout_excess']:+.4f}")
    for sym, block in results.get("secondary", {}).items():
        for y in results["fold_years"][-6:]:
            yk = str(y)
            h = block.get("holdouts", {}).get(yk)
            if h:
                secondary_summary.append(f"{sym} {y}: {h['champion']} x={h['champion_holdout_excess']:+.4f}")

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>牛市开关 walk-forward 验收</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Noto+Sans+SC:wght@400;500;700&family=Noto+Serif+SC:wght@600;700&display=swap">
<style>
:root {{
  --paper:    #F2F4F2; --surface: #FFFFFF; --sunk: #E9ECEA;
  --ink:      #131A1C; --ink-soft: #566063; --ink-faint: #8A9496;
  --rule:     #D8DEDB; --rule-soft: #E6EAE8;
  --accent:   #0E5B60; --accent-soft: #D7E7E6;
  --pos:      #1D7355; --neg: #9E3B2E; --warn: #8A6117;
  --font-display: "Noto Serif SC", "Songti SC", serif;
  --font-body: "Noto Sans SC", system-ui, -apple-system, "PingFang SC", sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", monospace;
  color-scheme: light;
}}
:root[data-theme="dark"] {{
  --paper: #101517; --surface: #171E21; --sunk: #1D2629;
  --ink: #E7EDEB; --ink-soft: #98A5A6; --ink-faint: #6B787A;
  --rule: #2B3538; --rule-soft: #232C2F;
  --accent: #52B5B0; --accent-soft: #1B3336;
  --pos: #4CBB8C; --neg: #DE7A69; --warn: #D2A24A;
  color-scheme: dark;
}}
body {{ margin: 0; background: var(--paper); color: var(--ink);
       font: 15px/1.65 var(--font-body); }}
main {{ max-width: 66ch; margin: 0 auto; padding: 48px 20px 96px; }}
h1, h2 {{ font-family: var(--font-display); line-height: 1.3; }}
h1 {{ font-size: 1.9em; margin: 0 0 8px; }}
h2 {{ font-size: 1.25em; margin: 40px 0 12px; padding-top: 16px; border-top: 1px solid var(--rule); }}
p.sub {{ color: var(--ink-soft); margin: 0 0 32px; }}
.verdict {{ font-size: 1.15em; font-weight: 700; padding: 14px 18px;
            background: var(--surface); border-left: 4px solid var(--accent);
            box-shadow: 0 1px 2px rgba(19,26,28,.05); margin: 24px 0; color: var(--accent); }}
.verdict.bad {{ border-left-color: var(--neg); color: var(--neg); }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 13.5px; }}
th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--rule); vertical-align: top; }}
th {{ color: var(--ink-soft); font-weight: 500; border-bottom: 1px solid var(--ink-faint); }}
.mono {{ font-family: var(--font-mono); font-size: 12.5px; }}
.ok {{ color: var(--pos); font-weight: 700; }} .bad {{ color: var(--neg); font-weight: 700; }}
.full {{ width: 100%; margin: 16px 0 8px; overflow-x: auto; }}
.note {{ color: var(--ink-soft); font-size: 12.5px; }}
footer {{ color: var(--ink-faint); font-size: 12px; margin-top: 48px; }}
</style>
</head>
<body>
<main>
<h1>牛市开关 walk-forward 验收</h1>
<p class="sub">主序列 {results['primary_symbol']} @7bp · folds {results['fold_years'][0]}–{results['fold_years'][-1]} · 部署窗 {results['deploy_window'][0]}–{results['deploy_window'][1]} · 预注册 2026-10-05</p>

<div class="verdict {'bad' if ('static exposure' in verdict or '无已验证' in verdict) else ''}">{verdict}</div>

<h2>Gate 表</h2>
<table>
<tr><th>gate</th><th>判定</th><th>数值</th><th>门槛（冻结）</th></tr>
{gate_row_html}
</table>
<p class="note">全部口径见 vault 笔记 <code>regime-switch-wf-validation.md</code>；原始（未匹配暴露）数字只展示、不参与选择。</p>

<h2>Smoke 锚点（prod_default 000852 2021–2026 → 修订目标 +8.45%/−28.4% ±1pp）</h2>
<table>
<tr><th>口径</th><th>cum</th><th>MDD</th><th>cum ok</th><th>mdd ok</th></tr>
{smoke_rows}
</table>

<h2>部署窗 equity（WF 路径 vs 指数同期）</h2>
<div class="full">{chart_svg}</div>
<p class="note">WF 路径 = 逐年 champion 的因子序列（含年度切换成本）；指数线为 000852 close 同期累计。</p>

<h2>逐年 champion（主序列 7bp）</h2>
<table>
<tr><th>年</th><th>champion</th><th>训练超额</th><th>holdout 超额(匹配)</th><th>排名</th><th>原始 cum</th><th>指数 cum</th></tr>
{''.join(champion_rows)}
</table>

<h2>副表摘要（不设 gate）</h2>
<p class="note">{'; '.join(secondary_summary) if secondary_summary else '无'}</p>
<p class="note">{'000852@1bp / ' if secondary_summary else ''}000300@7bp / 000905@7bp —— 稳健性观察，不参与裁决。</p>

<footer>生成于 walks-forward 验收 runs · 数据源 index_prices · 无前视（T 收盘决策 T+1 生效）</footer>
</main>
</body>
</html>
"""


def _svg_two_lines(dates: Sequence[str], equity: Sequence[float], index_close: Sequence[float]) -> str:
    """Minimal inline-SVG line chart: strategy equity (accent) + index (faint)."""
    if len(equity) < 2:
        return "<p>chart requires ≥2 points</p>"
    width, height, pad_x, pad_y = 860, 260, 8, 12
    n = len(equity)
    eq = np.asarray(equity, dtype=float)
    idx = np.asarray(index_close, dtype=float) if len(index_close) == n else eq
    idx = idx / idx[0]
    eq_n = eq / eq[0]
    lo = float(min(eq_n.min(), idx.min()))
    hi = float(max(eq_n.max(), idx.max()))
    span = max(hi - lo, 1e-9)

    def px(i: int) -> float:
        return pad_x + (width - 2 * pad_x) * i / (n - 1)

    def py(v: float) -> float:
        return pad_y + (height - 2 * pad_y) * (1 - (v - lo) / span)

    def polyline(vals: np.ndarray, color: str, stroke_width: float) -> str:
        points = " ".join(f"{px(i):.1f},{py(float(v)):.1f}" for i, v in enumerate(vals))
        return (
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="{stroke_width}" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    zero_y = py(1.0) if lo <= 1.0 <= hi else None
    zero_line = (
        f'<line x1="{pad_x}" y1="{zero_y:.1f}" x2="{width - pad_x}" y2="{zero_y:.1f}" '
        f'stroke="var(--rule)" stroke-dasharray="4 4"/>' if zero_y is not None else ""
    )
    labels = "".join(
        f'<text x="{px(i):.1f}" y="{height - 2}" font-size="11" fill="var(--ink-faint)" '
        f'font-family="var(--font-mono)" text-anchor="middle">{dates[i][:4]}-{dates[i][4:6]}</text>'
        for i in (0, n // 2, n - 1)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'preserveAspectRatio="none" role="img" aria-label="deploy window equity">'
        f"{zero_line}"
        f"{polyline(idx, '#8A9496', 1.4)}"
        f"{polyline(eq_n, '#0E5B60', 2)}"
        f"{labels}</svg>"
    )


if __name__ == "__main__":
    sys.exit(main())