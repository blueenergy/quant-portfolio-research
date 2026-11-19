# analysis/research — 组合研究 / 因子门禁工具箱

这里的模块**不是**一次性脚本，绝大部分带 `argparse`，是参数化的 CLI。
零 import 引用是设计使然（人手工跑），不是废弃信号 —— 不要用引用数判断能否删除。

**2026-11-19 从 `stock-scoring-system` 迁入**（35 个模块 + `market_spectrum/`）。
留在 scoring 的 14 个脚本依赖评分域代码（`data_manager` / `scorers/` /
`entropy/` / `daily_gold/`），本仓库没有那些层，它们随未来的 `quant-scorer` 走。

## 契约

`analysis/research/` 之外的任何代码不得 import 这里的东西。研究脚本可以
随时改写或删除，被生产 import 就等于把草稿变成了必须维护的 API。生产需要
的能力应提升到 `analysis/core/` 或 `analysis/factor/` 成为公开名字。
测试可以 import。

## 一条因子从提出到上线

    准备数据  ->  基线/验收  ->  诊断归因  ->  A/B 门禁  ->  接入生产

### 1. 准备数据

| 模块 | 作用 |
|---|---|
| `precompute_factor_overlay.py` | 把 `stock_scores` 里已有因子列导成 overlay，可反号 |

### 2. 基线与验收

| 模块 | 作用 |
|---|---|
| `growth_factor_baseline.py` | growth 因子 Phase-0 基线 |
| `entropy_validation.py` | 验证已落库的 entropy 分数 |
| `candidate_factor_gate.py` | 判断候选风险因子是真选股 alpha 还是伪装 |
| `moneyflow_elg_lg_ic.py` | 原始 elg+lg（非 `money_flow_score`）Phase-0 Rank IC / ICIR |
| `score_ic_by_year_regime.py` | 逐年 / 分市况 Rank IC、ICIR 与 horizon decay |
| `score_top10_subset_ic.py` | Top10 子集的 IC 检验 |

### 3. 诊断归因

| 模块 | 作用 |
|---|---|
| `ma_stack_diagnosis.py` | Top-N 的均线排列分布 |
| `path_gate_diagnosis.py` | Top-N 是否超配下跌趋势股 |
| `industry_slice_diagnosis.py` | 申万 L1 行业内 IC / Top-K 分面诊断 |
| `index_benchmark_diagnostics.py` | 指数基准与买入持有的收益差从哪来 |
| `growth_cycle_synergy_check.py` | 双因子协同是正交增量还是起点运气 |
| `cycle_parts_weight_sweep.py` | cycle 短/中/长三段子分的权重离线重组 |
| `regime_weight_diagnosis.py` | 指数 MA 三态标签分布 / 翻转率 / regime_segments |
| `growth_exposure_diagnostics.py` | growth 暴露诊断 |
| `industry_regime_ab_gate.py` | 行业 regime 个股仓位覆盖层门禁（**结论：否决**） |

### 4. A/B 门禁

| 模块 | 作用 |
|---|---|
| `path_gate_ab_gate.py` | 读 sweep 结果 json，判定是否达到上线标准 |
| `volatility_ab_gate.py` | 同上，波动率仓位缩放 |
| `industry_ab_gate.py` | 全市场 vs 申万 L1 过滤 sweep 对比 |
| `run_industry_ab_gate.py` | 驱动 baseline + 行业过滤 sweep 并调用上面的 gate |
| `run_industry_deep_sweep.py` | 过关行业的扩展 sweep 与汇总 |
| `run_volatility_ab_gate.py` | 驱动 sweep 并调用上面的 gate |
| `regime_weight_ab_gate.py` | 静态 20:80 / 60:40 vs `--regime-weight` 三窗门禁 |
| `run_regime_weight_ab_gate.py` | 驱动 full/bear/bull sweep 并调用上面的 gate |
| `run_non_bull_half_topn_ab.py` | 非牛半仓 vs 满仓配对（**结论：0/72 胜**） |

### 5. 入场风险 / 择时

| 模块 | 作用 |
|---|---|
| `entry_risk_threshold_report.py` | 标定入场风险审查阈值 |
| `adx_entry_risk_falsify.py` | ADX 入场风险证伪 |
| `index_adx_entry_risk.py` | 指数 ADX 入场风险 |
| `run_adx_entry_risk_ab.py` | 驱动 ADX 门禁 |
| `run_entry_filter_mvp.py` | 入场过滤 MVP |
| `regime_switch_rule_engine.py` | regime 切换规则引擎 |
| `regime_switch_walkforward.py` | regime 切换 walk-forward |

### 6. 模型分

| 模块 | 作用 |
|---|---|
| `model_score_adapter.py` | `analysis.core.model_scores` 的兼容再导出 |
| `model_score_holding_period_decay.py` | IC 衰减 + 年化净超额 vs 持有期（含行业中性构造） |

### 7. 子包

| 目录 | 作用 |
|---|---|
| `market_spectrum/` | 阴阳谱（全市场 breadth）对指数的预测力检验。8 个脚本构成一条 补数→对账→分析 的流水线，自带 README。**负结果**，原样保留。 |

⚠️ `market_spectrum/data/` 是 **gitignored 的 55MB 产物**，迁移时未带过来；
重跑必须先跑 `fetch_daily_range.py` 补数（`volume_price` 只到 2010）。

## 通过文件传递的隐式依赖

grep 看不出来，改动时要留意：

- `precompute_factor_overlay.py` 的产出由 `analysis/sweep/portfolio_sweep_report.py`
  的 `--cycle-score-source` 消费（不是 import，但改格式会打断 sweep）。
- `path_gate_ab_gate.py` 的 `--sweep-json` 吃 path_gate sweep 的输出，产物落在 `reports/`。
- `industry_ab_gate.py` 对比同一 sweep JSON 中无 `industry_l1` 行与过滤行。
- `regime_weight_ab_gate.py` 的 `--json-full/--json-bear/--json-bull` 分别吃全窗与
  **算法主 bear/bull 段** sweep 输出。
- `run_regime_weight_ab_gate.py --run-sweep` 先跑 diagnosis 得到 `regime_segments`，
  再按 primary 段切窗口。
- `--regime-rule` 支持预设、JSON 文件或 inline JSON；产物文件名带 `rule_id` 后缀。

## 调查结束之后

结论写进 llm-wiki 的 `projects/trading/repos/stock-scoring-system/` 与
`projects/trading/research/`。脚本本身留着 —— 它们是参数化工具，
下一个因子还会用到同一套诊断。
