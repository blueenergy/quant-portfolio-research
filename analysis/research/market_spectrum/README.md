# 阴阳谱 (market breadth) 预测力研究

问的是：**全市场「收盘站上自己 N 日均线」的占比，对指数未来涨跌有没有预测力。**

结论：**没有可用的边际**（负结果）。详见
`llm-wiki/projects/trading/research/market-spectrum-predictive-power.md`。

这套脚本不是一次性草稿——`volume_price` 只到 2010-01-04，重跑需要重新
补数，所以把口径、验证和分段检验都固化成可复现的 CLI。

## 数据口径

阴阳谱 = `close > MA(N)` 的股票数 / 当日可交易股票数。

- **分母**：`close` 与 `volume` 均非空。与生产定义一致。
- **股票池**：A 股，排除 B 股（`900xxx.SH` / `200xxx.SZ`）。
- **键**：`(trade_date, ma_period)`，与 `quant_data.market_spectrum` 同构。

三个来源，**不要混用而不知情**：

| 来源 | 覆盖 | 说明 |
|---|---|---|
| `mongo` | 2010-01-04 起 | `quant_data.volume_price` 重算，生产对齐 |
| `tushare` | 2005-01-04 起 | `fetch_daily_range.py` 拉到本地，含 2010-2011 重叠段 |
| `merged` | 2005-01-04 起 | 2010 起用 mongo，之前用 tushare。**先跑 reconcile.py** |

⚠️ `quant_data.market_spectrum`（生产集合）只有 **2024-01-02 起**，
不是长历史。早期笔记里「MA5 从 2005 / MA30 补到 2021」指的是
`temp/yyp_csi2000` 的一次性产物，未留在生产库。

## 流程

```bash
PY=/home/shuyolin/trading/quant-data-engine/.venv/bin/python   # 有 pymongo/pandas

# 1. 补 2005-2011（含 2010-2011 重叠，用于对账；约 10 分钟）
$PY fetch_daily_range.py --start 20041101 --end 20111231

# 2. 建两个口径的 breadth
$PY build_breadth.py --source mongo
$PY build_breadth.py --source tushare

# 3. 对账 —— 不过就不要 merge
$PY reconcile.py

# 4. 合并
$PY build_breadth.py --source merged

# 5. 分析
$PY analyze.py --start 20050101            # 分布 / 同期与前瞻相关
$PY event_study.py --threshold 0.20        # 事件研究
$PY deep.py --start 20050101               # 阈值扫描 + 聚集性 + 反向
$PY robust.py --start 20050101             # 留一 / 分期 / 执行时点
$PY decile.py --start 20050101             # 十分位 + 低位回归
$PY incremental.py --start 20050101        # 否决性检验（控制过去20日跌幅）
$PY extended_check.py                      # 分期对照 + 2008 专项
```

数据缓存在 `data/`（gitignored）。删掉 `data/` 后从第 1 步重来。

## 方法要点

- **事件去重**：连续低于阈值合并为一个 episode，取首日。否则一次股灾会被
  算成几十个独立样本。
- **显著性**：循环位移 permutation（滚动未来收益序列）。重叠窗口使朴素
  t 检验严重高估显著性。
- **独立性**：episode 间隔 >60 交易日视为独立市场事件。名义样本量与有效
  样本量差一个数量级——读 CI 宽度，不要读 n。
- **多重比较**：7 阈值 × 3 指数 = 21 个组合。扫出单个 p≈0.05 是噪声的
  正常产出。
- **指数回溯**：中证1000 于 2014-10-17 才发布、中证2000 于 2023-10-11，
  更早的数据是中证公司基日回溯，不是实盘指数。见 `common.INDEX_PUBLISHED`。

## 契约

`analysis/research/README.md` 的约定适用于本目录：**外部不得 import 这里的
东西**，研究脚本可以随时改写或删除。
