# quant-portfolio-research

Portfolio research worker, extracted from `stock-scoring-system`. It consumes
`portfolio_research_jobs` from `quant_trading`, runs production-data parameter
sweeps, persists result summaries, and materializes a candidate portfolio
strategy config that quantFinance can publish.

## Scope

- `analysis/sweep/`: parameter axes, combo backtests, entry filters, reports.
- `analysis/core/`: portfolio/score backtesters, benchmark prices, regime,
  path features, model-score merge, fee/trailing-stop helpers.
- `analysis/reporting/`, `analysis/factor/ic_metrics.py`.
- `tools/run_portfolio_research_loop.py` + `tools/process_portfolio_research_jobs.py`.
- `research_db.py`: research job/result/worker-status collections.

## Data boundary

Reads market data through `data-access-lib` and the legacy research tables
(`stock_scores`, `money_flow`) directly. It does **not** import `scorers/`,
`data_manager`, Tushare, or the portfolio planning package. `stock_scores` is
a data dependency only and will be replaced by `model_predictions` before the
collection is retired.

## Run

```
docker build -t quant-portfolio-research:dev .
docker run --rm --env-file ../quant-infrastructure/apps/env/common.env \
  --env-file ../quant-infrastructure/apps/env/quant-scorer.env \
  quant-portfolio-research:dev
```

## Tests

```
pip install -e '.[dev]'
pip install -e ./data-access-lib
pytest -q
```
