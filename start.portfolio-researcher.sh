#!/bin/bash
set -e

echo "[quant-portfolio-research] Starting portfolio research scheduler..."
echo "[quant-portfolio-research] Portfolio research loop interval: ${PORTFOLIO_RESEARCH_LOOP_INTERVAL_SECONDS:-5}s"
echo "[quant-portfolio-research] Portfolio research runtime: ${PORTFOLIO_RESEARCH_RUNTIME:-local_docker}"

WORKER_ID="${PORTFOLIO_RESEARCH_WORKER_ID:-${HOSTNAME:-quant-portfolio-researcher}}"
echo "[quant-portfolio-research] Portfolio research worker id: ${WORKER_ID}"

exec python /app/tools/run_portfolio_research_loop.py \
  --worker-id "${WORKER_ID}" \
  --interval-seconds "${PORTFOLIO_RESEARCH_LOOP_INTERVAL_SECONDS:-5}" \
  --limit-per-iteration "${PORTFOLIO_RESEARCH_LOOP_LIMIT:-2}"
