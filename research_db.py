"""Mongo accessors for the portfolio research worker.

Replaces ``portfolio.db.get_trading_collections`` so this repo never imports
the portfolio planning package (which eagerly loads 30+ plan/lineage/paper
modules). Only the three collections this worker owns live here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from pymongo import ASCENDING, DESCENDING

from utils.mongo_client import create_mongo_client


@dataclass(frozen=True)
class ResearchCollections:
    research_jobs: Any
    research_results: Any
    worker_status: Any


def get_trading_db(client=None):
    client = client or create_mongo_client()
    return client[os.getenv("TRADING_DB_NAME", "quant_trading")]


def get_research_collections(db=None) -> ResearchCollections:
    db = db if db is not None else get_trading_db()
    return ResearchCollections(
        research_jobs=db["portfolio_research_jobs"],
        research_results=db["portfolio_research_results"],
        worker_status=db["portfolio_worker_status"],
    )


def ensure_research_indexes(collections: Optional[ResearchCollections] = None) -> None:
    cols = collections or get_research_collections()
    cols.worker_status.create_index([("worker_type", ASCENDING), ("worker_id", ASCENDING)], unique=True)
    cols.worker_status.create_index([("last_seen_at", DESCENDING)])
    cols.research_jobs.create_index([("job_id", ASCENDING)], unique=True)
    cols.research_jobs.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
    cols.research_jobs.create_index([("status", ASCENDING), ("created_at", ASCENDING)])
    cols.research_results.create_index([("result_id", ASCENDING)], unique=True)
    cols.research_results.create_index([("job_id", ASCENDING)])
    cols.research_results.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
