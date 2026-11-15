"""Shared MongoClient construction with optional timeout / pool tuning."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from pymongo import MongoClient

_INT_ENV_KEYS = (
    ("MONGO_SERVER_SELECTION_TIMEOUT_MS", "serverSelectionTimeoutMS"),
    ("MONGO_CONNECT_TIMEOUT_MS", "connectTimeoutMS"),
    ("MONGO_SOCKET_TIMEOUT_MS", "socketTimeoutMS"),
    ("MONGO_MAX_POOL_SIZE", "maxPoolSize"),
)


def mongo_client_kwargs() -> Dict[str, Any]:
    """Build optional PyMongo kwargs from environment variables."""
    kwargs: Dict[str, Any] = {}
    for env_key, kwarg_key in _INT_ENV_KEYS:
        raw = os.getenv(env_key)
        if raw is None or raw == "":
            continue
        try:
            kwargs[kwarg_key] = int(raw)
        except ValueError:
            continue
    return kwargs


def create_mongo_client(uri: Optional[str] = None) -> MongoClient:
    mongo_url = uri or os.getenv("MONGO_URI", "mongodb://localhost:27017")
    return MongoClient(mongo_url, **mongo_client_kwargs())
