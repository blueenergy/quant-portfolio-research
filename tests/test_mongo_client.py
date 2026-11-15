import os

from utils.mongo_client import create_mongo_client, mongo_client_kwargs


def test_mongo_client_kwargs_reads_int_env(monkeypatch):
    monkeypatch.setenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "5000")
    monkeypatch.setenv("MONGO_CONNECT_TIMEOUT_MS", "3000")
    monkeypatch.setenv("MONGO_SOCKET_TIMEOUT_MS", "120000")
    monkeypatch.setenv("MONGO_MAX_POOL_SIZE", "50")

    kwargs = mongo_client_kwargs()

    assert kwargs == {
        "serverSelectionTimeoutMS": 5000,
        "connectTimeoutMS": 3000,
        "socketTimeoutMS": 120000,
        "maxPoolSize": 50,
    }


def test_create_mongo_client_uses_env_kwargs(monkeypatch):
    captured = {}

    class _FakeMongoClient:
        def __init__(self, uri, **kwargs):
            captured["uri"] = uri
            captured["kwargs"] = kwargs

    monkeypatch.setenv("MONGO_URI", "mongodb://example:27017")
    monkeypatch.setenv("MONGO_SOCKET_TIMEOUT_MS", "45000")
    monkeypatch.setattr("utils.mongo_client.MongoClient", _FakeMongoClient)

    create_mongo_client()

    assert captured["uri"] == "mongodb://example:27017"
    assert captured["kwargs"] == {"socketTimeoutMS": 45000}
