from tools.process_portfolio_research_jobs import _universe_pit_quality


class _Coll:
    def __init__(self, docs):
        self.docs = docs

    def find_one(self, query=None, projection=None, sort=None):
        query = query or {}
        rows = [d for d in self.docs if _matches(d, query)]
        if sort:
            for key, direction in reversed(sort):
                rows.sort(key=lambda row: row.get(key, ""), reverse=direction < 0)
        return rows[0] if rows else None

    def distinct(self, field, query=None):
        query = query or {}
        return sorted({d.get(field) for d in self.docs if _matches(d, query) and d.get(field)})


def _matches(doc, query):
    for key, expected in query.items():
        value = doc.get(key)
        if isinstance(expected, dict):
            if "$lte" in expected and not (value is not None and str(value) <= str(expected["$lte"])):
                return False
            continue
        if value != expected:
            return False
    return True


class _Db:
    def __init__(self, docs):
        self.index_constituents = _Coll(docs)


def test_universe_pit_quality_point_in_time_when_snapshot_exists_before_start():
    db = _Db(
        [
            {
                "index_code": "csi1000",
                "weight_trade_date": "20230131",
                "update_date": "20230201",
                "symbol": "000001.SZ",
            }
        ]
    )
    assert _universe_pit_quality(db, "csi1000", "20230315") == "point_in_time"


def test_universe_pit_quality_latest_only_when_no_historical_snapshot():
    db = _Db(
        [
            {
                "index_code": "csi1000",
                "weight_trade_date": "20260501",
                "update_date": "20260501",
                "symbol": "000001.SZ",
            }
        ]
    )
    assert _universe_pit_quality(db, "csi1000", "20230315") == "latest_only"
