from types import SimpleNamespace

from analysis.core.score_backtester import BacktestConfig
from analysis.research.entropy_validation import ENTROPY_SCORE_COLUMNS, EntropyScoreBacktester


class _Cursor:
    def __init__(self, docs):
        self.docs = list(docs)

    def sort(self, key, direction):
        self.docs = sorted(self.docs, key=lambda row: row.get(key, ""), reverse=direction < 0)
        return self

    def __iter__(self):
        return iter(self.docs)


class _Collection:
    def __init__(self, docs):
        self.docs = list(docs)
        self.last_query = None
        self.last_projection = None

    def find(self, query=None, projection=None):
        self.last_query = query or {}
        self.last_projection = projection

        def matches(doc):
            for key, value in self.last_query.items():
                current = doc.get(key)
                if isinstance(value, dict):
                    if "$gte" in value and current < value["$gte"]:
                        return False
                    if "$lte" in value and current > value["$lte"]:
                        return False
                    if "$in" in value and not set(current or []).intersection(value["$in"]):
                        return False
                    continue
                if current != value:
                    return False
            return True

        return _Cursor([doc for doc in self.docs if matches(doc)])


def test_entropy_backtester_extracts_entropy_score_columns():
    tester = EntropyScoreBacktester(SimpleNamespace(entropy_scores=_Collection([])), BacktestConfig(score_columns=ENTROPY_SCORE_COLUMNS))

    row = tester._score_row_from_doc(
        {
            "entropy_score": 66.0,
            "final_entropy_score": 61.0,
            "entropy_score_z": 1.6,
            "final_entropy_score_z": 1.1,
            "growth_score": 99.0,
        }
    )

    assert row == {
        "entropy_score": 66.0,
        "final_entropy_score": 61.0,
        "entropy_score_z": 1.6,
        "final_entropy_score_z": 1.1,
    }


def test_entropy_backtester_loads_from_entropy_scores_collection():
    coll = _Collection(
        [
            {"symbol": "000001", "score_date": "20250101", "entropy_score": 55.0, "index_codes": ["csi1000"]},
            {"symbol": "000002", "score_date": "20250201", "entropy_score": 65.0, "index_codes": ["hs300"]},
        ]
    )
    config = BacktestConfig(
        score_columns=ENTROPY_SCORE_COLUMNS,
        start_date="20250115",
        end_date="20251231",
        index_codes=("hs300",),
    )
    tester = EntropyScoreBacktester(SimpleNamespace(entropy_scores=coll), config)

    rows = list(tester._load_score_records())

    assert len(rows) == 1
    assert rows[0]["symbol"] == "000002"
    assert coll.last_projection["entropy_score"] == 1
