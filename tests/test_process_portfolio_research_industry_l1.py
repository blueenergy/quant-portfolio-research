"""Regression tests for industry_l1 handling in research worker."""

from unittest.mock import MagicMock

import pandas as pd

from shared.industry_universe import resolve_industry_l1_codes, sw_l1_catalog


def test_resolve_industry_l1_after_db_manager_available():
    """Industry codes must be resolved with an open db_manager (not before)."""
    classify_coll = MagicMock()
    classify_coll.find.return_value = [
        {"index_code": "801080.SI", "industry_name": "电子"},
    ]
    db_manager = type("Db", (), {"sw_index_classify": classify_coll})()

    catalog = sw_l1_catalog(db_manager)
    codes = resolve_industry_l1_codes(["电子"], catalog)
    assert codes == ["801080.SI"]


def test_run_research_job_resolves_industry_l1_after_db_open(monkeypatch):
    """sw_l1_catalog must run after open_research_data_access (NameError regression)."""
    from tools.process_portfolio_research_jobs import run_research_job

    call_order: list[str] = []
    mock_db = type("Db", (), {})()

    def fake_open():
        call_order.append("open")
        return mock_db

    def fake_catalog(db_manager):
        call_order.append("catalog")
        assert db_manager is mock_db
        return [{"code": "801080.SI", "name": "电子"}]

    monkeypatch.setattr("shared.data_access.open_research_data_access", fake_open)
    monkeypatch.setattr("shared.industry_universe.sw_l1_catalog", fake_catalog)
    monkeypatch.setattr(
        "tools.process_portfolio_research_jobs._latest_stock_score_date",
        lambda *args, **kwargs: "20240101",
    )
    monkeypatch.setattr(
        "analysis.core.benchmark_prices.load_index_benchmark_prices",
        lambda *args, **kwargs: pd.Series(dtype=float),
    )
    monkeypatch.setattr("analysis.sweep.sweep_engine.load_panel_cache", lambda *args, **kwargs: None)

    class _StubBacktester:
        def __init__(self, *args, **kwargs):
            self.price_cache = {}

        def prepare_score_panel(self):
            return []

        def build_dataset_from_panel(self, *args, **kwargs):
            return pd.DataFrame()

    monkeypatch.setattr("analysis.sweep.portfolio_sweep_report.ScoreBacktester", _StubBacktester)

    job = {
        "job_id": "test-industry-l1",
        "params": {
            "start_date": "20240101",
            "end_date": "20240102",
            "universe_index": "csi1000",
            "score_column": "composite_growth_cycle_score",
            "growth_cycle_weights": ["30:70"],
            "top_n_values": [5],
            "horizon": 20,
            "industry_l1": "电子",
            "use_panel_cache": False,
            "resume_checkpoint": False,
        },
    }

    try:
        run_research_job(job)
    except Exception:
        pass

    assert call_order == ["open", "catalog"]
