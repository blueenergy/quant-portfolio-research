import pandas as pd

from analysis.research.model_score_adapter import adapter_enabled, merge_model_score_column


def test_adapter_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MODEL_SCORE_ADAPTER_ENABLED", raising=False)
    assert not adapter_enabled()
    panel = pd.DataFrame({"trade_date": ["2024-01-02"], "ts_code": ["000001.SZ"]})
    out = merge_model_score_column(panel, "shadow-csi1000")
    assert "model_score" not in out.columns or out.equals(panel)
