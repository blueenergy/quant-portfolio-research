"""Tests for run_industry_ab_gate merge helper."""

import json
from pathlib import Path

from analysis.research.run_industry_ab_gate import _merge_sweep_payloads


def test_merge_sweep_payloads_combines_rows(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    industry = tmp_path / "industry.json"
    baseline.write_text(
        json.dumps({"rows": [{"top_n": 8, "index_excess_cumulative_return": 0.1}]}),
        encoding="utf-8",
    )
    industry.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "top_n": 8,
                        "industry_l1": "801080.SI",
                        "index_excess_cumulative_return": 0.09,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    merged = _merge_sweep_payloads(baseline, industry, industry_l1="801080.SI")
    assert len(merged["rows"]) == 2
    assert merged["industry_l1"] == "801080.SI"
