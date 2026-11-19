from analysis.research.index_benchmark_diagnostics import diagnose_index_benchmark, render_markdown


def test_diagnose_index_benchmark_reports_gaps_and_returns():
    payload = {
        "periods": [
            {"score_date": "20250101", "top_n": 10, "horizon": 2, "index_benchmark_return": 0.21},
            {"score_date": "20250104", "top_n": 10, "horizon": 2, "index_benchmark_return": 0.0909},
        ]
    }
    prices = [
        {"trade_date": "20250101", "close": 100},
        {"trade_date": "20250102", "close": 110},
        {"trade_date": "20250103", "close": 121},
        {"trade_date": "20250104", "close": 90},
        {"trade_date": "20250105", "close": 95},
        {"trade_date": "20250106", "close": 100},
    ]

    result = diagnose_index_benchmark(payload, prices)

    assert result["top_n"] == 10
    assert result["period_count"] == 2
    assert round(result["buy_hold_return"], 4) == 0.0
    assert round(result["periodic_compound_return"], 4) == round((1.21 * (100 / 90)) - 1, 4)
    assert result["total_gap_days"] == 1
    assert result["total_overlap_days"] == 0
    assert round(result["windows"][1]["bridge_return_from_previous_end"], 4) == round(90 / 121 - 1, 4)


def test_render_markdown_includes_diagnostic_sections():
    payload = {
        "periods": [
            {"score_date": "20250101", "top_n": 10, "horizon": 1, "index_benchmark_return": 0.1},
        ]
    }
    prices = [
        {"trade_date": "20250101", "close": 100},
        {"trade_date": "20250102", "close": 110},
    ]
    markdown = render_markdown(diagnose_index_benchmark(payload, prices))

    assert "Buy-hold return" in markdown
    assert "Periodic compound return" in markdown
    assert "Largest Positive Windows" in markdown
