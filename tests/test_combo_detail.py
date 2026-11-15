from analysis.sweep.combo_detail import annotate_combo_regime, compact_trade_date


def test_compact_trade_date_strips_separators():
    assert compact_trade_date("2024-01-02") == "20240102"
    assert compact_trade_date("20240102") == "20240102"
    assert compact_trade_date("") == ""


def test_annotate_combo_regime_copies_label_onto_periods_and_trades():
    detail = {
        "periods": [
            {"score_date": "2024-01-02", "portfolio_return_net": 0.01},
            {"score_date": "20240116", "portfolio_return_net": 0.0},
            {"score_date": "2024-01-30", "portfolio_return_net": 0.02},
        ],
        "trades": [
            {"score_date": "2024-01-02", "symbol": "600000.SH"},
            {"score_date": "20240116", "symbol": "000001.SZ"},
        ],
    }

    annotate_combo_regime(
        detail,
        [
            {"score_date": "20240102", "label": "bull", "action": "invest"},
            {"score_date": "2024-01-16", "label": "bear", "action": "cash"},
        ],
    )

    assert detail["periods"][0]["regime_label"] == "bull"
    assert detail["periods"][0]["regime_action"] == "invest"
    assert detail["periods"][1]["regime_label"] == "bear"
    assert detail["periods"][1]["regime_action"] == "cash"
    assert "regime_label" not in detail["periods"][2]
    assert detail["trades"][0]["regime_label"] == "bull"
    assert detail["trades"][1]["regime_label"] == "bear"


def test_annotate_combo_regime_is_noop_without_schedule():
    detail = {"periods": [{"score_date": "20240102"}], "trades": []}
    annotate_combo_regime(detail, None)
    assert "regime_label" not in detail["periods"][0]
