#!/usr/bin/env python3
"""Validate independently persisted entropy scores.

This entrypoint reuses ``ScoreBacktester`` for price joins, Rank IC, quantile
returns, and summaries, but reads from the independent ``entropy_scores``
collection instead of ``stock_scores``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence


from analysis.core.score_backtester import BacktestConfig, ScoreBacktester, ScoreBacktestResult
from shared.data_access import open_research_data_access_for_cli


ENTROPY_SCORE_COLUMNS = (
    "entropy_score",
    "final_entropy_score",
    "entropy_score_z",
    "final_entropy_score_z",
)


class EntropyScoreBacktester(ScoreBacktester):
    """ScoreBacktester variant backed by ``entropy_scores``."""

    def _load_score_records(self) -> Iterable[Dict[str, Any]]:
        query: Dict[str, Any] = {}
        if self.config.start_date or self.config.end_date:
            date_query: Dict[str, Any] = {}
            if self.config.start_date:
                date_query["$gte"] = self.config.start_date
            if self.config.end_date:
                date_query["$lte"] = self.config.end_date
            query["score_date"] = date_query
        if self.config.index_codes:
            query["index_codes"] = {"$in": list(self.config.index_codes)}

        coll = getattr(self.db_manager, "entropy_scores", None)
        if coll is None:
            coll = self.db_manager.db["entropy_scores"]

        projection = {
            "_id": 0,
            "symbol": 1,
            "score_date": 1,
            "name": 1,
            "industry": 1,
            "index_codes": 1,
            "entropy_score": 1,
            "final_entropy_score": 1,
            "entropy_score_z": 1,
            "final_entropy_score_z": 1,
        }
        try:
            return list(coll.find(query, projection).sort("score_date", 1))
        except TypeError:
            return list(coll.find(query))
        except AttributeError:
            return list(coll.find(query))

    def _score_row_from_doc(self, doc: Dict[str, Any]) -> Dict[str, Optional[float]]:
        return {col: self._to_float(doc.get(col)) for col in ENTROPY_SCORE_COLUMNS}


def run_entropy_backtest(
    db_manager,
    horizons: Sequence[int] = (20, 40, 60),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    index_codes: Sequence[str] = (),
    universe_index_codes: Sequence[str] = (),
    min_names_per_date: int = 30,
) -> ScoreBacktestResult:
    config = BacktestConfig(
        horizons=tuple(horizons),
        score_columns=ENTROPY_SCORE_COLUMNS,
        start_date=start_date,
        end_date=end_date,
        index_codes=tuple(index_codes),
        universe_index_codes=tuple(universe_index_codes),
        min_names_per_date=min_names_per_date,
    )
    return EntropyScoreBacktester(db_manager, config).run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate entropy_scores predictive power")
    parser.add_argument("--start-date", help="Inclusive score_date lower bound, YYYYMMDD")
    parser.add_argument("--end-date", help="Inclusive score_date upper bound, YYYYMMDD")
    parser.add_argument("--horizons", nargs="+", type=int, default=[20, 40, 60])
    parser.add_argument("--index-codes", nargs="+", default=[], help="Optional entropy_scores.index_codes filter")
    parser.add_argument("--universe-index", nargs="+", default=[], help="Optional universe index filter")
    parser.add_argument("--min-names", type=int, default=30)
    parser.add_argument("--output", default="docs/entropy_validation.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db_manager = open_research_data_access_for_cli(enable_tushare=False)
    result = run_entropy_backtest(
        db_manager,
        horizons=args.horizons,
        start_date=args.start_date,
        end_date=args.end_date,
        index_codes=args.index_codes,
        universe_index_codes=args.universe_index,
        min_names_per_date=args.min_names,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote entropy validation report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
