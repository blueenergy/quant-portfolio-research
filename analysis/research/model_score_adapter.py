"""Backward-compatible research-side import for model score helpers.

The implementation lives in ``analysis.core.model_scores`` so production
``tools/`` code can use it without importing the draft research layer. This
module stays for existing research scripts and tests; new code should import
from ``analysis.core.model_scores``.
"""

from analysis.core.model_scores import (
    adapter_enabled,
    load_model_scores,
    load_model_scores_range,
    merge_model_score_column,
    merge_model_score_into_dataset,
)

__all__ = [
    "adapter_enabled",
    "load_model_scores",
    "load_model_scores_range",
    "merge_model_score_column",
    "merge_model_score_into_dataset",
]
