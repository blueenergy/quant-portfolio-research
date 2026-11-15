"""Backward-compatible re-exports; prefer ``analysis.core.html_format``."""

from analysis.core.html_format import as_float, fmt_num, fmt_pct, html_document

__all__ = ["as_float", "fmt_num", "fmt_pct", "html_document"]
