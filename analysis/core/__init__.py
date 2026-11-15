"""Backtest engines and shared primitives.

Everything else in ``analysis`` builds on this layer, and the research
workers reach it indirectly through ``analysis.sweep`` and
``analysis.factor``. Treat the public names here as a contract: changing
them ripples into production.
"""
