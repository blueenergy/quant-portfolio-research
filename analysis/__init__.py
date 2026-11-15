"""Analysis utilities for validating stock scoring signals.

Organised by functional domain, roughly in dependency order:

``core``
    Backtest engines and shared primitives. Everything builds on this.
``sweep``
    Portfolio parameter-sweep subsystem, driven by the quant-researcher
    worker.
``factor``
    Alpha factor construction and backtesting, driven by the
    quant-factor-researcher worker.
``reporting``
    HTML/JSON rendering shared across the layers above.
``research``
    Standalone one-off investigations. Nothing in production imports
    this package.

The first four are reachable from production workers; treat their public
names as a contract. ``research`` is free to change.
"""
