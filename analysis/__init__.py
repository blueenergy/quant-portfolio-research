"""Analysis utilities for validating stock scoring signals.

Organised by functional domain, roughly in dependency order:

``core``
    Backtest engines and shared primitives. Everything builds on this.
``sweep``
    Portfolio parameter-sweep subsystem, driven by the
    quant-portfolio-researcher worker.
``factor``
    IC/ranking helpers used by portfolio research.
``reporting``
    HTML/JSON rendering shared across the layers above.
The four packages here are reachable from the production worker; treat
their public names as a contract.
"""
