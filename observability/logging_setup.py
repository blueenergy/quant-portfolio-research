"""Logging configuration for long-running workers.

Workers previously wrote everything with ``print``, so a crash and a routine
status line were indistinguishable in the container log: same stream, no
level, no timestamp. That is how two cron jobs stayed broken for three weeks.

Call :func:`configure_worker_logging` once at the top of a worker entry point.
Everything goes to stdout, which is what container runtimes collect, and the
level field is what an alert can key on.
"""

from __future__ import annotations

import logging
import os
import sys
import time

DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

_configured = False


class UtcFormatter(logging.Formatter):
    """Stamp every line in UTC.

    The host, the container and the person reading the log are routinely in
    three different zones, so a local timestamp makes correlating a worker
    failure with anything else guesswork.
    """

    converter = time.gmtime


def configure_worker_logging(*, default_level: str = "INFO", force: bool = False) -> logging.Logger:
    """Send levelled, timestamped logs to stdout. Safe to call more than once.

    ``LOG_LEVEL`` overrides the level so a noisy incident can be debugged
    without a redeploy.
    """
    global _configured
    root = logging.getLogger()

    if _configured and not force:
        return root

    level_name = (os.getenv("LOG_LEVEL") or default_level).upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        level = logging.INFO

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(UtcFormatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)

    # pymongo's per-command DEBUG stream drowns the worker's own output
    logging.getLogger("pymongo").setLevel(max(level, logging.WARNING))

    _configured = True
    return root
