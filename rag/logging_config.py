"""
rag/logging_config.py
---------------------
Centralized logging configuration for the Enterprise RAG pipeline.

All modules import `get_logger(__name__)` to obtain a named logger.
Logging format is structured and human-readable — no external libraries.

Log levels:
    DEBUG   — internal state (scores, chunk counts, model params)
    INFO    — normal pipeline events (startup, query received, success)
    WARNING — recoverable issues
    ERROR   — failures that propagate to the caller

To change the global log level at runtime:
    import logging
    logging.getLogger("rag").setLevel(logging.DEBUG)
"""

import logging
import sys


# ── Configuration ──────────────────────────────────────────────────────────────

_LOG_FORMAT  = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_ROOT_NAME   = "rag"   # parent logger; all pipeline loggers are children


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configures the root 'rag' logger with a stdout StreamHandler.

    Safe to call multiple times — handlers are not duplicated.

    Args:
        level: Logging level for the rag namespace (default: INFO).
    """
    root = logging.getLogger(_ROOT_NAME)

    # Avoid adding duplicate handlers on repeated imports
    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False   # don't bubble up to the Python root logger


def get_logger(name: str) -> logging.Logger:
    """
    Returns a child logger under the 'rag' namespace.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A configured logging.Logger instance.
    """
    configure_logging()   # idempotent — safe to call on every import
    return logging.getLogger(name)
