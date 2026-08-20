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

_LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_ROOT_NAME = "rag"  # parent logger; all pipeline loggers are children


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

    # On Windows, a non-interactive stdout (piped, redirected, or captured
    # by a CI runner) defaults to the console codepage (cp1252) instead of
    # UTF-8. This project's log messages use em-dashes and arrows, which
    # then crash the handler with UnicodeEncodeError — not a caught
    # exception, since the logging module swallows handler errors and just
    # prints "--- Logging error ---", so this was silently corrupting
    # output rather than failing loudly. A real terminal already
    # negotiates UTF-8 correctly, so this only matters for the
    # non-interactive case. Same fix as app.py's __main__ block, but
    # centralized here so every entry point that logs gets it, not just
    # the CLI.
    if (
        hasattr(sys.stdout, "reconfigure")
        and sys.stdout.encoding is not None
        and sys.stdout.encoding.lower() != "utf-8"
    ):
        sys.stdout.reconfigure(encoding="utf-8")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False  # don't bubble up to the Python root logger


def get_logger(name: str) -> logging.Logger:
    """
    Returns a child logger under the 'rag' namespace.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A configured logging.Logger instance.
    """
    configure_logging()  # idempotent — safe to call on every import
    return logging.getLogger(name)
