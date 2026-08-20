"""
Tests for rag/logging_config.py.

The UTF-8 reconfigure test is a regression guard for a real bug: running
eval/run_eval.py for real (Ollama embeddings, actual corpus) on Windows
produced a silent "--- Logging error ---" + UnicodeEncodeError on every
log line containing an em-dash or arrow, because logging swallows handler
exceptions instead of raising them — the script still exited 0, so this
was easy to miss. app.py's __main__ block had its own fix for the same
root cause already; this is the same fix applied where it actually
belongs, so every entry point that logs gets it, not just the CLI.
"""

import importlib
import io
import logging

import rag.logging_config as logging_config_module


def _reset_root_logger():
    root = logging.getLogger(logging_config_module._ROOT_NAME)
    for handler in list(root.handlers):
        root.removeHandler(handler)


def test_get_logger_returns_child_of_rag_namespace():
    logger = logging_config_module.get_logger("rag.some_module")
    assert logger.name == "rag.some_module"


def test_configure_logging_is_idempotent():
    _reset_root_logger()
    try:
        logging_config_module.configure_logging()
        first_handler_count = len(logging.getLogger("rag").handlers)
        logging_config_module.configure_logging()
        second_handler_count = len(logging.getLogger("rag").handlers)
        assert first_handler_count == second_handler_count == 1
    finally:
        _reset_root_logger()
        logging_config_module.configure_logging()  # restore for subsequent tests


def test_root_logger_does_not_propagate_to_python_root():
    _reset_root_logger()
    try:
        logging_config_module.configure_logging()
        assert logging.getLogger("rag").propagate is False
    finally:
        _reset_root_logger()
        logging_config_module.configure_logging()


def test_configure_logging_reconfigures_non_utf8_stdout(monkeypatch):
    # A stand-in for Windows' non-interactive stdout: a real TextIOWrapper
    # (so .reconfigure() exists) explicitly opened with a non-UTF-8 codec.
    non_utf8_stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr("sys.stdout", non_utf8_stdout)

    _reset_root_logger()
    try:
        importlib.reload(logging_config_module)
        logging_config_module.configure_logging()  # reload() alone doesn't call it — get_logger() does, lazily
        assert non_utf8_stdout.encoding.lower() == "utf-8"
    finally:
        _reset_root_logger()
        importlib.reload(logging_config_module)  # restore real sys.stdout binding


def test_configure_logging_leaves_already_utf8_stdout_alone(monkeypatch):
    utf8_stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    monkeypatch.setattr("sys.stdout", utf8_stdout)

    _reset_root_logger()
    try:
        importlib.reload(logging_config_module)
        logging_config_module.configure_logging()
        assert utf8_stdout.closed is False  # reconfigure() would have replaced the buffer otherwise
        assert utf8_stdout.encoding.lower() == "utf-8"
    finally:
        _reset_root_logger()
        importlib.reload(logging_config_module)


def test_log_message_with_em_dash_and_arrow_does_not_raise(monkeypatch):
    # The actual failure mode: log a message containing the exact characters
    # (em-dash, arrow) this project's real log lines use, through a non-UTF-8
    # stdout, and confirm configure_logging()'s fix prevents the crash.
    non_utf8_stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr("sys.stdout", non_utf8_stdout)

    _reset_root_logger()
    try:
        importlib.reload(logging_config_module)
        logger = logging_config_module.get_logger("rag.test")
        logger.info("Ingestion complete — %d vectors indexed", 86)
        non_utf8_stdout.flush()
        non_utf8_stdout.buffer.seek(0)
        output = non_utf8_stdout.buffer.read().decode("utf-8")
        assert "Ingestion complete — 86 vectors indexed" in output
    finally:
        _reset_root_logger()
        importlib.reload(logging_config_module)
