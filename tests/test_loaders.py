"""Tests for rag/loaders.py. PDF fixtures come from tests/conftest.py."""

import pytest

from rag.loaders import SUPPORTED_EXTENSIONS, load_document_text
from tests.conftest import make_empty_pdf, make_minimal_pdf


def test_supported_extensions_includes_txt_and_pdf():
    assert ".txt" in SUPPORTED_EXTENSIONS
    assert ".pdf" in SUPPORTED_EXTENSIONS


def test_load_txt(tmp_path):
    filepath = tmp_path / "doc.txt"
    filepath.write_text("plain text content", encoding="utf-8")
    assert load_document_text(filepath) == "plain text content"


def test_load_pdf_extracts_text(tmp_path):
    filepath = tmp_path / "doc.pdf"
    filepath.write_bytes(make_minimal_pdf("Hello World from a test PDF"))
    text = load_document_text(filepath)
    assert "Hello World from a test PDF" in text


def test_load_pdf_case_insensitive_extension(tmp_path):
    filepath = tmp_path / "doc.PDF"
    filepath.write_bytes(make_minimal_pdf("uppercase extension"))
    text = load_document_text(filepath)
    assert "uppercase extension" in text


def test_load_pdf_with_no_text_layer_returns_empty_string(tmp_path):
    filepath = tmp_path / "scanned.pdf"
    filepath.write_bytes(make_empty_pdf())
    text = load_document_text(filepath)
    assert text.strip() == ""


def test_load_document_text_rejects_unsupported_extension(tmp_path):
    filepath = tmp_path / "doc.docx"
    filepath.write_text("some content", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported document type"):
        load_document_text(filepath)


def test_load_pdf_raises_clear_error_when_pypdf_missing(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "pypdf", None)
    filepath = tmp_path / "doc.pdf"
    filepath.write_bytes(make_minimal_pdf("won't get this far"))
    with pytest.raises(ImportError, match="pypdf"):
        load_document_text(filepath)
