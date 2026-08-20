"""
rag/loaders.py
---------------
Per-format document text extraction, dispatched by file extension.

Adding a new format means adding one function here and one entry in
_LOADERS — rag/ingestion.py's directory-walking and chunking logic never
change; it just asks for supported extensions and calls load_document_text().
"""

from pathlib import Path

from rag.logging_config import get_logger

log = get_logger(__name__)


def _load_txt(filepath: Path) -> str:
    return filepath.read_text(encoding="utf-8")


def _load_pdf(filepath: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("PDF ingestion requires pypdf.\n  → pip install pypdf") from exc

    reader = PdfReader(str(filepath))
    page_texts = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(page_texts)

    if not text.strip():
        log.warning(
            "%s: no extractable text (likely a scanned/image-only PDF — "
            "OCR is not implemented). Contributing zero chunks.",
            filepath.name,
        )
    return text


_LOADERS = {
    ".txt": _load_txt,
    ".pdf": _load_pdf,
}
SUPPORTED_EXTENSIONS = tuple(sorted(_LOADERS))


def load_document_text(filepath: Path) -> str:
    """
    Extracts plaintext from a document, dispatched by its file extension.

    Args:
        filepath: Path to a document whose extension is in SUPPORTED_EXTENSIONS.

    Returns:
        Extracted text. May be an empty string (e.g. a scanned PDF with no
        text layer) — callers should treat that as "zero chunks", not an error.

    Raises:
        ValueError:  If the file extension isn't supported.
        ImportError: If the extension needs an optional dependency that
                     isn't installed (currently: .pdf needs pypdf).
    """
    ext = filepath.suffix.lower()
    if ext not in _LOADERS:
        raise ValueError(
            f"Unsupported document type '{ext}' ({filepath.name}). "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return _LOADERS[ext](filepath)
