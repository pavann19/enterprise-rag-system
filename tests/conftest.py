"""
Shared test fixtures/helpers.

make_minimal_pdf() builds a valid single-page PDF by hand, byte-by-byte,
rather than via a PDF-writing library — pypdf itself only reads/manipulates
PDFs, it doesn't lay out text, and pulling in a separate writer library
(reportlab, fpdf2) just for test fixtures wasn't worth the added
dependency. The format is simple enough to construct directly for a
single-page, single-string document.
"""

import io


def make_minimal_pdf(text: str) -> bytes:
    """A one-page PDF containing a single text string, extractable by pypdf."""
    content = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 5 0 R >> >> "
        b"/MediaBox [0 0 612 792] /Contents 4 0 R >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    return _assemble_pdf(objects)


def make_empty_pdf() -> bytes:
    """A valid one-page PDF with no /Contents — simulates a scanned/image-only PDF."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
    ]
    return _assemble_pdf(objects)


def _assemble_pdf(objects: list) -> bytes:
    buf = io.BytesIO()
    buf.write(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(buf.tell())
        buf.write(str(i).encode() + b" 0 obj\n" + obj + b"\nendobj\n")
    xref_offset = buf.tell()
    buf.write(b"xref\n0 " + str(len(objects) + 1).encode() + b"\n")
    buf.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        buf.write(f"{off:010d} 00000 n \n".encode())
    buf.write(b"trailer\n<< /Size " + str(len(objects) + 1).encode() + b" /Root 1 0 R >>\n")
    buf.write(b"startxref\n" + str(xref_offset).encode() + b"\n%%EOF")
    return buf.getvalue()
