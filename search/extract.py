"""Phase 1 · Load a corpus folder and extract text from PDF, DOCX, TXT and HTML.

Contract
--------
extract_folder(folder) -> (pages, skipped)
    pages:   list of dicts {"doc_id", "title", "page", "text"} — one per PDF page,
             one per non-PDF document (page = 1).
    skipped: list of (path, reason) — e.g. scanned PDFs with almost no text (OCR is out of scope).

Libraries: PyMuPDF (fitz) for PDF, python-docx for DOCX, BeautifulSoup for HTML
(drop script / style / nav tags), plain read for TXT.
"""
from pathlib import Path

SUPPORTED = {".pdf", ".docx", ".txt", ".html", ".htm"}
MIN_CHARS_PER_PAGE = 30  # below this on average, treat the PDF as a scan


def extract_folder(folder: Path) -> tuple[list[dict], list[tuple[str, str]]]:
    raise NotImplementedError("Phase 1")


def extract_file(path: Path) -> list[dict]:
    raise NotImplementedError("Phase 1")
