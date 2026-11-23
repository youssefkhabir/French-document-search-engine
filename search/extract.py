"""Phase 1 · Load a corpus folder and extract text from PDF, DOCX, TXT, HTML and CSV.

Contract
--------
extract_folder(folder) -> (pages, skipped)
    pages:   list of dicts {"doc_id", "title", "path", "format", "page", "text"}:
             one per PDF page, one per non-PDF document (page = 1).
             `text` is raw: paragraphs separated by a blank line ("\\n\\n"), lines inside
             a paragraph by "\\n" (so clean.py can rejoin hyphenated line ends).
    skipped: list of (relative path, reason), e.g. scanned PDFs (OCR is out of scope),
             empty or unreadable files, unsupported formats.

Libraries: PyMuPDF for PDF (page by page, so results can show the page number),
python-docx for DOCX, BeautifulSoup for HTML, plain reading for TXT, the standard
`csv` module for CSV (one paragraph per row, "column : value" pairs).
"""
import csv
import io
import re
from pathlib import Path

SUPPORTED = {".pdf", ".docx", ".txt", ".html", ".htm", ".csv"}
CSV_DELIMITERS = ";,\t|"  # ";" first: French Excel exports use it
RE_NUMBER = re.compile(r"[-+]?\d[\d\s]*([.,]\d+)?\s*%?")  # 12 | -3,5 | 10 000 | 45 %
MIN_CHARS_PER_PAGE = 30  # below this on average, a PDF is treated as a scan
TXT_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")  # latin-1 never fails: last resort

# HTML elements that never hold document content.
HTML_DROP = ["script", "style", "noscript", "nav", "footer", "aside", "form", "iframe",
             "svg", "button", "template"]
# Inline elements unwrapped so a sentence with <strong> or <a> stays on one line.
HTML_INLINE = ["a", "b", "strong", "i", "em", "u", "span", "small", "abbr", "code",
               "sup", "sub", "mark", "font"]


class SkipFile(Exception):
    """Raised when a file is readable but must not be indexed (scan, empty)."""


def extract_folder(folder: Path) -> tuple[list[dict], list[tuple[str, str]]]:
    folder = Path(folder)
    pages: list[dict] = []
    skipped: list[tuple[str, str]] = []
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        rel = path.relative_to(folder).as_posix()
        if path.name.startswith((".", "~$")):  # .gitkeep, Office lock files
            continue
        if path.suffix.lower() not in SUPPORTED:
            skipped.append((rel, f"unsupported format ({path.suffix or 'no extension'})"))
            continue
        try:
            doc_pages = extract_file(path)
        except SkipFile as e:
            skipped.append((rel, str(e)))
            continue
        except Exception as e:  # noqa: BLE001 - a corrupt file must not stop indexing
            skipped.append((rel, f"unreadable: {type(e).__name__}: {e}"))
            continue
        for p in doc_pages:
            p["doc_id"] = rel
        pages.extend(doc_pages)
    return pages, skipped


def extract_file(path: Path) -> list[dict]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        title, texts = _pdf(path)
    elif ext == ".docx":
        title, texts = _docx(path)
    elif ext in (".html", ".htm"):
        title, texts = _html(path)
    elif ext == ".txt":
        title, texts = None, [_read_txt(path)]
    elif ext == ".csv":
        title, texts = None, [_csv(path)]
    else:
        raise SkipFile(f"unsupported format ({ext})")

    if not any(t.strip() for t in texts):
        raise SkipFile("no text extracted (empty document)")
    title = (title or "").strip() or path.stem.replace("_", " ")
    fmt = "html" if ext == ".htm" else ext.lstrip(".")
    return [{"doc_id": path.name, "title": title, "path": str(path), "format": fmt,
             "page": i + 1, "text": t}
            for i, t in enumerate(texts) if t.strip()]


# --- PDF -------------------------------------------------------------------

def _pdf(path: Path) -> tuple[str | None, list[str]]:
    import pymupdf

    with pymupdf.open(path) as doc:
        texts = []
        for page in doc:
            # Blocks follow the visual layout: one block ~ one paragraph. Image blocks
            # (type 1) are skipped; sort=True orders blocks top-to-bottom, left-to-right.
            blocks = page.get_text("blocks", sort=True)
            paras = [b[4].strip() for b in blocks if b[6] == 0 and b[4].strip()]
            texts.append("\n\n".join(paras))
        n_chars = sum(len(t) for t in texts)
        if doc.page_count and n_chars / doc.page_count < MIN_CHARS_PER_PAGE:
            raise SkipFile(f"scanned PDF: {n_chars} characters on {doc.page_count} page(s), "
                           "OCR is out of scope")
        title = (doc.metadata or {}).get("title")
    return title, texts


# --- DOCX ------------------------------------------------------------------

def _docx(path: Path) -> tuple[str | None, list[str]]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(path)
    paras: list[str] = []
    # Walk the body in order so tables (common in CVs) stay where they are.
    # Section headers/footers are not part of the body, so they are left out.
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = Paragraph(child, doc).text.strip()
            if text:
                paras.append(text)
        elif tag == "tbl":
            for row in Table(child, doc).rows:
                cells = []
                for cell in row.cells:
                    t = cell.text.strip()
                    if t and t not in cells:  # merged cells repeat their text
                        cells.append(t)
                if cells:
                    paras.append(" | ".join(cells))
    return doc.core_properties.title, ["\n\n".join(paras)]


# --- HTML ------------------------------------------------------------------

def _html(path: Path) -> tuple[str | None, list[str]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_read_txt(path), "lxml")
    title = soup.title.get_text(strip=True) if soup.title else None
    for tag in soup.find_all(HTML_DROP):
        tag.decompose()
    for tag in soup.find_all(HTML_INLINE):
        tag.unwrap()
    soup.smooth()  # merge the text pieces left side by side by unwrap()
    body = soup.body or soup
    # After unwrapping inline tags, each remaining text line comes from a block element.
    lines = [" ".join(line.split()) for line in body.get_text("\n").split("\n")]
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else None
    return title, ["\n\n".join(line for line in lines if line)]


# --- CSV -------------------------------------------------------------------

def _csv(path: Path) -> str:
    """One paragraph per row. With a header row: 'Col1 : v1 ; Col2 : v2'; empty cells skipped."""
    text = _read_txt(path)
    sample = text[:20000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=CSV_DELIMITERS)
    except csv.Error:  # one column, or no clear delimiter
        found = next((d for d in CSV_DELIMITERS if d in sample), ",")
        dialect = type("Fallback", (csv.excel,), {"delimiter": found})  # never edit csv.excel
    try:
        has_header = csv.Sniffer().has_header(sample)
    except csv.Error:
        has_header = False

    # StringIO, not splitlines(): a quoted cell may contain line breaks.
    rows = [[c.strip() for c in row] for row in csv.reader(io.StringIO(text), dialect)]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    # Sniffer.has_header guesses from types, so a table of text only is often reported as
    # headerless. A first row of 2+ cells with no empty cell, no number and no repeated
    # value is taken as a header too. (With one column, a header cannot be told apart.)
    first = rows[0]
    looks_like_header = (len(first) >= 2 and all(first) and len(set(first)) == len(first)
                         and not any(RE_NUMBER.fullmatch(c) for c in first))
    if (has_header or looks_like_header) and len(rows) > 1:
        header, body = first, rows[1:]
        lines = []
        for row in body:
            pairs = [f"{h or f'colonne {i + 1}'} : {v}"
                     for i, (h, v) in enumerate(zip(header, row)) if v]
            pairs += [v for v in row[len(header):] if v]  # extra cells beyond the header
            if pairs:
                lines.append(" ; ".join(pairs))
    else:
        lines = [" ; ".join(c for c in row if c) for row in rows]
    return "\n\n".join(lines)


# --- TXT -------------------------------------------------------------------

def _read_txt(path: Path) -> str:
    raw = path.read_bytes()
    for enc in TXT_ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise SkipFile("unknown text encoding")  # unreachable with latin-1, kept for safety
