"""Phase 1 · Regex cleaning of extracted text (course technique: expressions régulières).

Steps: Unicode NFC, rejoin hyphenated line breaks (`(\\w+)-\\n(\\w+)` -> `\\1\\2`),
collapse whitespace and blank lines, drop lines repeated on every page (headers,
footers, page numbers). Accents and capitals are KEPT: the embedding model needs natural text.
"""


def clean_text(text: str) -> str:
    raise NotImplementedError("Phase 1")


def remove_repeated_lines(pages: list[str]) -> list[str]:
    """Drop lines that appear on most pages of the same document."""
    raise NotImplementedError("Phase 1")
