"""Phase 1 · Regex cleaning of extracted text (course technique: expressions régulières).

Input format (from extract.py): paragraphs separated by "\\n\\n", lines inside a paragraph
separated by "\\n". Output: same paragraph structure, each paragraph on one line.

Accents and capital letters are KEPT: the embedding model needs natural text. The lexical
form (lowercase, lemmas, no accents) is built later by lexical.lexical_prep.
"""
import math
import re
import unicodedata

# Characters PDFs and Word files use that hurt matching.
CHAR_MAP = str.maketrans({
    " ": " ", " ": " ", " ": " ", " ": " ",  # non-breaking / thin spaces
    "­": "",                                                # soft hyphen
    "’": "'", "‘": "'", "ʼ": "'",                 # typographic apostrophes
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    "‐": "-", "‑": "-",                                # Unicode hyphens
})

# word-\nword : a word cut at the end of a line.
RE_HYPHEN_BREAK = re.compile(r"(\w+)-\n[ \t]*(\w+)")
RE_SPACES = re.compile(r"[ \t\f\v]+")
RE_DIGITS = re.compile(r"\d+")
# "3", "Page 3", "page 3 / 12", "3 sur 12", "- 3 -"
RE_PAGE_NUMBER = re.compile(r"^[-–\s]*(page\s*)?\d{1,4}(\s*(/|sur|of)\s*\d{1,4})?[-–\s]*$",
                            re.IGNORECASE)

REPEAT_MIN_SHARE = 0.6  # a paragraph on >= 60% of a document's pages is a header/footer


def clean_text(text: str) -> str:
    """Normalize Unicode, rejoin hyphenated line ends, unwrap lines, collapse whitespace."""
    text = unicodedata.normalize("NFC", text).translate(CHAR_MAP)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _rejoin_hyphens(text)
    # Lines inside a paragraph are layout, not meaning: join them with a space.
    paragraphs = []
    for para in re.split(r"\n\s*\n", text):
        para = RE_SPACES.sub(" ", para.replace("\n", " ")).strip()
        if para:
            paragraphs.append(para)
    return "\n\n".join(paragraphs)


def _rejoin_hyphens(text: str) -> str:
    """dévelop-\\npement -> développement, but keep real compounds (aides-\\nsoignants, Wi-\\nFi).

    A hyphen is kept when the second part starts with a capital letter, or when the
    hyphenated form appears elsewhere in the same text (on one line).
    """
    lowered = text.lower()

    def repl(m: re.Match) -> str:
        left, right = m.group(1), m.group(2)
        compound = f"{left}-{right}".lower()
        if right[0].isupper() or lowered.count(compound) > 0:
            return f"{left}-{right}"
        return left + right

    return RE_HYPHEN_BREAK.sub(repl, text)


def _key(paragraph: str) -> str:
    """Comparison key: case, digits and spacing ignored, so 'Page 1 / 3' == 'Page 2 / 3'."""
    return RE_SPACES.sub(" ", RE_DIGITS.sub("#", paragraph.lower())).strip()


def remove_repeated_lines(pages: list[str]) -> tuple[list[str], list[str]]:
    """Drop headers, footers and page numbers from the pages of ONE document.

    A paragraph is dropped when it is a bare page number, or when (digits ignored) it
    appears on at least 60% of the pages of a document with 2 pages or more.
    Returns (cleaned pages, removed paragraphs) — the second list feeds the report's
    before/after example.
    """
    split = [[p for p in re.split(r"\n\s*\n", page) if p.strip()] for page in pages]
    n = len(pages)
    repeated: set[str] = set()
    if n >= 2:
        counts: dict[str, int] = {}
        for paras in split:
            for key in {_key(p) for p in paras}:
                counts[key] = counts.get(key, 0) + 1
        threshold = max(2, math.ceil(REPEAT_MIN_SHARE * n))
        repeated = {k for k, c in counts.items() if c >= threshold}

    out, removed = [], []
    for paras in split:
        kept = []
        for p in paras:
            if _key(p) in repeated or RE_PAGE_NUMBER.match(p.strip()):
                removed.append(p.strip())
            else:
                kept.append(p)
        out.append("\n\n".join(kept))
    return out, removed
