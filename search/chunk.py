"""Phase 1 · Sentence-aware chunker (course technique: segmentation en phrases).

Each document is cut into sentences (regex splitter below); a sentence longer than the
limit (e.g. a list without punctuation) is cut into fixed windows of words. Sentences are
packed into chunks of at most CHUNK_MAX_WORDS words, keeping paragraph breaks. A new
chunk starts with the last sentences of the previous one, up to CHUNK_OVERLAP_WORDS words,
so an idea cut at a boundary is still found whole in one chunk. Short documents (a
one-page CV) stay a single chunk.

Chunk record: {"chunk_id", "doc_id", "title", "page", "n_words", "text_raw"}
(`text_lex` is added later by lexical.lexical_prep).
"""
import re

from search import config

# Abbreviations whose final dot does not end a sentence.
ABBREVIATIONS = {
    "m", "mme", "mmes", "mlle", "mm", "dr", "pr", "me", "mgr", "st", "ste",
    "etc", "cf", "ex", "env", "art", "al", "fig", "vol", "chap", "p", "pp",
    "n", "no", "tél", "tel", "av", "bd", "réf", "ref", "min", "max", "approx",
    "janv", "févr", "fév", "avr", "juil", "sept", "oct", "nov", "déc",
}
# Candidate boundary: . ! ? or … (optionally followed by a closing quote/bracket),
# then spaces, then something that can start a sentence.
RE_BOUNDARY = re.compile(r"([.!?…]+[»\")\]]*)\s+(?=[«\"(\[]?[A-ZÀ-ÖØ-Þ0-9])")
RE_WORD_BEFORE_DOT = re.compile(r"(\w+)\.$")


def split_sentences(text: str) -> list[str]:
    """Regex sentence splitter for French, aware of common abbreviations and initials."""
    sentences, start = [], 0
    for m in RE_BOUNDARY.finditer(text):
        candidate = text[start:m.end(1)]
        if m.group(1).startswith("."):
            w = RE_WORD_BEFORE_DOT.search(candidate)
            if w and (w.group(1).lower() in ABBREVIATIONS
                      or (len(w.group(1)) == 1 and w.group(1).isupper())):  # "J. Dupont"
                continue
        sentences.append(candidate.strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]


def _units(text: str, page: int, max_words: int) -> list[tuple[str, int, int, bool]]:
    """Cut one page into (sentence, n_words, page, starts_paragraph) units."""
    units = []
    for para in text.split("\n\n"):
        first = True
        for sent in split_sentences(para.strip()):
            words = sent.split()
            for i in range(0, len(words), max_words):  # one window unless the sentence is huge
                piece = words[i:i + max_words]
                units.append((" ".join(piece), len(piece), page, first))
                first = False
    return units


def _join(units: list[tuple[str, int, int, bool]]) -> str:
    """Sentences of one paragraph joined by a space, paragraphs by a newline."""
    out = ""
    for k, u in enumerate(units):
        out += u[0] if k == 0 else ("\n" if u[3] else " ") + u[0]
    return out


def chunk_document(pages: list[dict], max_words: int | None = None,
                   overlap_words: int | None = None, min_tail_words: int = 50) -> list[dict]:
    """Chunk the cleaned pages of ONE document (dicts with doc_id, title, page, text)."""
    max_words = max_words or config.CHUNK_MAX_WORDS
    overlap_words = config.CHUNK_OVERLAP_WORDS if overlap_words is None else overlap_words
    if not pages:
        return []
    doc_id, title = pages[0]["doc_id"], pages[0]["title"]

    units = [u for p in pages for u in _units(p["text"], p["page"], max_words)]
    groups: list[list[tuple[str, int, int, bool]]] = []  # each group = units of one chunk
    firsts: list[int] = []  # index of the first NEW (non-overlap) unit in each group
    current: list[tuple[str, int, int, bool]] = []
    n_overlap = 0
    for unit in units:
        size = sum(u[1] for u in current)
        if current and size + unit[1] > max_words and len(current) > n_overlap:
            groups.append(current)
            firsts.append(n_overlap)
            # Start the next chunk with the tail of this one (whole units only).
            tail, tail_words = [], 0
            for u in reversed(current):
                if tail_words + u[1] > overlap_words:
                    break
                tail.insert(0, u)
                tail_words += u[1]
            # Overlap must leave room for the new unit.
            while tail and tail_words + unit[1] > max_words:
                tail_words -= tail.pop(0)[1]
            current, n_overlap = tail, len(tail)
        current.append(unit)
    if len(current) > n_overlap:
        new_words = sum(u[1] for u in current[n_overlap:])
        if groups and new_words < min_tail_words:
            # A tiny last piece reads badly alone: append it to the previous chunk.
            groups[-1].extend(current[n_overlap:])
        else:
            groups.append(current)
            firsts.append(n_overlap)

    chunks = []
    for k, (group, first) in enumerate(zip(groups, firsts)):
        text = _join(group)
        chunks.append({
            "chunk_id": f"{doc_id}#{k}",
            "doc_id": doc_id,
            "title": title,
            "page": group[first][2],  # page where the chunk's new content starts
            "n_words": len(text.split()),
            "text_raw": text,
        })
    return chunks


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Chunk every document in a list of cleaned pages (as returned by extract + clean)."""
    by_doc: dict[str, list[dict]] = {}
    for p in pages:
        by_doc.setdefault(p["doc_id"], []).append(p)
    chunks = []
    for doc_pages in by_doc.values():
        chunks.extend(chunk_document(sorted(doc_pages, key=lambda p: p["page"])))
    return chunks
