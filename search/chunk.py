"""Phase 1 · Sentence-aware chunker.

Split on paragraphs, then sentences when a paragraph is too long; pack sentences into
chunks of CHUNK_MIN_WORDS..CHUNK_MAX_WORDS words with ~CHUNK_OVERLAP_WORDS of overlap.
Short documents (a one-page CV) stay one chunk.

Chunk record: {"chunk_id", "doc_id", "title", "page", "text_raw"}
(`text_lex` is added later by lexical.lexical_prep).
"""


def chunk_pages(pages: list[dict]) -> list[dict]:
    raise NotImplementedError("Phase 1")
