"""Phase 2 · Lexical branch: preprocessing, TF-IDF baseline, BM25.

ONE RULE: documents and queries go through the same `lexical_prep`.
Pipeline: spaCy tokenization -> lowercase -> drop stopwords/punctuation -> lemma -> strip accents.
Options exist so the ablations (raw words vs lemmas, with/without stopwords) reuse this code.
"""


def lexical_prep(text: str, lemmatize: bool = True, remove_stopwords: bool = True) -> str:
    raise NotImplementedError("Phase 2")


def build_tfidf(texts_lex: list[str]):
    """Return (vectorizer, matrix). Saved as index/tfidf.pkl."""
    raise NotImplementedError("Phase 2")


def search_tfidf(query_lex: str, vectorizer, matrix, top_k: int) -> list[tuple[int, float]]:
    raise NotImplementedError("Phase 2")


def build_bm25(texts_lex: list[str]):
    """Return a bm25s index. Saved under index/bm25/."""
    raise NotImplementedError("Phase 2")


def search_bm25(query_lex: str, index, top_k: int) -> list[tuple[int, float]]:
    raise NotImplementedError("Phase 2")
