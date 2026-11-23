"""Phase 2 · Lexical branch: preprocessing, TF-IDF baseline, BM25.

ONE RULE: documents and queries go through the same `lexical_prep`.
Pipeline: spaCy tokenization -> drop stopwords/punctuation -> lemma -> lowercase -> strip accents.
Options exist so the ablations (raw words vs lemmas, with/without stopwords) reuse this code.

A lexical form is a string of space-separated terms: "developpeur travailler python".
"""
import pickle
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np

from search import config

# spaCy components not needed for lemmas: skipping them makes indexing ~3x faster.
SPACY_DISABLE = ["parser", "ner"]
SPACY_BATCH = 64


@lru_cache(maxsize=1)
def _nlp():
    import spacy
    return spacy.load(config.SPACY_MODEL, disable=SPACY_DISABLE)


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    """spaCy's French stopwords in their final form (lowercase, no accents)."""
    return frozenset(strip_accents(w.lower()) for w in _nlp().Defaults.stop_words)


def strip_accents(text: str) -> str:
    """développé -> developpe (letters kept, combining marks dropped)."""
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def _token_terms(tok, lemmatize: bool, remove_stopwords: bool) -> list[str]:
    """Terms of one spaCy token: usually one, none for punctuation and stopwords."""
    if tok.is_punct or tok.is_space or not any(c.isalnum() for c in tok.text):
        return []
    if remove_stopwords and tok.is_stop:
        return []
    word = tok.lemma_ if lemmatize and tok.lemma_.strip() else tok.text
    out = []
    # spaCy can return multi-word lemmas ("à le"): keep each part.
    for part in strip_accents(word.lower()).split():
        part = part.strip("'’-.")
        # "est-ce" -> "-ce" is not a stopword for spaCy, "ce" is.
        if remove_stopwords and part in _stopwords():
            continue
        if len(part) >= 2 or part.isdigit():
            out.append(part)
    return out


def _terms(doc, lemmatize: bool, remove_stopwords: bool) -> list[str]:
    return [t for tok in doc for t in _token_terms(tok, lemmatize, remove_stopwords)]


def lexical_prep(text: str, lemmatize: bool = True, remove_stopwords: bool = True) -> str:
    """Lexical form of one text (a query, usually)."""
    return " ".join(_terms(_nlp()(text), lemmatize, remove_stopwords))


def lexical_prep_many(texts: list[str], lemmatize: bool = True,
                      remove_stopwords: bool = True) -> list[str]:
    """Lexical forms of many texts (the corpus), batched through spaCy."""
    docs = _nlp().pipe(texts, batch_size=SPACY_BATCH)
    return [" ".join(_terms(d, lemmatize, remove_stopwords)) for d in docs]


def query_terms(text: str) -> set[str]:
    """Distinct terms of a query, for highlighting."""
    return set(lexical_prep(text).split())


def snippet(text: str, terms: set[str], n_words: int = 45) -> tuple[str, list[tuple[int, int]]]:
    """Window of about `n_words` tokens of `text` holding the most query terms.

    A word matches when its lexical form is a query term, so "développeurs" in the text is
    highlighted for the query "développeur". Returns (snippet, [(start, end)] character
    spans of the matches inside the snippet).
    """
    toks = [t for t in _nlp()(text) if not t.is_space]
    if not toks:
        return "", []
    hit = [bool(terms) and any(x in terms for x in _token_terms(t, True, True)) for t in toks]
    hits = [i for i, h in enumerate(hit) if h]

    start = 0
    if hits:
        best = -1
        for h in hits:
            s = max(0, h - 8)
            n = sum(hit[s:s + n_words])
            if n > best:
                best, start = n, s
    end = min(len(toks), start + n_words)

    lo, hi = toks[start].idx, toks[end - 1].idx + len(toks[end - 1])
    prefix = "… " if start > 0 else ""
    body = text[lo:hi].replace("\n", " ")
    spans = [(len(prefix) + toks[i].idx - lo, len(prefix) + toks[i].idx - lo + len(toks[i]))
             for i in range(start, end) if hit[i]]
    return prefix + body + (" …" if end < len(toks) else ""), spans


# --- top-k helper -------------------------------------------------------------

def _top_k(scores: np.ndarray, top_k: int, mask: np.ndarray | None) -> list[tuple[int, float]]:
    """Indices and scores of the best rows with a positive score (rows outside `mask` excluded)."""
    scores = np.asarray(scores, dtype=np.float32).ravel()
    if mask is not None:
        scores = np.where(mask, scores, 0.0)
    k = min(top_k, int((scores > 0).sum()))
    if k == 0:
        return []
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return [(int(i), float(scores[i])) for i in idx]


# --- S1 · TF-IDF ----------------------------------------------------------------

def build_tfidf(texts_lex: list[str]):
    """Return (vectorizer, matrix). Saved as index/tfidf.pkl.

    Terms are already normalized by lexical_prep, so the vectorizer only splits on spaces.
    Rows are L2-normalized, so a dot product with the query vector is the cosine.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts_lex)
    return vectorizer, matrix


def search_tfidf(query_lex: str, vectorizer, matrix, top_k: int,
                 mask: np.ndarray | None = None) -> list[tuple[int, float]]:
    """Cosine similarity between the query and every chunk -> [(row, score)] best first."""
    q = vectorizer.transform([query_lex])
    if q.nnz == 0:
        return []
    scores = (matrix @ q.T).toarray().ravel()
    return _top_k(scores, top_k, mask)


def save_tfidf(vectorizer, matrix, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": matrix}, f)


def load_tfidf(path: Path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["vectorizer"], d["matrix"]


# --- S2 · BM25 -------------------------------------------------------------------

def build_bm25(texts_lex: list[str]):
    """Return a bm25s index (Lucene variant, k1=1.5, b=0.75). Saved under index/bm25/."""
    import bm25s

    index = bm25s.BM25()
    index.index([t.split() for t in texts_lex], show_progress=False)
    return index


def search_bm25(query_lex: str, index, top_k: int,
                mask: np.ndarray | None = None) -> list[tuple[int, float]]:
    """BM25 score of every chunk -> [(row, score)] best first. Unknown query terms are ignored."""
    ids = index.get_tokens_ids(query_lex.split())
    if not ids:
        return []
    return _top_k(index.get_scores(ids), top_k, mask)


def save_bm25(index, path: Path) -> None:
    index.save(str(path), show_progress=False)


def load_bm25(path: Path):
    import bm25s
    return bm25s.BM25.load(str(path), show_progress=False)
