"""OPTIONAL (course Ch. 3) · Averaged word2vec baseline (system S6).

Default: gensim Word2Vec trained on the lexical form of the chunks.
Optional: pretrained French KeyedVectors from config.W2V_PRETRAINED_PATH.
See docs/plan_addendum.md, component B.
"""
import numpy as np


def train_or_load(texts_lex: list[str]):
    raise NotImplementedError("Phase 4 stretch")


def doc_vector(kv, text_lex: str) -> np.ndarray:
    """Mean of the word vectors (L2-normalized); zero vector if no word is known."""
    raise NotImplementedError("Phase 4 stretch")


def search_w2v(q_vec: np.ndarray, W: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    raise NotImplementedError("Phase 4 stretch")
