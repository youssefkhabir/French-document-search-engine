"""Phase 3 tests: embeddings, semantic search, index files.

The end-to-end tests use the small e5 model (384 d, ~0.5 GB), not BGE-M3, so they stay
quick and also run on a CPU-only machine. The model must be in models/hub
(python scripts/download_models.py).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from search import config, semantic
from search.engine import build_index, load_index, search

CORPUS = Path(__file__).parent / "fixtures" / "corpus"
SMALL = config.EMBED_MODEL_FALLBACK   # intfloat/multilingual-e5-small


# --- search_semantic on hand-made vectors ---------------------------------------------

def test_search_semantic_ranks_by_cosine():
    E = np.array([[1, 0], [0.6, 0.8], [0, 1], [-1, 0]], dtype=np.float32)
    q = np.array([1, 0], dtype=np.float32)
    hits = semantic.search_semantic(q, E, 10)
    assert [row for row, _ in hits] == [0, 1, 2, 3]            # every row is scored
    assert hits[0][1] == pytest.approx(1.0) and hits[-1][1] == pytest.approx(-1.0)
    assert [row for row, _ in semantic.search_semantic(q, E, 2)] == [0, 1]
    mask = np.array([False, True, True, False])
    assert [row for row, _ in semantic.search_semantic(q, E, 10, mask)] == [1, 2]
    assert semantic.search_semantic(q, E, 10, np.zeros(4, dtype=bool)) == []


# --- the model ------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    return semantic.load_model(SMALL)


def test_vectors_are_normalized(model):
    E = semantic.embed_passages(model, ["Le chat dort.", "La voiture roule vite."], SMALL)
    q = semantic.embed_query(model, "un animal qui dort", SMALL)
    assert E.dtype == np.float32 and E.shape == (2, 384)
    assert np.allclose(np.linalg.norm(E, axis=1), 1, atol=1e-3)
    assert np.linalg.norm(q) == pytest.approx(1, abs=1e-3)
    assert E[0] @ q > E[1] @ q                                   # meaning, not shared words


def test_e5_prefixes_are_applied(model):
    # e5 was trained with "query: " / "passage: ": the same text must give different vectors.
    q = semantic.embed_query(model, "infirmière de nuit", SMALL)
    p = semantic.embed_passages(model, ["infirmière de nuit"], SMALL)[0]
    assert not np.allclose(q, p, atol=1e-3)


def test_model_is_loaded_once(model):
    assert semantic.load_model(SMALL) is model


# --- index and engine (end to end on the fixtures) -----------------------------------------

@pytest.fixture(scope="module")
def index_dir(tmp_path_factory, model):
    d = tmp_path_factory.mktemp("index")
    build_index(CORPUS, d, embed_model=SMALL)
    return d


def test_embedding_files(index_dir):
    E = np.load(index_dir / "embeddings.npy")
    info = json.loads((index_dir / "embeddings.json").read_text(encoding="utf-8"))
    n_chunks = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))["n_chunks"]
    assert E.shape == (n_chunks, 384) and info["model"] == SMALL and info["dim"] == 384
    assert "semantic" in load_index(index_dir)["modes"]


def test_semantic_finds_a_document_sharing_no_word(index_dir):
    # Type-C query: no word of the query is in the nurse's CV, so BM25 finds nothing.
    q = "quelqu'un qui soigne les malades la nuit"
    assert search(q, mode="bm25", index_dir=index_dir)["results"] == []
    r = search(q, mode="semantic", index_dir=index_dir)
    assert r["results"][0]["doc_id"] == "cv/cv_infirmiere.txt"
    assert r["results"][0]["rank_semantic"] == 1 and r["results"][0]["rank_bm25"] is None
    assert "embed" in r["timings"]


def test_semantic_uses_the_index_model_not_the_config(index_dir):
    # config.EMBED_MODEL is BGE-M3 (1024 d); this index is e5 (384 d). A query embedded with
    # the config model would not even have the right dimension.
    assert config.EMBED_MODEL != SMALL
    assert search("enseigner les mathématiques", mode="semantic", index_dir=index_dir)["results"]


def test_semantic_filters_and_one_result_per_document(index_dir):
    r = search("protéger le réseau informatique", mode="semantic", index_dir=index_dir,
               formats=["pdf"], top_k=30)
    ids = [x["doc_id"] for x in r["results"]]
    assert ids and all(x["format"] == "pdf" for x in r["results"])
    assert len(ids) == len(set(ids))
    chunks = load_index(index_dir)["chunks"]
    assert r["n_chunks_matched"] == (chunks["format"] == "pdf").sum()   # every PDF chunk scored
    scores = [x["score"] for x in r["results"]]
    assert scores == sorted(scores, reverse=True)


def test_lexical_only_rebuild_removes_stale_embeddings(tmp_path):
    folder = tmp_path / "corpus"
    folder.mkdir()
    (folder / "a.txt").write_text("Le chat dort sur le canapé.", encoding="utf-8")
    d = tmp_path / "index"
    build_index(folder, d, embed_model=SMALL)
    assert (d / "embeddings.npy").exists()
    build_index(folder, d, embed=False)
    assert not (d / "embeddings.npy").exists()
    assert load_index(d)["modes"] == ("tfidf", "bm25", "word2vec")
