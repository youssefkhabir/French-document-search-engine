"""Phase 2 tests: lexical preprocessing, TF-IDF, BM25, engine.search, snippets."""
from pathlib import Path

import numpy as np
import pytest

from search import lexical
from search.engine import build_index, search

CORPUS = Path(__file__).parent / "fixtures" / "corpus"


# --- lexical_prep ----------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("Les développeurs travaillaient sur des applications.", "developpeur travailler application"),
    ("candidat pour un poste de développeur", "candidat poste developpeur"),
    ("est-ce que", ""),                                   # stopwords, incl. "-ce"
    ("Norme ISO 27001, version 2022 !", "norme iso 27001 version 2022"),  # numbers kept
])
def test_lexical_prep(text, expected):
    assert lexical.lexical_prep(text) == expected


def test_lexical_prep_options():
    text = "Les développeurs travaillaient"
    assert lexical.lexical_prep(text, lemmatize=False) == "developpeurs travaillaient"
    assert lexical.lexical_prep(text, remove_stopwords=False).split()[0] == "le"


def test_documents_and_queries_share_the_same_form():
    # Plural/singular, accents and case meet on the same term.
    doc = lexical.lexical_prep_many(["Les Infirmières de nuit"])[0]
    assert lexical.lexical_prep("infirmière") in doc.split()


def test_known_gap_query_without_accents():
    # spaCy cannot lemmatize a word typed without its accents: "infirmiere" stays as is,
    # while the document gives "infirmier". Phase 4 (typo/accent correction) should fix it.
    doc = lexical.lexical_prep_many(["Les Infirmières de nuit"])[0]
    assert lexical.lexical_prep("infirmiere") not in doc.split()


def test_strip_accents():
    assert lexical.strip_accents("Été à l'hôpital, œuvre") == "Ete a l'hopital, œuvre"


# --- TF-IDF and BM25 -----------------------------------------------------------------

TEXTS = ["chat noir dormir", "chien courir parc", "chat chien jouer", "voiture rouge rapide"]


@pytest.mark.parametrize("method", ["tfidf", "bm25"])
def test_ranking(method):
    if method == "tfidf":
        vec, mat = lexical.build_tfidf(TEXTS)
        run = lambda q, mask=None: lexical.search_tfidf(q, vec, mat, 10, mask)  # noqa: E731
    else:
        index = lexical.build_bm25(TEXTS)
        run = lambda q, mask=None: lexical.search_bm25(q, index, 10, mask)  # noqa: E731

    hits = run("chat")
    assert [row for row, _ in hits] in ([0, 2], [2, 0])         # only rows with the term
    assert hits[0][1] >= hits[1][1] > 0
    assert run("chat noir")[0][0] == 0                          # both terms beat one term
    assert run("inconnu") == [] and run("") == []               # unknown / empty query
    mask = np.array([False, True, True, True])
    assert [row for row, _ in run("chat", mask)] == [2]          # filter applied


def test_tfidf_and_bm25_round_trip(tmp_path):
    vec, mat = lexical.build_tfidf(TEXTS)
    lexical.save_tfidf(vec, mat, tmp_path / "tfidf.pkl")
    vec2, mat2 = lexical.load_tfidf(tmp_path / "tfidf.pkl")
    assert lexical.search_tfidf("chien", vec2, mat2, 10) == lexical.search_tfidf("chien", vec, mat, 10)
    index = lexical.build_bm25(TEXTS)
    lexical.save_bm25(index, tmp_path / "bm25")
    assert lexical.search_bm25("chien", lexical.load_bm25(tmp_path / "bm25"), 10) == \
        lexical.search_bm25("chien", index, 10)


# --- snippet ---------------------------------------------------------------------------

def test_snippet_highlights_inflected_forms():
    text = "Intro. " + "mot " * 60 + "Nous recrutons des développeurs Python expérimentés. " + "fin " * 60
    snip, spans = lexical.snippet(text, {"developpeur", "python"}, n_words=20)
    assert [snip[a:b] for a, b in spans] == ["développeurs", "Python"]
    assert snip.startswith("… ") and snip.endswith(" …")


def test_snippet_without_match_starts_at_the_beginning():
    snip, spans = lexical.snippet("Premier mot. Second mot.", {"absent"})
    assert snip == "Premier mot. Second mot." and spans == []


# --- engine (end to end on the fixtures) ----------------------------------------------------

@pytest.fixture(scope="module")
def index_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("index")
    build_index(CORPUS, d, embed=False)   # lexical only: no model to load
    return d


def test_index_files_written(index_dir):
    assert (index_dir / "tfidf.pkl").is_file() and (index_dir / "bm25").is_dir()


@pytest.mark.parametrize("mode", ["tfidf", "bm25"])
def test_search_finds_the_right_document(index_dir, mode):
    r = search("développeur backend Python", mode=mode, index_dir=index_dir)
    assert r["results"][0]["doc_id"] == "offres/offre_developpeur_backend.html"
    ids = [x["doc_id"] for x in r["results"]]
    assert len(ids) == len(set(ids))                       # one entry per document
    scores = [x["score"] for x in r["results"]]
    assert scores == sorted(scores, reverse=True)
    assert r["results"][0]["highlights"]                    # snippet shows the matches


def test_search_filters(index_dir):
    r = search("rapport", mode="bm25", index_dir=index_dir, formats=["pdf"])
    assert r["results"] and all(x["format"] == "pdf" for x in r["results"])
    r = search("développeur", mode="bm25", index_dir=index_dir, collections=["divers"])
    assert all(x["collection"] == "divers" for x in r["results"])


def test_search_no_match_and_unready_mode(index_dir):
    assert search("xyzzyq", mode="bm25", index_dir=index_dir)["results"] == []
    with pytest.raises(NotImplementedError):
        search("test", mode="word2vec", index_dir=index_dir)
    for mode in ("semantic", "hybrid"):                # built with embed=False
        with pytest.raises(ValueError):
            search("test", mode=mode, index_dir=index_dir)
    with pytest.raises(ValueError):
        search("test", mode="nope", index_dir=index_dir)
