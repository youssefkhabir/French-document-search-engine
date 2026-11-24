"""Phase 4 tests: typo correction, fusion, reranker, hybrid search."""
from pathlib import Path

import pytest

from search import config, fusion, spelling
from search.engine import build_index, search

CORPUS = Path(__file__).parent / "fixtures" / "corpus"
SMALL = config.EMBED_MODEL_FALLBACK

VOCAB = {"infirmière": 40, "infirmier": 25, "rationnelle": 12, "photosynthèse": 9,
         "française": 347, "francaise": 6, "théorème": 30, "fonction": 200, "nuit": 50,
         "équivalente": 8, "rationnel": 3, "rare": 1, "raree": 1}


@pytest.fixture(scope="module")
def speller():
    return spelling.Speller(VOCAB)


# --- spelling -------------------------------------------------------------------------

@pytest.mark.parametrize("query, expected", [
    ("infirmiere de nuit", "infirmière de nuit"),           # accents only (distance 0)
    ("la fonction rationelle", "la fonction rationnelle"),    # 1 edit
    ("la photosinthèse", "la photosynthèse"),                 # 1 edit, accents kept
    ("theoreme", "théorème"),
    ("la francaise", "la française"),                         # in vocab but 50x rarer
    ("fonction", "fonction"),                                 # known word
    ("norme ISO Bonaparte", "norme ISO Bonaparte"),           # capitals: protected
    ("version 2022b", "version 2022b"),                       # digits: protected
    ("nuiit", "nuit"),                                        # short word, 1 edit
    ("xyzwvut", "xyzwvut"),                                   # nothing close: kept
])
def test_correct_query(speller, query, expected):
    assert spelling.correct_query(query, speller)[0] == expected


def test_short_words_and_rare_targets_untouched(speller):
    assert spelling.correct_query("nui", speller)[0] == "nui"       # < TYPO_MIN_LEN letters
    assert spelling.correct_query("rarre", speller)[0] == "rarre"   # "rare" appears only once


def test_corrections_are_reported(speller):
    fixed, corrections = spelling.correct_query("infirmiere rationelle", speller)
    assert corrections == [("infirmiere", "infirmière"), ("rationelle", "rationnelle")]


def test_capitals_query_is_lowercased_then_corrected(speller):
    q = "INFIRMIERE DE NUIT SVP"
    fixed, corrections = spelling.correct_query(q, speller)
    assert fixed == "infirmière de nuit svp"
    assert corrections[0] == (q, q.lower())                   # whole-query correction first


def test_fix_capitals():
    assert spelling.fix_capitals("BONJOUR LE PASSÉ COMPOSÉ") == "bonjour le passé composé"
    assert spelling.fix_capitals("la norme ISO 27001") == "la norme ISO 27001"


def test_acronym_in_a_normal_query_is_not_lowercased(speller):
    assert spelling.correct_query("la norme ISO pour les infirmieres", speller)[0].count("ISO") == 1


def test_build_vocabulary():
    v = spelling.build_vocabulary(["L'Infirmière de nuit.", "infirmière 2024 a"])
    assert v == {"infirmière": 2, "de": 1, "nuit": 1}         # lowercase, no digits, 2+ letters


# --- fusion -----------------------------------------------------------------------------

def test_minmax():
    assert fusion.minmax({1: 2.0, 2: 4.0, 3: 3.0}) == {1: 0.0, 2: 1.0, 3: 0.5}
    assert fusion.minmax({7: 5.0}) == {7: 1.0}                # max == min guard
    assert fusion.minmax({}) == {}


LEX = [(1, 10.0), (2, 5.0), (3, 0.0)]          # BM25 scores
SEM = [(2, 0.9), (4, 0.8), (1, 0.5)]           # cosine


def test_weighted_fusion_extremes_and_middle():
    assert [r for r, _ in fusion.weighted_fusion(LEX, SEM, 0.0)][:2] == [1, 2]   # BM25 order
    assert fusion.weighted_fusion(LEX, SEM, 1.0)[0][0] == 2                      # semantic best
    fused = dict(fusion.weighted_fusion(LEX, SEM, 0.5))
    # 2: lex 0.5, sem 1.0 -> 0.75 ; 1: lex 1.0, sem 0.0 -> 0.5 ; 4: sem 0.75 -> 0.375
    assert fused[2] == pytest.approx(0.75) and fused[1] == pytest.approx(0.5)
    assert fused[4] == pytest.approx(0.375) and fused[3] == pytest.approx(0.0)   # absent -> 0


def test_rrf_fusion():
    fused = fusion.rrf_fusion(LEX, SEM, k=60)
    assert fused[0][0] in (1, 2)                                # in both lists, top ranks
    assert dict(fused)[2] == pytest.approx(1 / 62 + 1 / 61)
    assert dict(fused)[4] == pytest.approx(1 / 62)


# --- engine (end to end, e5-small, no reranker) ---------------------------------------------

@pytest.fixture(scope="module")
def index_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("index")
    build_index(CORPUS, d, embed_model=SMALL)
    return d


def test_vocabulary_saved_with_the_index(index_dir):
    assert (index_dir / "vocab.json").is_file()


def test_hybrid_gets_both_kinds_of_query(index_dir):
    # A word query (BM25's strength) and a no-shared-word query (semantic's strength).
    r = search("développeur backend Python", mode="hybrid", index_dir=index_dir)
    assert r["results"][0]["doc_id"] == "offres/offre_developpeur_backend.html"
    r = search("quelqu'un qui soigne les malades la nuit", mode="hybrid", index_dir=index_dir)
    assert "cv/cv_infirmiere.txt" in [x["doc_id"] for x in r["results"][:3]]
    assert r["fusion"] == "weighted" and r["alpha"] == config.DEFAULT_ALPHA


def test_hybrid_alpha_extremes_follow_the_branches(index_dir):
    q = "protéger le réseau informatique"
    bm25 = search(q, mode="bm25", index_dir=index_dir)["results"][0]["doc_id"]
    sem = search(q, mode="semantic", index_dir=index_dir)["results"][0]["doc_id"]
    assert search(q, mode="hybrid", alpha=0.0, index_dir=index_dir)["results"][0]["doc_id"] == bm25
    assert search(q, mode="hybrid", alpha=1.0, index_dir=index_dir)["results"][0]["doc_id"] == sem


def test_hybrid_reports_branch_ranks_and_rrf(index_dir):
    r = search("infirmière de nuit", mode="hybrid", fusion_method="rrf", index_dir=index_dir)
    top = r["results"][0]
    assert top["rank_bm25"] is not None and top["rank_semantic"] is not None
    assert r["alpha"] is None and r["fusion"] == "rrf"


def test_typo_fix_in_search(index_dir):
    r = search("infirmiere de nuit", mode="bm25", index_dir=index_dir)
    assert ("infirmiere", "infirmière") in r["corrections"]
    assert "infirmier" in r["query_lex"].split()             # now lemmatized (Phase 2 gap)
    top = [x["doc_id"] for x in r["results"][:2]]
    assert set(top) == {"cv/cv_infirmiere.txt", "offres/offre_infirmier_ehpad.txt"}
    off = search("infirmiere de nuit", mode="bm25", index_dir=index_dir, typo_fix=False)
    assert off["corrections"] == [] and off["corrected_query"] == "infirmiere de nuit"
    assert "infirmiere" in off["query_lex"].split()          # the gap, without correction


def test_no_snippets_option(index_dir):
    r = search("infirmière", mode="hybrid", index_dir=index_dir, snippets=False)
    assert r["results"] and all(x["snippet"] == "" for x in r["results"])


def test_unknown_fusion(index_dir):
    with pytest.raises(ValueError):
        search("test", mode="hybrid", fusion_method="max", index_dir=index_dir)


# --- WordNet expansion (optional, course Ch. 3) ----------------------------------------

from search import wordnet_expand  # noqa: E402


def test_wordnet_synonyms_respect_pos_and_filters():
    syns = wordnet_expand.synonyms_fr("voiture", 5, pos="NOUN")
    assert "automobile" in syns
    assert all("_" not in s and s.islower() for s in syns)
    assert wordnet_expand.synonyms_fr("voiture", 5, pos="NOUN", known=lambda f: False) == []
    assert len(wordnet_expand.synonyms_fr("travailler", 2, pos="VERB")) <= 2


def test_wordnet_expand_query_repeats_originals():
    q, added = wordnet_expand.expand_query("une voiture", max_terms=1, original_repeat=2,
                                           known=lambda f: f == "automobile")
    assert added == {"voiture": ["automobile"]}
    assert q.split() == ["voiture", "voiture", "automobile"]
    q, added = wordnet_expand.expand_query("xyzwvut", max_terms=2, original_repeat=2)
    assert added == {}


def test_wu_palmer():
    assert wordnet_expand.wu_palmer_fr("guerre", "conflit") == 1.0
    assert 0 < wordnet_expand.wu_palmer_fr("voiture", "camion") < 1
    assert wordnet_expand.wu_palmer_fr("xyzwvut", "voiture") is None


def test_search_with_wordnet(tmp_path):
    build_index(CORPUS, tmp_path, embed=False)
    plain = search("développeur", mode="bm25", index_dir=tmp_path)
    r = search("développeur", mode="bm25", wordnet=True, index_dir=tmp_path)
    assert "expansions" in r and r["results"]
    assert r["query_lex"].startswith(plain["query_lex"])
    sem_free = search("développeur", mode="tfidf", wordnet=True, index_dir=tmp_path)
    assert sem_free["results"]


# --- word2vec, S6 (optional, course Ch. 3) ---------------------------------------------

import numpy as np  # noqa: E402

from search import word2vec  # noqa: E402

W2V_TEXTS = ["chat chien animal maison", "chat animal domestique", "chien animal maison jardin",
             "fraction nombre calcul", "nombre calcul division fraction"] * 20


@pytest.fixture(scope="module")
def kv():
    return word2vec.train_or_load(W2V_TEXTS)


def test_w2v_vectors_are_normalized(kv):
    v = word2vec.doc_vector(kv, "chat chien inconnu")
    assert v.shape == (config.W2V_DIM,) and np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)
    assert not word2vec.doc_vector(kv, "xyzwvut").any()      # no known word -> zero vector
    idf = word2vec.idf_weights(W2V_TEXTS)
    assert idf["jardin"] > idf["animal"]


def test_w2v_search_and_neighbours(kv):
    W = word2vec.doc_matrix(kv, W2V_TEXTS[:5])
    hits = word2vec.search_w2v(word2vec.doc_vector(kv, "division fraction"), W, 5)
    assert hits[0][0] in (3, 4)
    assert word2vec.search_w2v(word2vec.doc_vector(kv, "xyzwvut"), W, 5) == []
    mask = np.array([True, True, True, False, False])
    assert all(r < 3 for r, _ in word2vec.search_w2v(word2vec.doc_vector(kv, "fraction"), W, 5, mask))
    assert word2vec.neighbours(kv, "chat", 3) and word2vec.neighbours(kv, "xyzwvut") == []


def test_w2v_tiny_corpus_has_no_vocabulary():
    kv = word2vec.train_or_load(["un seul passage"])
    assert len(kv.key_to_index) == 0
    assert not word2vec.doc_vector(kv, "seul").any()


def test_search_word2vec_mode(tmp_path):
    s = build_index(CORPUS, tmp_path, embed=False)
    assert s["word2vec"]["n"] == s["n_chunks"]
    r = search("développeur python", mode="word2vec", index_dir=tmp_path)
    assert r["results"] and r["mode"] == "word2vec"
    # An index without w2v files (built before S6) gets them on first load.
    for name in ("w2v.kv", "w2v_matrix.npy", "w2v.json"):
        (tmp_path / name).unlink()
    from search.engine import _INDEX_CACHE
    _INDEX_CACHE.clear()
    assert search("développeur python", mode="word2vec", index_dir=tmp_path)["results"]
    assert (tmp_path / "w2v_matrix.npy").exists()
