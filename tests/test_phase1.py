"""Phase 1 tests: extraction, cleaning, sentence splitting, chunking, ingestion.

Run: python -m pytest -q
Fixtures: tests/fixtures/corpus (synthetic, rebuilt by scripts/make_fixtures.py).
"""
from pathlib import Path

import pytest

from search.chunk import chunk_document, split_sentences
from search.clean import clean_text, remove_repeated_lines
from search.engine import ingest
from search.extract import SkipFile, extract_file, extract_folder

CORPUS = Path(__file__).parent / "fixtures" / "corpus"


# --- clean -------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("dévelop-\npement", "développement"),                          # line-end hyphen
    ("aides-soignants et aides-\nsoignants", "aides-soignants et aides-soignants"),  # real compound
    ("réseau Wi-\nFi", "réseau Wi-Fi"),                              # capital after the cut
    ("l’expérience", "l'expérience"),                          # typographic apostrophe
    ("10 000 €", "10 000 €"),                              # non-breaking spaces
    ("été", "été"),                                      # Unicode NFC
    ("ﬁlière", "filière"),                                      # PDF ligature
    ("ligne un\nligne deux", "ligne un ligne deux"),                 # wrapped lines
    ("a   b\t c", "a b c"),                                          # spaces
])
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_clean_keeps_paragraphs_and_accents():
    assert clean_text("Été\n\n\n\nHiver  ÉTÉ") == "Été\n\nHiver ÉTÉ"


def test_remove_repeated_lines():
    pages = [f"Ministère X - Offre 12\n\nContenu {w}.\n\nPage {i} / 3"
             for i, w in enumerate(["un", "deux", "trois"], 1)]
    kept, removed = remove_repeated_lines(pages)
    assert kept == ["Contenu un.", "Contenu deux.", "Contenu trois."]
    assert "Page 2 / 3" in removed and "Ministère X - Offre 12" in removed


def test_single_page_keeps_content_drops_page_number():
    kept, removed = remove_repeated_lines(["Titre\n\nTexte.\n\n1"])
    assert kept == ["Titre\n\nTexte."] and removed == ["1"]


# --- sentences ---------------------------------------------------------------

def test_split_sentences_french_abbreviations():
    text = ("M. Dupont travaille avec Mme Martin, etc. Il est arrivé en 2020. "
            "J. Durand est parti ! Pourquoi ? Voir l'art. 3 du contrat.")
    assert split_sentences(text) == [
        "M. Dupont travaille avec Mme Martin, etc. Il est arrivé en 2020.",
        "J. Durand est parti !",
        "Pourquoi ?",
        "Voir l'art. 3 du contrat.",
    ]


# --- chunk -------------------------------------------------------------------

def _doc(paragraphs: list[str], pages: int = 1) -> list[dict]:
    per = max(1, len(paragraphs) // pages)
    return [{"doc_id": "d", "title": "T", "page": p + 1,
             "text": "\n\n".join(paragraphs[p * per:(p + 1) * per] if p < pages - 1
                                 else paragraphs[p * per:])}
            for p in range(pages)]


def _sentence(i: int) -> str:
    return f"Phrase numéro {i} avec quelques mots de remplissage pour le test."  # 11 words


def test_short_document_is_one_chunk():
    chunks = chunk_document(_doc(["Titre", _sentence(1), _sentence(2)]))
    assert len(chunks) == 1
    assert chunks[0]["text_raw"] == f"Titre\n{_sentence(1)}\n{_sentence(2)}"


def test_long_document_sizes_overlap_and_pages():
    paras = [" ".join(_sentence(10 * p + s) for s in range(5)) for p in range(12)]  # 660 words
    chunks = chunk_document(_doc(paras, pages=3), max_words=250, overlap_words=40)
    assert len(chunks) >= 3
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    assert all(c["n_words"] <= 250 + 50 for c in chunks)  # +50: small tail may be merged
    for prev, nxt in zip(chunks, chunks[1:]):
        first_sentence = split_sentences(nxt["text_raw"].split("\n")[0])[0]
        assert first_sentence in prev["text_raw"], "next chunk must start with overlap"
    assert chunks[0]["page"] == 1 and chunks[-1]["page"] >= 2


def test_sentence_without_punctuation_is_windowed():
    chunks = chunk_document(_doc([" ".join(["mot"] * 600)]), max_words=250, overlap_words=0)
    assert [c["n_words"] for c in chunks] == [250, 250, 100]


# --- extract -----------------------------------------------------------------

def test_scanned_pdf_is_skipped():
    with pytest.raises(SkipFile, match="scanned PDF"):
        extract_file(CORPUS / "cv" / "cv_scanne_agent_accueil.pdf")


def test_pdf_pages_and_title():
    pages = extract_file(CORPUS / "offres" / "offre_rssi_ministere.pdf")
    assert [p["page"] for p in pages] == [1, 2, 3]
    assert pages[0]["title"].startswith("Offre - Responsable")


def test_html_drops_navigation_scripts_and_keeps_inline_text():
    text = extract_file(CORPUS / "offres" / "offre_developpeur_backend.html")[0]["text"]
    assert "Connexion" not in text and "dataLayer" not in text and "Offres similaires" not in text
    assert "Entreprise : Startup Fictive Mobilité, Lyon." in text


def test_txt_in_cp1252_is_decoded():
    text = extract_file(CORPUS / "cv" / "cv_professeure_maths.txt")[0]["text"]
    assert "Expérience" in text and "mathématiques" in text


def test_docx_body_only():
    text = extract_file(CORPUS / "cv" / "cv_auditrice_iso27001.docx")[0]["text"]
    assert "ISO 27001 Lead Auditor" in text
    assert "document confidentiel" not in text  # section header is not body text


def test_extract_folder_reports_unsupported(tmp_path):
    (tmp_path / "a.txt").write_text("Bonjour.", encoding="utf-8")
    (tmp_path / "b.xlsx").write_bytes(b"x")
    (tmp_path / "c.pdf").write_bytes(b"not a pdf")
    pages, skipped = extract_folder(tmp_path)
    assert [p["doc_id"] for p in pages] == ["a.txt"]
    reasons = dict(skipped)
    assert "unsupported" in reasons["b.xlsx"] and "unreadable" in reasons["c.pdf"]


# --- ingest (end to end) -----------------------------------------------------

def test_ingest_fixture_corpus():
    chunks, skipped, report = ingest(CORPUS)
    assert report["n_documents"] == 15
    assert [p for p, _ in skipped] == ["cv/cv_scanne_agent_accueil.pdf"]
    text = "\n".join(chunks["text_raw"])
    for noise in ["Page 1 / 3", "Offre n° 2026-114", "thomas.fictif@example.org", "analytics"]:
        assert noise not in text
    for joined in ["traitement", "configuration", "management"]:
        assert joined in text
    assert (chunks["n_words"] > 0).all()
    assert chunks.groupby("doc_id").size()["offres/offre_rssi_ministere.pdf"] == 2
