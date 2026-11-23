"""Phase 1 tests: extraction, cleaning, sentence splitting, chunking, ingestion.

Run: python -m pytest -q
Fixtures: tests/fixtures/corpus (synthetic, rebuilt by scripts/make_fixtures.py).
"""
from pathlib import Path

import pytest

from search.chunk import chunk_document, split_sentences
from search.clean import clean_text, remove_repeated_lines
from search.engine import ingest
from search.extract import SkipFile, clean_title, extract_file, extract_folder

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
    ("p &amp;= 3 et a &lt; b", "p &= 3 et a < b"),                   # leftover HTML entities
    ("R&D et AT&T", "R&D et AT&T"),                                  # not entities: kept
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


def test_unpaginated_keeps_bare_numbers():
    # A year alone on its line in a TXT/HTML file is content, not a page number.
    kept, removed = remove_repeated_lines(["Candide\n\n1759\n\nChapitre premier."], paginated=False)
    assert kept == ["Candide\n\n1759\n\nChapitre premier."] and removed == []


def test_ingest_keeps_year_line_in_txt(tmp_path):
    (tmp_path / "livre.txt").write_text("Candide, ou l'optimisme\n\n1759\n\nIl y avait en Vestphalie un jeune garçon.",
                                        encoding="utf-8")
    chunks, _, report = ingest(tmp_path)
    assert "1759" in chunks["text_raw"].iloc[0]


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


def test_html_drops_hidden_blocks_and_tooltips(tmp_path):
    (tmp_path / "p.html").write_text(
        "<html><body><p>Le mineur et son <span><strong>représentant légal</strong>"
        "<span role='tooltip'>: titleContent</span></span> vont en mairie.</p>"
        "<div hidden><h2>Vos remarques pour améliorer la page</h2></div>"
        "<p aria-hidden='true'>icône</p><p>Fin.</p></body></html>", encoding="utf-8")
    text = extract_file(tmp_path / "p.html")[0]["text"]
    assert text == "Le mineur et son représentant légal vont en mairie.\n\nFin."


def test_txt_in_cp1252_is_decoded():
    text = extract_file(CORPUS / "cv" / "cv_professeure_maths.txt")[0]["text"]
    assert "Expérience" in text and "mathématiques" in text


def test_docx_body_only():
    text = extract_file(CORPUS / "cv" / "cv_auditrice_iso27001.docx")[0]["text"]
    assert "ISO 27001 Lead Auditor" in text
    assert "document confidentiel" not in text  # section header is not body text


def test_csv_with_header_semicolon_cp1252():
    text = extract_file(CORPUS / "divers" / "catalogue_formations.csv")[0]["text"]
    rows = text.split("\n\n")
    assert len(rows) == 4                                   # header row is not a paragraph
    assert rows[0].startswith("Intitulé : Initiation à la cybersécurité ; Domaine : Informatique")
    assert "Adopter les bons réflexes" in rows[0]           # quoted cell with a line break
    assert "Prérequis" not in rows[3]                       # empty cell skipped


def test_csv_numeric_table_keeps_column_names():
    text = extract_file(CORPUS / "divers" / "indicateurs_emploi.csv")[0]["text"]
    assert "Région : Île-de-France ; Année : 2025 ; Taux de chômage (%) : 7.1" in text


def test_csv_without_header_and_single_column(tmp_path):
    (tmp_path / "a.csv").write_text("Paris,75\nLyon,69\nLille,59\n", encoding="utf-8")
    assert extract_file(tmp_path / "a.csv")[0]["text"] == "Paris ; 75\n\nLyon ; 69\n\nLille ; 59"
    (tmp_path / "b.csv").write_text("une phrase\nune autre phrase\n", encoding="utf-8")
    assert extract_file(tmp_path / "b.csv")[0]["text"] == "une phrase\n\nune autre phrase"
    (tmp_path / "c.csv").write_text("\n\n", encoding="utf-8")
    with pytest.raises(SkipFile):
        extract_file(tmp_path / "c.csv")


@pytest.mark.parametrize("title, expected", [
    ("Microsoft Word - cv_final.docx", "mon fichier"),
    ("C:\\Users\\x\\rapport.pdf", "mon fichier"),
    ("untitled", "mon fichier"),
    ("", "mon fichier"),
    ("Développeur backend Python (H/F)", "Développeur backend Python (H/F)"),  # "/" is not a path
    ("  Rapport   annuel ", "Rapport annuel"),
])
def test_clean_title(title, expected):
    assert clean_title(title, Path("mon_fichier.pdf")) == expected


def test_txt_title_from_first_line_only_when_it_stands_alone():
    assert extract_file(CORPUS / "divers" / "resume_vegetalisation_ecoles.txt")[0]["title"] \
        .startswith("Effet de la végétalisation")
    assert extract_file(CORPUS / "cv" / "cv_infirmiere.txt")[0]["title"] == "cv infirmiere"


def test_pdf_junk_metadata_title_falls_back_to_file_name():
    pages = extract_file(CORPUS / "divers" / "decret_horaires_dechetteries.pdf")
    assert pages[0]["title"] == "decret horaires dechetteries"


def test_docx_table_rows_kept_in_order():
    text = extract_file(CORPUS / "divers" / "procedure_carte_identite.docx")[0]["text"]
    assert "Perte ou vol | Déclaration de perte ou de vol" in text
    assert text.index("Étapes") < text.index("Situation | Pièces à fournir") \
        < text.index("Le délai de fabrication")


def test_folder_metadata_and_collection(tmp_path):
    folder = tmp_path / "alloprof" / "math"
    folder.mkdir(parents=True)
    (folder / "p1.txt").write_text("Une fraction est un nombre.", encoding="utf-8")
    (folder / "p2.txt").write_text("Le théorème de Pythagore.", encoding="utf-8")
    (folder / ".metadata.json").write_text(
        '{"p1.txt": {"title": "Les fractions", "subject": "math"}}', encoding="utf-8")
    (tmp_path / "racine.txt").write_text("Un fichier à la racine.", encoding="utf-8")
    pages, skipped = extract_folder(tmp_path)
    by_id = {p["doc_id"]: p for p in pages}
    assert skipped == []                                       # .metadata.json is not indexed
    assert by_id["alloprof/math/p1.txt"]["title"] == "Les fractions"
    assert by_id["alloprof/math/p1.txt"]["meta"] == {"subject": "math"}
    assert by_id["alloprof/math/p2.txt"]["meta"] == {}        # file not listed: no metadata
    assert by_id["alloprof/math/p1.txt"]["collection"] == "alloprof"
    assert by_id["racine.txt"]["collection"] == ""


def test_title_prefix_in_index_text_only():
    pages = [{"doc_id": "d", "title": "Rapport", "page": 1, "text": _sentence(1)}]
    on = chunk_document(pages, title_prefix=True)[0]
    off = chunk_document(pages, title_prefix=False)[0]
    assert on["text_index"] == f"Rapport\n{_sentence(1)}" and on["text_raw"] == _sentence(1)
    assert off["text_index"] == off["text_raw"] == _sentence(1)


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
    assert report["n_documents"] == 22
    assert report["per_collection"] == {"cv": 9, "divers": 7, "offres": 6}
    assert [p for p, _ in skipped] == ["cv/cv_scanne_agent_accueil.pdf"]
    text = "\n".join(chunks["text_raw"])
    for noise in ["Page 1 / 3", "Offre n° 2026-114", "thomas.fictif@example.org", "analytics"]:
        assert noise not in text
    for joined in ["traitement", "configuration", "management"]:
        assert joined in text
    assert (chunks["n_words"] > 0).all()
    assert chunks.groupby("doc_id").size()["offres/offre_rssi_ministere.pdf"] == 2
    assert set(chunks.columns) >= {"collection", "format", "text_index", "meta"}


def test_long_report_chunks():
    chunks, _, report = ingest(CORPUS)
    rep = chunks[chunks.doc_id == "divers/rapport_activite_2025.pdf"]
    assert len(rep) >= 4
    assert list(rep.page) == sorted(rep.page) and rep.page.iloc[-1] >= 8
    assert len(report["removed_headers_footers"]["divers/rapport_activite_2025.pdf"]) == 22
    text = "\n".join(rep.text_raw)
    assert "Agence Fictive de l'Eau et de l'Environnement - Rapport" not in text
    # Section 6 continues on the next page without a heading: both parts must be kept.
    assert "Énergie et climat" in text and "îlots de fraîcheur" in text
