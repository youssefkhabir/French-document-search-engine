"""Streamlit interface. Run: streamlit run app.py

Phases 2-4: TF-IDF, BM25, semantic (BGE-M3) and hybrid modes, reranker, typo correction,
collection and file-type filters, snippets.
The page calls engine.search(), the same function the evaluation uses.
"""
import html
from pathlib import Path

import streamlit as st

from search import config
from search import semantic
from search.engine import load_index, search
from search.lexical import lexical_prep

st.set_page_config(page_title="Moteur de recherche documentaire", layout="wide")

MODE_LABELS = {"TF-IDF": "tfidf", "BM25": "bm25", "Sémantique": "semantic",
               "Hybride": "hybrid", "Word2vec (moyenne)": "word2vec"}
MODE_PHASE = {"word2vec": 4}
FUSION_LABELS = {"Pondérée (α)": "weighted", "RRF (rangs)": "rrf"}
FORMAT_LABELS = {"pdf": "PDF", "docx": "Word", "txt": "Texte", "html": "HTML", "csv": "CSV"}
SUBJECT_LABELS = {"chemistry": "Chimie", "contemporary_world": "Monde contemporain",
                  "english": "Anglais", "financial_ed": "Éducation financière",
                  "french": "Français", "geography": "Géographie", "history": "Histoire",
                  "math": "Mathématiques", "other": "Autre", "physics": "Physique",
                  "science": "Sciences"}


def index_dirs() -> list[Path]:
    """Folders of the project that hold a built index (index/, index_test/, …)."""
    found = [p for p in sorted(config.ROOT.iterdir()) if (p / "meta.json").is_file()]
    default = config.INDEX_DIR
    return sorted(found, key=lambda p: p != default)


@st.cache_resource(show_spinner="Chargement de l'index…")
def get_index(path: str, stamp: float) -> dict:
    """Cached per index folder; `stamp` (meta.json mtime) reloads it after a rebuild."""
    lexical_prep("préchauffage")  # load spaCy now, not on the first query
    idx = load_index(Path(path))
    if idx["embeddings"] is not None:  # load the embedding model now too
        semantic.load_model(idx["embeddings"][1]["model"])
    return idx


def highlight(snippet: str, spans: list[tuple[int, int]]) -> str:
    """Escape the snippet and wrap matched words in <mark>."""
    out, pos = [], 0
    for a, b in spans:
        out += [html.escape(snippet[pos:a]), "<mark>", html.escape(snippet[a:b]), "</mark>"]
        pos = b
    out.append(html.escape(snippet[pos:]))
    # "$" would start LaTeX math in st.markdown: show it as a plain character.
    return "".join(out).replace("$", "&#36;")


def collection_label(c: str) -> str:
    return {"": "(racine)", "alloprof": "Alloprof", "divers": "Corpus équipe"}.get(c, c)


st.title("Moteur de recherche documentaire")

dirs = index_dirs()
if not dirs:
    st.error("Aucun index trouvé. Lancez d'abord : `python scripts/build_index.py`")
    st.stop()

with st.sidebar:
    st.subheader("Index")
    index_dir = st.selectbox("Dossier d'index", dirs, format_func=lambda p: p.name,
                             label_visibility="collapsed")
    try:
        idx = get_index(str(index_dir), (index_dir / "meta.json").stat().st_mtime)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    chunks = idx["chunks"]

    st.subheader("Mode de recherche")
    default_mode = 3 if "hybrid" in idx["modes"] else 1
    mode_label = st.radio("Mode", list(MODE_LABELS), index=default_mode,
                          label_visibility="collapsed")
    mode = MODE_LABELS[mode_label]
    fusion_label = st.radio("Fusion", list(FUSION_LABELS), horizontal=True,
                            disabled=(mode != "hybrid"))
    fusion_method = FUSION_LABELS[fusion_label]
    alpha = st.slider("Poids sémantique α", 0.0, 1.0, config.DEFAULT_ALPHA, 0.05,
                      disabled=(mode != "hybrid" or fusion_method != "weighted"),
                      help="0 = BM25 seul, 1 = sémantique seul")

    st.subheader("Filtres")
    collections = sorted(chunks["collection"].unique())
    chosen_collections = st.multiselect("Collection", collections, format_func=collection_label,
                                        placeholder="Toutes")
    formats = sorted(chunks["format"].unique())
    chosen_formats = st.multiselect("Type de fichier", formats,
                                    format_func=lambda f: FORMAT_LABELS.get(f, f),
                                    placeholder="Tous")
    top_k = st.slider("Nombre de résultats", 5, 30, config.RESULTS_TOP_K, 5)

    st.subheader("Options")
    typo_fix = st.checkbox("Corriger les fautes", value=True,
                           help="Accents manquants, fautes de frappe, requête en majuscules")
    st.checkbox("Expansion WordNet (optionnel)", value=False, disabled=True)
    use_rerank = st.checkbox("Reranker (plus lent)", value=False,
                             help=f"Réordonne les {config.RERANK_TOP_K} premiers documents "
                                  "avec un cross-encoder")

    with st.expander("Corpus indexé"):
        meta = idx["meta"]
        docs = chunks.groupby("collection")["doc_id"].nunique()
        n_chunks = chunks.groupby("collection").size()
        st.dataframe({"Collection": [collection_label(c) for c in docs.index],
                      "Documents": docs.to_list(), "Passages": n_chunks.to_list()},
                     hide_index=True)
        emb = meta.get("embeddings")
        emb_text = (f"plongements {emb['model'].split('/')[-1]} ({emb['dim']} d)" if emb
                    else "sans plongements")
        st.caption(f"{meta['n_documents']} documents · {meta['n_chunks']} passages · "
                   f"{meta.get('lexical', {}).get('vocabulary', '?')} termes · {emb_text} · "
                   f"construit le {meta['built_at'].replace('T', ' à ')}")

query = st.text_input("Recherche", placeholder="ex. : comment calculer l'aire d'un triangle")

if query.strip():
    if mode not in idx["modes"]:
        if mode in MODE_PHASE:
            st.info(f"Le mode « {mode_label} » arrive en phase {MODE_PHASE[mode]}.")
        else:
            st.info(f"Cet index n'a pas de plongements : le mode « {mode_label} » n'est pas "
                    "disponible. Reconstruisez-le sans `--no-embed`.")
        st.stop()

    spinner = ("Chargement du reranker (une fois) et réordonnancement…" if use_rerank
               else "Recherche…")
    with st.spinner(spinner):
        r = search(query, mode=mode, alpha=alpha, top_k=top_k, collections=chosen_collections,
                   formats=chosen_formats, index_dir=index_dir, typo_fix=typo_fix,
                   rerank=use_rerank, fusion_method=fusion_method)
    total_ms = sum(r["timings"].values())
    if r["corrections"]:
        st.markdown(f"Recherche corrigée : **{html.escape(r['corrected_query'])}**")
        st.caption(" · ".join("requête en minuscules" if a == r["query"] else f"{a} → {b}"
                              for a, b in r["corrections"]))
    matched = {"semantic": "passages comparés",
               "hybrid": "passages candidats (BM25 + sémantique)"}.get(mode, "passages correspondants")
    st.caption(f"{len(r['results'])} documents · {r['n_chunks_matched']} {matched} · "
               f"{total_ms:.0f} ms · forme lexicale : `{r['query_lex'] or '∅'}`")

    if not r["results"]:
        st.warning("Aucun document ne contient les termes de la requête."
                   + (" Essayez sans filtre." if chosen_collections or chosen_formats else ""))

    for rank, res in enumerate(r["results"], 1):
        with st.container(border=True):
            st.markdown(f"**{rank}. {html.escape(res['title'])}**")
            tags = [collection_label(res["collection"]), FORMAT_LABELS.get(res["format"], res["format"])]
            if res["format"] == "pdf":
                tags.append(f"page {res['page']}")
            subject = res["meta"].get("subject")
            if subject:
                tags.append(SUBJECT_LABELS.get(subject, subject))
            if mode == "hybrid":
                ranks = [f"BM25 #{res['rank_bm25']}" if res["rank_bm25"] else "BM25 —",
                         f"sémantique #{res['rank_semantic']}" if res["rank_semantic"] else "sémantique —"]
                tags.append(" / ".join(ranks))
            score = (f"reranker {res['rerank_score']:.3f}" if res["rerank_score"] is not None
                     else f"score {res['score']:.3f}")
            st.caption(" · ".join(tags) + f" · {score} · `{res['doc_id']}`")
            st.markdown(highlight(res["snippet"], res["highlights"]), unsafe_allow_html=True)
