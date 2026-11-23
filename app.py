"""Streamlit interface (Phase 2+). Run: streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="Moteur de recherche documentaire", layout="wide")
st.title("Moteur de recherche documentaire")

with st.sidebar:
    st.subheader("Mode de recherche")
    mode = st.radio("Mode", ["TF-IDF", "BM25", "Sémantique", "Hybride", "Word2vec (moyenne)"],
                    index=3, label_visibility="collapsed")
    alpha = st.slider("Poids sémantique α", 0.0, 1.0, 0.5, 0.05, disabled=(mode != "Hybride"))
    st.subheader("Options")
    typo = st.checkbox("Corriger les fautes", value=True)
    wordnet = st.checkbox("Expansion WordNet (optionnel)", value=False)
    rerank = st.checkbox("Reranker (plus lent)", value=False)

query = st.text_input("Recherche", placeholder="candidat pour un poste de développeur")
if query:
    st.info("Le moteur n'est pas encore branché (phase 0 : squelette).")
