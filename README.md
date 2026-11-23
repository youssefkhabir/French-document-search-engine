# Moteur de recherche documentaire intelligent

A French document search engine (TALN project): it finds relevant documents even when
the query and the document use different words. It runs hybrid retrieval: BM25 on
lemmatized text plus BGE-M3 embeddings, fused with a tuned weight, with an optional
reranker. Everything runs locally and offline in one Streamlit page.

- Full plan: `Plan_Moteur_Recherche_Documentaire.pdf`
- Optional course components (WordNet expansion, word2vec baseline): [docs/plan_addendum.md](docs/plan_addendum.md)

## Setup

Python 3.12 recommended.

```powershell
python -m venv .venv
.venv\Scripts\activate
# GPU (NVIDIA) only: install the CUDA build of torch first
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt          # or requirements-lock.txt for the exact tested versions
python scripts/download_models.py        # add --light on a weak laptop, --reranker for the cross-encoder
```

Models are stored in `models/hub/` inside the project (see `search/config.py`), not in
your user profile.

### Reuse models already on your machine

If BGE-M3 or the reranker is already downloaded for another project, link it in
instead of downloading 6 GB again. No copy is made and no admin rights are needed:

```powershell
New-Item -ItemType Junction -Path models\hub\models--BAAI--bge-m3 -Target "<other project>\models\hub\models--BAAI--bge-m3"
```

## Check that everything works offline

```powershell
$env:HF_HUB_OFFLINE = "1"
python scripts/check_env.py
```

Run it once with Wi-Fi off before presentation day.

## Phase 0 benchmark

```powershell
python scripts/bench_embeddings.py               # auto device
python scripts/bench_embeddings.py --device cpu  # worst case
```

Results on the dev machine (i7-10750H, RTX 2060 6 GB), 100 chunks of ~220 words,
extrapolated to 1,500 chunks:

| Model | Device | Index 1,500 chunks | Query | Memory |
|---|---|---:|---:|---|
| BGE-M3 (1024 d) | GPU | ~1.5 min | 48 ms | 2.6 GB VRAM |
| BGE-M3 (1024 d) | CPU | ~26 min | 134 ms | 2.7 GB RAM |
| multilingual-e5-small (384 d) | GPU | ~0.2 min | 15 ms | 0.6 GB VRAM |
| multilingual-e5-small (384 d) | CPU | ~2.6 min | 24 ms | 1.7 GB RAM |

**Decision:** BGE-M3 stays the main model. Build the final index on the GPU machine and
share the `index/` folder. A CPU laptop only embeds the query (~0.1 s). Teammates can
develop with e5-small (`EMBED_MODEL` in `search/config.py`) and switch back for the
final run.

## Run

```powershell
streamlit run app.py
```

## Layout

```
app.py              Streamlit page
search/             extract, clean, chunk, lexical, semantic, spelling, fusion, rerank, engine
                    + optional wordnet_expand, word2vec (course Chapter 3)
data/corpus/        documents (not committed: may hold personal data)
index/              generated index files
eval/               queries.json, qrels.csv, metrics.py, run_eval.py
scripts/            download_models, check_env, bench_embeddings
docs/               plan addendum
```

## Licences

PyMuPDF is AGPL-licensed, which is fine for this student project.
