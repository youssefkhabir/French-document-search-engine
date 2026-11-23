# Phase 0 — Technical review

**Project:** Moteur de recherche documentaire intelligent (French semantic search, TALN course)
**Phase:** 0 · Setup and de-risking
**Date:** 23 November 2026
**Commit:** `0f75477` on `main`
**Status:** done. All required environment checks pass offline.

---

## 1. Objectives

Phase 0 was added in front of the original five-phase plan. It answers the questions that
could block the project later, before any feature code is written:

| Objective | Result |
|---|---|
| Put the skipped Chapter 3 content (WordNet, word2vec) back into the plan | Done, as optional components with their own ablations |
| Fix module boundaries so the three tracks can work in parallel | Done: every module has a stub with its signature and contract |
| Get a reproducible environment that runs **offline** | Done: `check_env.py` passes with `HF_HUB_OFFLINE=1` |
| Answer the plan's first risk: is BGE-M3 too slow on our laptops? | Done: measured on GPU and CPU, decision taken |
| Avoid downloading models that are already on the machine | Done: about 6.5 GB of models and 2.5 GB of torch reused |

---

## 2. Scope change: Chapter 3 components

The original plan left out WordNet and word2vec on purpose, as optional extras "if you
finish early". Chapter 3 of the course (*Représentation vectorielle des mots*) is built
around exactly these topics: word senses and relations, WordNet with Wu–Palmer similarity,
sparse vectors (bag of words, term–context matrix, TF-IDF), then dense vectors (word2vec).

**Decision:** both come back as **optional, measured** components. The core system (S1–S5)
does not depend on them. They are implemented in Phase 4 once the hybrid system (S4)
works. Full design: [plan_addendum.md](plan_addendum.md).

| Component | Module | Course concept | Measured by |
|---|---|---|---|
| WordNet query expansion | `search/wordnet_expand.py` | Semantic database, synonymy, hyponymy | BM25 vs BM25 + WordNet, nDCG@10 per query type |
| Wu–Palmer illustration | `search/wordnet_expand.py` | Taxonomy-based similarity | Report figure next to cosine similarity |
| Averaged word2vec (S6) | `search/word2vec.py` | Static dense vectors | Extra row in the main results table |

With S6, the results table follows the course's own order: sparse (TF-IDF, BM25) → static
dense (word2vec) → contextual dense (BGE-M3) → combination (hybrid).

**Early evidence of the WordNet risk:** the offline check queries French WordNet for
*voiture* and gets `auto, automobile, carrosse, entraîneur, voiture, Voiture_de_chemin_de_fer`.
*Entraîneur* (a coach, from the "vehicle" vs "trainer" senses) shows the polysemy noise the
addendum predicted. Expansion may therefore hurt precision on type-A queries, and the
ablation will show it.

---

## 3. Repository layout

```
app.py                     Streamlit page (placeholder: sidebar controls only)
search/
  config.py                all paths, model names and tunable parameters
  extract.py               Phase 1 · PDF / DOCX / TXT / HTML → pages
  clean.py                 Phase 1 · regex cleaning
  chunk.py                 Phase 1 · sentence-aware chunker
  lexical.py               Phase 2 · lexical_prep, TF-IDF, BM25
  semantic.py              Phase 3 · embeddings, cosine search
  spelling.py              Phase 4 · Levenshtein typo correction
  fusion.py                Phase 4 · weighted fusion and RRF
  rerank.py                Phase 4 · cross-encoder (optional)
  wordnet_expand.py        Phase 4 stretch · WordNet expansion (optional)
  word2vec.py              Phase 4 stretch · averaged word2vec, S6 (optional)
  engine.py                build_index() and search(), the two entry points
eval/
  queries.json             query template (qid, type A–D, split dev/test, text)
  qrels.csv                judgments header (qid, doc_id, grade)
  metrics.py               Recall@k, MRR@k, nDCG@k stubs
  run_eval.py              evaluation runner stub
scripts/
  download_models.py       one-time model and data download
  check_env.py             offline environment check
  bench_embeddings.py      embedding speed benchmark
docs/
  plan_addendum.md         optional Chapter 3 components
  phase0_review.md         this document
data/corpus/               documents (git-ignored)
index/                     generated index (git-ignored)
models/hub/                Hugging Face cache (git-ignored)
```

### Design choices in the skeleton

- **Stubs with contracts, not empty files.** Each function has its final signature, return
  type and a docstring that states the rules, for example "absent chunk → 0" in fusion,
  "guard max == min" in min-max, and "skip tokens with capitals or digits" in typo
  correction. Each stub raises `NotImplementedError("Phase N")`, so a missing piece fails
  loudly. The three tracks can code against these interfaces without waiting on each other.
- **One chunk record format**, defined in `chunk.py`:
  `{chunk_id, doc_id, title, page, text_raw}`, plus `text_lex` added by `lexical_prep`.
- **One search result format**, defined in `engine.search()`. It includes `rank_bm25` and
  `rank_semantic` so the interface can show both ranks, as in the plan's wireframe.
- **`search()` already has the optional flags** (`typo_fix`, `rerank`, `wordnet`), and
  `MODES` includes `word2vec`. Phase 4 adds behaviour without changing the signature, so
  `app.py` and `run_eval.py` do not change.
- **`lexical_prep` takes `lemmatize` and `remove_stopwords` options.** The two planned
  ablations (raw words vs lemmas, with vs without stopwords) then reuse the production
  function instead of a copy. This keeps the plan's rule that documents and queries share
  exactly one preprocessing function.
- **All tunables live in `config.py`**: chunk sizes, top-k values, α, typo distances, the
  WordNet and word2vec parameters. Switching from BGE-M3 to e5-small is one line
  (`EMBED_MODEL`), and the e5 `query:` / `passage:` prefixes are looked up from
  `EMBED_PREFIXES`, so they cannot be forgotten.

---

## 4. Environment

### Stack

| Item | Version |
|---|---|
| Python | 3.12.12 (uv-managed venv in `.venv/`) |
| torch | 2.11.0 + CUDA 12.8 |
| sentence-transformers | 6.3.0 |
| transformers | 5.24.0 |
| bm25s | 0.3.16 |
| spaCy / `fr_core_news_sm` | 3.8.18 / 3.8.0 |
| scikit-learn | 1.9.1 |
| rapidfuzz | 3.14.6 |
| PyMuPDF | 2.1 |
| streamlit | 1.67.0 |
| nltk / gensim | 3.10.4 / 4.4.0 |

Two dependency files:
- `requirements.txt`: unpinned top-level packages, for teammates on other OS/hardware.
- `requirements-lock.txt`: the 106 exact versions tested on the dev machine. The torch
  line was changed from `2.11.0+cu128` to `2.11.0`. The CUDA tag does not exist on PyPI,
  and a CPU-only teammate could not install the file with it.

### Model storage

`config.py` sets `HF_HUB_CACHE` to `models/hub/` inside the project (`os.environ.setdefault`,
so an explicit env var still wins). This keeps models off the system drive and makes
"what the project needs" visible in one folder. Because the variable is set in
`config.py`, every script must import `search.config` **before** any Hugging Face library.
All three scripts do.

| Model | Role | Size on disk | How obtained |
|---|---|---:|---|
| `BAAI/bge-m3` | Main embedding model | 4.3 GB | Reused (junction) |
| `BAAI/bge-reranker-v2-m3` | Optional cross-encoder | 2.2 GB | Reused (junction) |
| `intfloat/multilingual-e5-small` | Fallback embedding model | 0.9 GB | Downloaded |
| `fr_core_news_sm` | Tokenization, lemmas, stopwords | 16 MB | Downloaded (pip wheel) |
| NLTK `wordnet` + `omw-2.0` | French WordNet (optional) | small | Downloaded |

### Reusing existing downloads

BGE-M3 and the reranker were already fully downloaded for another local project. Instead
of downloading 6.5 GB again:

1. **Models:** two NTFS directory junctions in `models/hub/` point to the existing model
   folders. There is no copy, and no admin rights are needed. Hugging Face's internal
   links (`snapshots/ → ../../blobs/`) are relative, so they resolve through the junction.
   New models such as e5-small are written to this project's own folder, not the other one.
   After the download script ran, a check confirmed that **no new file was written** into
   the linked folders.
2. **Packages:** `uv` was pointed at the other project's package cache (`UV_CACHE_DIR`).
   `torch 2.11.0+cu128` was linked from that cache instead of being downloaded (~2.5 GB).
   Only about 100 MB of new packages was fetched.

The README documents the junction command for teammates in the same situation.

---

## 5. Scripts

### `download_models.py`
Downloads the spaCy model, NLTK WordNet data, BGE-M3 and e5-small. `--reranker` adds the
cross-encoder and `--light` skips BGE-M3 on weak laptops. Uses `snapshot_download` with
`ignore_patterns` to skip ONNX, OpenVINO, TF and Flax weights that sentence-transformers
does not use. It is idempotent: files already in the cache are only checked, not fetched
again.

### `check_env.py`
Forces `HF_HUB_OFFLINE=1` and checks:
- every library imports
- torch and its device
- spaCy loads and lemmatizes
- French WordNet answers
- each model is in the local cache (the reranker is reported but optional)

Exits 1 if anything required fails, so it can serve as the pre-presentation check.

### `bench_embeddings.py`
Times the embedding models on 100 synthetic French chunks of ~220 words. The chunks are
built from 15 varied sentences across the plan's candidate domains (CVs, job offers,
university admin, news). It reports:
- load time, ms per chunk, and indexing time extrapolated to `--corpus-chunks` (default 1,500)
- query latency, RAM, and VRAM on GPU
- max token length, with a warning when e5 chunks exceed the 480-token cap

Measurement hygiene: one warm-up batch, `torch.cuda.synchronize()` around the timed region,
and VRAM peak reset before encoding.

---

## 6. Benchmark results

Machine: Intel i7-10750H (6 cores), 16 GB RAM, NVIDIA RTX 2060 6 GB. Batch size 16.

| Model | Device | Load | Per chunk | 1,500 chunks | Query | Memory |
|---|---|---:|---:|---:|---:|---|
| BGE-M3 (1024 d) | GPU | 9.4 s | 58 ms | **~1.5 min** | 48 ms | 2.6 GB VRAM |
| BGE-M3 (1024 d) | CPU | 2.4 s | 1,025 ms | **~26 min** | 134 ms | 2.7 GB RAM |
| e5-small (384 d) | GPU | 3.2 s | 8 ms | ~0.2 min | 15 ms | 0.6 GB VRAM |
| e5-small (384 d) | CPU | 4.0 s | 102 ms | ~2.6 min | 24 ms | 1.7 GB RAM |

No e5 chunk exceeded the 480-token cap at ~220 words.

### Decision

**BGE-M3 stays the main model.**
- On the GPU machine, a full rebuild of the planned corpus takes about 1.5 minutes. That
  fits the plan's choice to rebuild everything when documents change.
- On a CPU-only laptop, indexing takes about 26 minutes. That is too slow for repeated
  development rebuilds, but query embedding stays at 134 ms, so search remains usable.
- **Workflow:** build the final index on the GPU machine and share the `index/` folder.
  CPU teammates develop with e5-small (one line in `config.py`), then switch back for the
  final run.
- The "upload a document in the interface" feature only embeds that document's chunks, so
  it stays fast even on CPU. A rebuild of BM25 and TF-IDF is cheap.

### Caveats
- The figures are **extrapolated** from 100 chunks, and synthetic chunks repeat sentences.
  Real chunks will vary in length, so the real indexing time should be re-measured in
  Phase 3.
- The reranker was not benchmarked yet. Its cost is per query (20 pairs), so it belongs
  in the Phase 4 measurement.

---

## 7. Issues found and fixed

| Issue | Cause | Fix |
|---|---|---|
| French WordNet failed in `check_env.py` with `Resource 'omw-2.0' not found` | NLTK 3.10 reads Open Multilingual WordNet from `omw-2.0`, not `omw-1.4` | `download_models.py` now fetches `wordnet`, `omw-2.0` and `omw-1.4` (for older NLTK); addendum updated |
| `python -m spacy download` would fail in a uv venv | uv venvs have no `pip`, and spaCy's downloader calls pip | Installed `pip` into the venv. Teammates using `python -m venv` already have it |
| Lock file not installable on CPU-only machines | `torch==2.11.0+cu128` is not on PyPI | Pinned `torch==2.11.0`; the README tells GPU users to install the CUDA build first |
| Unreadable VRAM print line in the benchmark | Formatting written with `chr()` codes | Replaced with a plain `%`-formatted string |
| BGE-M3 download step took ~3 minutes with nothing to fetch | `snapshot_download` checks every remote file, and one request was slow | No change needed. Confirmed that no files were downloaded or written |

---

## 8. Verification

| Check | Result |
|---|---|
| `python -m compileall app.py search eval scripts` | OK |
| `import search.engine`; `HF_HUB_CACHE` points to `models/hub` | OK |
| `check_env.py` with `HF_HUB_OFFLINE=1` | All required checks pass |
| spaCy lemmas on *Les développeurs ont développé des applications.* | `le, développeur, avoir, développer, de, application` |
| torch device | `cuda (NVIDIA GeForce RTX 2060)` |
| Repo content before push | 29 files, 78 KB; scanned for local paths and personal data, none found |

Not verified yet:
- the Streamlit page has not been launched; it is a placeholder
- the setup has not been tested on a teammate's machine or on a CPU-only install from
  `requirements-lock.txt`
- the full offline run with Wi-Fi physically off (it was simulated with `HF_HUB_OFFLINE=1`)

---

## 9. Git and repository hygiene

- Public repository: `youssefkhabir/French-document-search-engine`, branch `main`.
- Committed: source code, requirements files, README, docs.
- Git-ignored and kept local:
  - both PDFs, including the course slides
  - `.venv/`, `models/`, `nltk_data/`
  - `index/*` and `data/corpus/*` (the corpus may hold personal data such as CVs)
  - `.env` and Streamlit secrets
- `data/corpus/` and `index/` are kept in the tree with `.gitkeep`.
- Line endings: Git warns that LF will become CRLF on Windows checkout. This is harmless
  for Python. A `.gitattributes` with `* text=auto eol=lf` would make it consistent if the
  team mixes OSes.

---

## 10. Risks carried into the next phases

| Risk | Status after Phase 0 | Next action |
|---|---|---|
| BGE-M3 too slow on laptops | Measured: fine on GPU, slow on CPU for indexing only | Share `index/`; e5-small for CPU development |
| Models unavailable on presentation day | Offline check passes | Repeat with Wi-Fi off on the demo laptop |
| Corpus not chosen | **Open** | Pick the domain (CVs/offers, university admin, or news) before Phase 1 ends; the query writer starts from it |
| WordNet expansion adds noise | Confirmed by the *voiture → entraîneur* example | Cap expansions, repeat original terms, and measure per query type |
| word2vec trained on a small corpus is weak | Expected | Optional pretrained vectors through `W2V_PRETRAINED_PATH` |
| Teammates' setup differs | Not tested | Each teammate runs `check_env.py` and `bench_embeddings.py` and posts the summary |

---

## 11. Handoff to Phase 1

Phase 1 implements `extract.py`, `clean.py` and `chunk.py`, then the first half of
`engine.build_index()` to produce `index/chunks.parquet`.

Inputs needed:
1. **A corpus domain** and a first batch of documents in `data/corpus/`, in mixed formats,
   including at least one scanned PDF to test the skip path.
2. Agreement on the chunk record (section 3) between Track 1 (ingestion) and Track 2
   (retrieval).

Done when a folder of files becomes a table of clean chunks that can be read. Phase 1 also
produces the regex before/after examples for the report.
