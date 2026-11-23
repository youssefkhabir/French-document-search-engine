# Phase 3 — Technical review

**Project:** Moteur de recherche documentaire intelligent (French semantic search, TALN course)
**Phase:** 3 · Semantic branch
**Date:** 23 November 2026
**Status:** done. 70 tests pass (9 new). The semantic mode runs on the full index (Alloprof
+ team corpus) and in the Streamlit page.

---

## 1. What Phase 3 delivers

The plan's "done when" criterion: *the semantic mode answers questions that share no word
with the page (type C), where TF-IDF and BM25 fail.*

```
chunks.text_index ──BGE-M3 (GPU, fp16)──▶ E  (n × 1024, L2-normalized)  ──▶ index/embeddings.npy
                                                                          + index/embeddings.json (model, dim…)

query ──same model──▶ q (1024)  ──▶ scores = E @ q  (cosine) ──▶ filter ──▶ best chunk per document
```

| File | Role | Course technique |
|---|---|---|
| `search/semantic.py` | Load the model, embed passages and queries, cosine search, save/load | **Plongements contextuels (transformers), similarité cosinus** |
| `search/engine.py` | `build_index(embed=…, embed_model=…)`, `semantic` mode in `search()` | — |
| `scripts/build_index.py` | New options `--no-embed`, `--model` | — |
| `app.py` | "Sémantique" mode enabled when the index has embeddings | — |
| `search/config.py` | `EMBED_MAX_TOKENS = 1024`, `EMBED_BATCH = 16` | — |
| `tests/test_phase3.py` | 9 tests | — |

**Index today:** 2,360 documents in two collections (`alloprof`: 2,350, `divers`: 10) →
**10,295 chunks**. Embedding them with BGE-M3 takes **214 s on the RTX 2060**. The full
build takes 6.6 min, with spaCy (Phase 2) still the larger share. `embeddings.npy` is
42 MB (float32).

---

## 2. `semantic.py`

### Loading the model

```python
@lru_cache(maxsize=2)
def load_model(name=None):
    model = SentenceTransformer(name or config.EMBED_MODEL, device=device())
    if device() == "cuda":
        model.half()                                   # fp16
    model.max_seq_length = min(model.max_seq_length, config.EMBED_MAX_TOKENS)
    return model
```

| Choice | Why |
|---|---|
| **`lru_cache`** | The model loads once per process (15 s for BGE-M3). The page and the evaluation reuse it. |
| **fp16 on GPU** | Half the VRAM and faster on the RTX 2060's tensor cores. Cosine rankings are unchanged in practice. On CPU the model stays fp32, since fp16 is slow there. |
| **Token cap = min(model, config)** | BGE-M3 accepts 8,192 tokens, which would waste memory on 6 GB. e5 keeps its own 512 limit. See §5 for how 1,024 was chosen. |
| **`EMBED_BATCH = 16`** | Fits BGE-M3 fp16 with 1,024-token inputs on 6 GB. sentence-transformers sorts inputs by length, so most batches are short. |

### Passages, queries, prefixes

```python
def embed_passages(model, texts, name=None): ...   # "passage: " + text for e5, nothing for BGE-M3
def embed_query(model, query, name=None): ...      # "query: " + text for e5
```

Both use `normalize_embeddings=True`, so each vector has length 1. e5 models were trained
with the `query:` / `passage:` prefixes and lose quality without them; BGE-M3 needs none.
The prefixes come from `config.EMBED_PREFIXES`, keyed by model name.

### Search

```python
scores = E @ q_vec                                 # cosine, since rows and q have norm 1
scores = np.where(mask, scores, -np.inf)           # filters
```

One matrix-vector product over 10,295 × 1,024 floats, plus sorting every score: about **6 ms**. Every chunk gets a score
(cosine similarity has no "no match"), so the filters use `-inf` and not 0: a negative
cosine is still a valid score.

### The index records its model

`embeddings.json`:

```json
{"model": "BAAI/bge-m3", "dim": 1024, "n": 10295, "device": "cuda",
 "max_tokens": 1024, "n_truncated": 51, "load_seconds": 15.3, "seconds": 214.5}
```

At query time `search()` loads **the model named in this file**, not `config.EMBED_MODEL`.
A teammate who set `EMBED_MODEL` to e5 for development can still open the shared BGE-M3
index: the query is embedded with BGE-M3, and the two kinds of vector are never compared.
A test checks this (an e5 index searched while the config says BGE-M3).

`load_index()` also refuses an index whose embeddings don't have one row per chunk, and a
lexical-only rebuild (`--no-embed`) deletes old embedding files so they can't be paired
with the wrong chunks.

---

## 3. Changes in `engine.py` and the page

- `build_index(folder, index_dir, embed=True, embed_model=None, show_progress=False)`
- `load_index()` adds `"embeddings": (E, info) or None` and `"modes"`: the modes this index
  can serve. A lexical-only index offers only TF-IDF and BM25.
- `search(mode="semantic")` embeds the query (timing `embed`), scores every chunk, then
  follows the Phase 2 path: filters, one result per document, snippet.
- `rank_semantic` is filled in semantic mode. `n_chunks_matched` is the number of chunks
  *compared*, and the page says "passages comparés" in this mode.
- The page loads BGE-M3 together with the index and shows the model in the corpus panel
  (*"plongements bge-m3 (1024 d)"*).

**Snippets in semantic mode** still highlight the query's lexical terms. When the page
shares no word with the query (the type-C case), nothing is highlighted and the snippet
shows the start of the best chunk. Choosing the snippet by meaning would need a second
embedding pass per result and is left out.

---

## 4. Results on the 300 Alloprof dev questions

Sanity check, not the Phase 5 evaluation (that one uses the 2,281 test questions, nDCG and
a significance test). Index with both collections.

| Mode | Recall@10 | MRR@10 | Time / query* |
|---|---|---|---|
| S1 · TF-IDF | 0.618 | 0.393 | 129 ms |
| S2 · BM25 | 0.582 | 0.379 | 111 ms |
| **S3 · Semantic (BGE-M3)** | **0.733** | **0.553** | 187 ms |

\* Most of it is the snippets (spaCy on the 10 best chunks): 100–300 ms. The query
embedding takes about 30 ms on the GPU, and the scoring about 6 ms.

- **+11.5 points of recall and +16 points of MRR** over the best lexical mode. The first
  relevant page is also much higher in the list.
- Relevant page in the top 10: **only semantic: 65 questions**, **only BM25: 18**,
  **neither: 52**. The two branches fail on different questions, which is exactly what
  Phase 4's hybrid fusion is meant to use.
- TF-IDF and BM25 are slightly lower than in Phase 2 (0.620 → 0.618). The `divers`
  collection changed the document frequencies. Expected.

### Where the semantic mode wins (relevant page 1st, BM25 has none in the top 10)

| Question (student's spelling kept) | Semantic, 1st | BM25, 1st |
|---|---|---|
| *Bonjour, j'ai un examen demain sur La **probaliter** des événement…* | Probabilités | Comment bien s'organiser en prévision des examens |
| *En générale, C'est quoi la fonction **rationelle**?* | La fonction rationnelle | La somme de fonctions |
| *J'ai oublié ce qu'était un Complément du nom ?…* | La fonction complément du nom ou du pronom | Les sources de biais |
| *Quelle date a commencé la première guerre mondiale?* | La Première Guerre mondiale | Les Années folles et la Grande Dépression… |

The first two are **typos**: the misspelled word isn't in the lexical vocabulary, but the
transformer's sub-word tokens still carry the meaning. The last two are chatty questions
where BM25 is pulled away by common words (*examen*, *date*, *oublié*).

### Where BM25 wins

| Question | BM25 finds (rank) | Semantic, 1st |
|---|---|---|
| ***BONJOUR J'AURAIS BESOIN DE SAVOIR LE PASSÉ COMPOSÉ ET LE PASSÉ SIMPLE MERCI 🤩*** | Le passé simple de l'indicatif (1) | Trucs pour conjuguer |
| *c'est quoi un **antécédent*** | Le pronom (6) | Le passé antérieur de l'indicatif |
| *c'est quoi une **constante** ?* | Algèbre - Expressions algébriques (10) | La constante d'acidité (Ka) |

- **All-capitals text** hurts the transformer: capital letters split into different
  sub-word tokens than the usual lowercase words. BM25 lowercases everything. Lowercasing a
  query that is mostly capitals would be a cheap fix for Phase 4.
- A **polysemous term** (*constante*, *antécédent*) is matched to its most "typical"
  meaning by the semantic model (chemistry, a verb tense), while the exact word in BM25
  reaches the right page.

---

## 5. Finding: LaTeX-heavy chunks were truncated

The first build used a 512-token cap. It reported **717 chunks over the cap (7%)**, 707 of
them from Alloprof. Measured with the BGE-M3 tokenizer:

| | Tokens |
|---|---|
| All chunks, median / 95th / 99th percentile | 367 / 568 / 895 |
| Over-512 chunks, median / max | 640 / 2,201 |
| Share of text cut in an over-512 chunk, median | 20% |

The cause is **LaTeX** in math, chemistry and physics pages
(`|N_{2(g)} + 3 H_{2(g)} \rightarrow 2 NH_{3(g)}|`): symbols and braces split into many
sub-word tokens, so a 244-word chunk becomes 640 tokens.

Raising `EMBED_MAX_TOKENS` to **1,024** leaves **51 chunks** truncated (0.5%):

| Cap | Truncated | Recall@10 | MRR@10 | Embedding time |
|---|---|---|---|---|
| 512 | 717 | 0.733 | 0.552 | 187 s |
| **1,024** | **51** | **0.733** | **0.553** | 214 s |

**No measurable difference.** The first 80% of a chunk already carries its topic, and the
chunks overlap by 40 words. 1,024 is kept because it loses no text for 27 s more at build
time. That's a small result worth a line in the report: *truncation at 512 tokens did not
hurt on this corpus*.

---

## 6. Team corpus

`data/corpus/divers/` now holds the 10 public documents used to test Phase 1 (2 PDF, 3
DOCX, 2 HTML, 1 TXT, 2 CSV, from data.gouv.fr, Service-Public, Wikipedia, ACL Anthology
and Project Gutenberg). It's the first batch of the team corpus. It makes the collection
filter meaningful in the page (*"Corpus équipe"*).

- **The festival CSV is 2,895 of the collection's 3,221 chunks.** It doesn't affect
  Alloprof searches, but when the team adds its documents, replacing it with a smaller
  table keeps `divers` balanced.
- The team's own 50–100 documents and the ~40 judged queries (`eval/queries.json`,
  `eval/qrels.csv`, still templates) remain to be done.

---

## 7. Tests

`tests/test_phase3.py`, 9 tests, with **multilingual-e5-small** (384 d, about 20 s for the
module including the model load). They also run on a CPU-only machine.

| Test | What it checks |
|---|---|
| `test_search_semantic_ranks_by_cosine` | Hand-made vectors: order, every row scored, top k, filter mask, empty mask |
| `test_vectors_are_normalized` | Shape, float32, norm 1; *"un animal qui dort"* is closer to *"Le chat dort."* than to a car |
| `test_e5_prefixes_are_applied` | The same text as query and as passage gives different vectors |
| `test_model_is_loaded_once` | `load_model` returns the cached object |
| `test_embedding_files` | `embeddings.npy` has one row per chunk; `embeddings.json` names the model; the index offers the semantic mode |
| `test_semantic_finds_a_document_sharing_no_word` | *"quelqu'un qui soigne les malades la nuit"*: **BM25 returns nothing, semantic puts the nurse's CV first** |
| `test_semantic_uses_the_index_model_not_the_config` | An e5 index works while the config says BGE-M3 |
| `test_semantic_filters_and_one_result_per_document` | PDF filter, every PDF chunk scored, one entry per document, scores sorted |
| `test_lexical_only_rebuild_removes_stale_embeddings` | `embed=False` deletes old embeddings; the index then offers TF-IDF and BM25 only |

The Phase 2 tests now build lexical-only indexes (`embed=False`), so they don't load a model.

---

## 8. Known limitations

| # | Limitation | Effect | Where it goes |
|---|---|---|---|
| 1 | **All-capitals queries** embed badly. | Misses like *"BONJOUR… PASSÉ COMPOSÉ…"* | Phase 4: lowercase a query that is mostly capitals, before embedding |
| 2 | **Polysemous words** go to their most common sense. | *constante* → chemistry | Phase 4: the hybrid fusion gives the exact word weight again |
| 3 | **Building needs a GPU** in practice: BGE-M3 on CPU would take about 2 hours for 10,000 chunks. | A teammate without a GPU can't rebuild with BGE-M3 | Build once on the GPU machine and share `index/` (as decided in Phase 0); `--model intfloat/multilingual-e5-small` for development |
| 4 | **Semantic snippets** are chosen by shared words. | On type-C results, the snippet is the start of the chunk, not the most relevant part | Accepted |
| 5 | **First query after start-up waits for the model** (about 15 s), shown as the loading spinner. | Only once per session | Accepted |
| 6 | The library checks the Hugging Face Hub when loading a model (a warning about unauthenticated requests). | Harmless online; offline it falls back to the local copy | Possible: set `HF_HUB_OFFLINE=1` once the models are downloaded |

---

## 9. How to review the code yourself

1. Run the tests: `python -m pytest` (the Phase 3 tests need e5-small in `models/hub`).
2. Rebuild the full index on the GPU machine (about 7 min): `python scripts/build_index.py`.
   Check the line `embeddings: BAAI/bge-m3 on cuda, 10295 x 1024 … (51 chunks … truncated)`.
3. Start the page, choose "Sémantique", and try a question with no shared word, for example
   *"quelqu'un qui aide les élèves à comprendre la photosynthèse"*. Then compare with BM25.
4. Try a typo (*"la fonction rationelle"*) in BM25 and in semantic mode.
5. Try the "Corpus équipe" filter.

## 10. Handoff to Phase 4

Phase 4 combines the branches and adds the corrections:
- **Hybrid fusion (S4)**: normalize the BM25 and semantic scores, combine them with weight
  α (`DEFAULT_ALPHA = 0.5`), using the top `BRANCH_TOP_K` of each branch. Tune α on the 300
  dev questions. The 65 / 18 split above is the expected gain.
- **Reranker (S5)**: `bge-reranker-v2-m3` on the top 20 chunks.
- **Typo and accent correction** against the index vocabulary. Fixes the accent gap from
  Phase 2 (*infirmiere*), and lowercases all-capitals queries.
- Optional: WordNet expansion and averaged word2vec (S6).
