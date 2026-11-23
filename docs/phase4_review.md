# Phase 4 — Technical review

**Project:** Moteur de recherche documentaire intelligent (French semantic search, TALN course)
**Phase:** 4 · Hybrid fusion, reranker, typo correction (core). WordNet and word2vec, the
optional Chapter 3 components, are not done yet.
**Date:** 23 November 2026
**Status:** core done. 96 tests pass (26 new). α tuned on the 300 Alloprof dev questions.

---

## 1. What Phase 4 delivers

```
query ──typo / accent / capitals correction──▶ corrected query ──lexical_prep──▶ BM25 top 50 ──┐
  │                                                                                          ├─▶ fusion (min-max + α, or RRF)
  └──capitals lowered only──▶ BGE-M3 ──▶ semantic top 50 ─────────────────────────────────────┘
                                                                     │
                       best chunk per document ◀─────────────────────┘
                                 │
                 (optional) cross-encoder reorders the top 20 documents ──▶ results
```

| File | Role | Course technique |
|---|---|---|
| `search/spelling.py` | Vocabulary, accent restoration, Levenshtein correction, capitals | **Distance d'édition (Levenshtein)** |
| `search/fusion.py` | Min-max weighted fusion, reciprocal rank fusion | — |
| `search/rerank.py` | Cross-encoder `bge-reranker-v2-m3` | **Transformers (cross-encoder)** |
| `search/engine.py` | `search()` rewritten: correction → branches → fusion → rerank; `vocab.json` in the index | — |
| `search/config.py` | `DEFAULT_ALPHA = 0.7`, `FUSION`, `RRF_K`, `TYPO_*`, `SEMANTIC_USES_CORRECTION`, `RERANK_*` | — |
| `app.py` | Hybrid by default, fusion choice, α, typo and reranker options, corrected query, branch ranks | — |
| `tests/test_phase4.py` | 26 tests | — |

---

## 2. Results on the 300 Alloprof dev questions

Tuning set only. The test questions are kept for Phase 5. Index: 10,295 chunks (Alloprof +
team corpus). Binary relevance; nDCG@10 added this phase.

| System | Recall@10 | MRR@10 | nDCG@10 | ms / query* |
|---|---|---|---|---|
| S1 · TF-IDF | 0.618 | 0.393 | 0.435 | 11 |
| S1 · TF-IDF + typo | 0.638 | 0.407 | 0.451 | 10 |
| S2 · BM25 | 0.582 | 0.379 | 0.416 | 8 |
| S2 · BM25 + typo | 0.606 | 0.395 | 0.434 | 10 |
| S3 · Semantic | 0.733 | 0.553 | 0.584 | 48 |
| S4 · Hybrid RRF | 0.741 | 0.529 | 0.564 | 38 |
| **S4 · Hybrid weighted, α = 0.7** | **0.765** | **0.573** | **0.605** | 39 |
| **S5 · Hybrid α = 0.7 + reranker** | **0.787** | **0.584** | **0.622** | 531 |

\* Without snippets, as the evaluation runs; the page adds 100–300 ms of snippets.

### Tuning α (weighted fusion, typo on)

| α | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | **0.7** | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nDCG@10 | 0.434 | 0.453 | 0.479 | 0.511 | 0.550 | 0.583 | 0.599 | **0.603** | 0.599 | 0.596 | 0.585 |
| Recall@10 | 0.606 | 0.624 | 0.648 | 0.693 | 0.728 | 0.747 | 0.761 | **0.761** | 0.756 | 0.747 | 0.732 |

- **α = 0.7** is the best, on a flat top from 0.6 to 0.8: the choice isn't fragile.
- The ends match the single branches: α = 0 is BM25 (0.434), α = 1 is semantic (0.585).
  This checks the fusion code.
- The plan's default 0.5 would have been **2 points lower** (0.583). Tuning on held-out
  train questions was worth it.
- The curve rises fast from 0 and slowly falls after 0.7: on Alloprof, the semantic branch
  carries most of the signal, and BM25 adds the exact-word cases.

### What each component brings (nDCG@10)

| Step | Gain |
|---|---|
| Typo correction, lexical modes | **+1.6 to +1.8** points |
| Hybrid over semantic alone | **+2.0** points (0.585 → 0.605) |
| Weighted fusion over RRF | **+4.1** points (0.564 → 0.605) |
| Reranker over the hybrid | **+1.7** points nDCG, **+2.2** recall, at **14×** the time |

Differences under about 0.5 point are noise on 300 questions. Phase 5 tests significance
on the 2,281 test questions.

---

## 3. `spelling.py` — typo, accent and capitals correction

### Rules (from the plan, plus two from earlier phases)

| Rule | Example | Why |
|---|---|---|
| A query mostly in **capitals** (≥ 60% of ≥ 8 letters) is lowercased first | *BONJOUR… PASSÉ COMPOSÉ* → *bonjour… passé composé* | Phase 3 finding: the transformer handles capitals badly, and the capitals rule below would protect every word |
| Words with a **capital or a digit** are kept | *ISO*, *Bonaparte*, *2022b* | Acronyms and names are often absent from the corpus but correct |
| Words under **4 letters** are kept | *nui* | Too many neighbours at distance 1 |
| A word that only **lacks accents** gets them back, at distance 0 | *infirmiere* → *infirmière*, *theoreme* → *théorème* | Phase 2 gap: spaCy can't lemmatize the accent-free form |
| A word **in the vocabulary** still gets its accents if the accented spelling is **≥ 20× more frequent** | *francaise* (6 in the corpus) → *française* (347) | The corpus has its own misspellings: 156 such words |
| Otherwise, a word **not in the vocabulary** goes to the closest word within **1 edit** (4–7 letters) or **2 edits** (8+) | *rationelle* → *rationnelle*, *photosinthèse* → *photosynthèse* | Levenshtein distance, the course technique |
| Targets must appear **≥ 2 times**; ties go to the more frequent word | | A word seen once may be a typo itself |

Comparison is accent-free; the answer is given in its usual accented spelling. Distances
are computed by `rapidfuzz` over the ~28,000 candidate words: about **1–3 ms** per word, up
to 50 ms for a long word with no match.

### Vocabulary

`build_vocabulary(text_raw)`: every word of 2+ letters, lowercase, accents kept, with its
frequency. **46,000 words** on the current index, built in about 1 s, saved as
`index/vocab.json`. An older index without the file gets it rebuilt at load time.

### Which branch gets the corrected query

| Variant (hybrid α = 0.7) | nDCG@10 |
|---|---|
| No correction | 0.606 |
| Corrected query to both branches | 0.603 |
| **Corrected query to BM25; original (capitals lowered) to semantic** | **0.605** |

The differences are noise, but the reason for the choice is sound: correction helps BM25 by
about 2 points (an unknown word scores nothing there), while the transformer reads typos
through its sub-word tokens. A "correction" can also change the meaning, so the semantic
branch gets the user's own words. `SEMANTIC_USES_CORRECTION = False`.

### Limits

- *probaliter* → *probabilité* is **3 edits**, beyond the limit: not corrected. The
  semantic branch finds it anyway (Phase 3).
- A rare but correct word absent from the corpus can be "corrected" to a neighbour. The
  page always shows the corrected query (*"Recherche corrigée : …"*) and the checkbox turns
  correction off.

---

## 4. `fusion.py`

### Weighted fusion (S4, the main system)

```python
def weighted_fusion(lex, sem, alpha):
    nl, ns = minmax(dict(lex)), minmax(dict(sem))
    fused = {row: alpha * ns.get(row, 0.0) + (1 - alpha) * nl.get(row, 0.0)
             for row in nl.keys() | ns.keys()}
```

- **Min-max per list**: BM25 scores (0 to ~30) and cosines (0.3 to 0.8) live on different
  scales. Each list is rescaled to [0, 1] **over its own 50 candidates**.
- **Top 50 per branch, not every chunk**: rescaling over all 10,000 chunks would squash the
  scores that matter into a narrow band near 1. The single modes still rank every chunk
  (Phase 2 reasoning); the hybrid follows the plan's `BRANCH_TOP_K = 50`.
- **Absent → 0**: a chunk found by one branch only gets nothing from the other. Being in
  both lists is a signal of its own.
- **Guard max = min → 1** (the plan said 0): when a branch returns a single candidate, or
  ties, there is no spread to rescale. Giving 0 would erase the branch's only find; 1 keeps
  it. A test covers it.

### Reciprocal rank fusion (comparison row)

`score = Σ 1 / (60 + rank)`. Ranks only: no normalization, no α. It is **4 points below**
weighted fusion here. RRF treats both branches as equal, while the semantic branch is
clearly stronger on Alloprof; α lets the weighted fusion follow that.

### Branch ranks on the page

Each hybrid result shows its rank in both branches, for example *"BM25 #10 / sémantique #9"*,
or *"BM25 —"* when BM25 didn't have it in its top 50. That explains a result in the demo.

---

## 5. `rerank.py` — cross-encoder (S5)

A bi-encoder (Phase 3) embeds the query and the passage **separately** and compares two
vectors. A cross-encoder reads **the pair together**, so its attention can link each query
word to the passage. It is much more precise and much slower: it can't score 10,000 chunks
per query, only reorder a short list.

| Choice | Why |
|---|---|
| Model `BAAI/bge-reranker-v2-m3` | Multilingual, same family as BGE-M3, already downloaded |
| **Rerank the best chunk of the top 20 documents** | Reranking the top 20 *chunks* could leave fewer than 10 documents once grouped |
| `text_index` (title + chunk) as the passage | The title often states the topic the chunk assumes |
| fp16 on GPU, `max_length = 1024`, batch 4 | Same reasoning as the embeddings; fits beside BGE-M3 in 6 GB |
| Scores shown as *"reranker 0.888"* | Sigmoid outputs in [0, 1], comparable within one query |

**Cost:** about **0.5 s per query** on the RTX 2060 (20 pairs), and **~20 s to load** the
model the first time (the page shows a spinner). The first load of this session took 441 s:
a cold read of the 2.2 GB file through the junction to another folder. It didn't happen
again. On CPU the reranker would take several seconds per query, so it stays an option,
off by default.

**Example.** *"c'est quoi une constante ?"*: the hybrid ranks the chemistry constants first
(*Ka*, *équilibre*, *Kps*), because *constante* is frequent there. The reranker moves
*"Algèbre - Expressions algébriques"* from 10th to **1st**, which is the page the question's
relevance judgment names.

---

## 6. Changes in `engine.search()`

```python
search(query, mode="hybrid", alpha=None, rerank=False, typo_fix=True, wordnet=False,
       top_k=10, collections=None, formats=None, index_dir=None,
       fusion_method=None, snippets=True)
```

- New: `fusion_method` ("weighted" or "rrf"), `snippets=False` for the evaluation, which
  makes a hybrid query **39 ms** instead of ~300 ms.
- Result: `corrected_query` and `corrections` are filled; new fields `alpha`, `fusion`,
  `reranked`, and per result `rerank_score`. `rank_bm25` / `rank_semantic` are now the
  document's rank in each branch the mode ran.
- Timings: `correct`, `prep`, `embed`, `search`, `fusion`, `rerank`, `snippets`.
- `MODES_READY` gains "hybrid"; `load_index()` offers it when the index has embeddings.

---

## 7. Tests

`tests/test_phase4.py`, 26 tests. The end-to-end ones use e5-small and no reranker, so they
need no 2.2 GB download.

| Group | What it checks |
|---|---|
| Spelling (16) | Accents only; 1 edit; accents kept in the answer; in-vocabulary misspelling with a 50× rarer count; known word; capitals and digits protected; short word; nothing close; rare targets refused; corrections reported; capitals query lowered then corrected; `fix_capitals`; an acronym in a normal query stays; vocabulary building |
| Fusion (3) | Min-max incl. the max = min guard and empty input; weighted fusion at α = 0, 1 and 0.5 with hand-computed values, absent → 0; RRF values |
| End to end (7) | `vocab.json` written; hybrid finds both a word query and a no-shared-word query; α = 0 / 1 reproduce BM25 / semantic; RRF with branch ranks; typo fix makes *infirmiere* reach the nurse documents, and the gap is back with `typo_fix=False`; `snippets=False`; unknown fusion refused |

The Phase 2 test now uses "word2vec" as the mode that isn't ready, and checks that both
semantic and hybrid refuse an index without embeddings.

---

## 8. Known limitations

| # | Limitation | Note |
|---|---|---|
| 1 | Hybrid fusion uses 50 chunks per branch, so in rare cases fewer than 10 documents come out | Never seen on Alloprof; raise `BRANCH_TOP_K` if needed |
| 2 | Typo correction can change a correct rare word | The corrected query is always shown; one click turns it off |
| 3 | Typos 3+ edits away are not corrected | The semantic branch covers them |
| 4 | Reranker: 0.5 s per query on GPU, several seconds on CPU, 20 s first load | Optional, off by default |
| 5 | α is tuned on Alloprof questions | The team corpus may prefer another α; to check with its ~40 queries in Phase 5 |
| 6 | WordNet expansion and averaged word2vec (S6) not done | Optional; see §10 |

---

## 9. How to review the code yourself

1. `python -m pytest` (96 tests).
2. In the page, type *"En générale, C'est quoi la fonction rationelle?"*: see the
   correction line and *"BM25 #1 / sémantique #1"* on the first card.
3. Move α to 0 and to 1: the results become BM25's and the semantic mode's.
4. Type *"c'est quoi une constante ?"*, then tick "Reranker": the algebra page moves to 1st.
5. Type a question in capitals: *"BONJOUR C'EST QUOI LE PASSÉ COMPOSÉ"*.

## 10. Handoff

**Optional, Phase 4 stretch (plan addendum, Part 1):**
- **WordNet query expansion** (`wordnet_expand.py`): synonyms from Open Multilingual
  WordNet added to the BM25 query; ablation BM25 vs BM25 + WordNet; Wu–Palmer examples.
- **Averaged word2vec** (`word2vec.py`, S6): gensim trained on the lexical forms; one more
  row between BM25 and the transformer.

**Phase 5, evaluation and report:**
- `eval/run_eval.py` on the **2,281 test questions**: S1–S5 (and S6), Recall@10, MRR@10,
  nDCG@10, per subject, with a paired significance test (S4 vs S3, S5 vs S4).
- Part B: the team corpus and its ~40 hand-judged queries.
- Error analysis: five failures, from both collections.
