# Plan addendum — optional components from TALN Chapter 3

The original plan (`Plan_Moteur_Recherche_Documentaire.pdf`, section 06 "Left out on purpose")
skipped WordNet and word2vec. Chapter 3 of the course is built around exactly these two
ideas, so they come back as **optional, measurable components**. The core system (S1–S5)
does not depend on them; they are added after S4 works.

## Why

Chapter 3 goes: word meaning → relations between meanings (synonymy, hyponymy, similarity)
→ semantic databases (WordNet, Wu–Palmer) → sparse vectors (bag of words, term–context
matrix, TF-IDF) → dense vectors (word2vec). With both additions, the project covers the
whole progression and the report can compare each family on the same queries:

| Family (course) | System in the project |
|---|---|
| Sparse vectors — TF-IDF | S1 · TF-IDF |
| Sparse, refined | S2 · BM25 |
| Semantic database — WordNet | **S2+W · BM25 + WordNet expansion** (optional) |
| Static dense vectors — word2vec | **S6 · averaged word2vec** (optional) |
| Contextual dense vectors | S3 · BGE-M3 / e5 |
| Combination | S4 · Hybrid, S5 · Hybrid + rerank |

## Optional component A — WordNet query expansion (`search/wordnet_expand.py`)

- Source: NLTK WordNet + Open Multilingual WordNet (`omw-2.0` for NLTK ≥ 3.10), French lemmas via
  `wn.synsets(lemma, lang="fra")` and `synset.lemma_names("fra")`.
- Applied to the **query only**, after typo correction and lemmatization, on nouns,
  verbs and adjectives.
- For each query lemma: add synonyms from its synsets (optionally one level of hypernyms),
  capped at `k` new terms per word (default 3). Expanded terms go through the same
  `lexical_prep` so they match the index.
- Original terms are repeated so they keep more weight than the expansions
  (bm25s has no per-term weights; repetition is the simple workaround).
- Interface: a checkbox "Expansion WordNet" (BM25 and hybrid modes).
- Report extra: Wu–Palmer similarity (`wup_similarity`) on a few word pairs from the
  corpus (e.g. *développeur / ingénieur*) next to their cosine similarity, to contrast a
  taxonomy-based measure with a vector-based one.
- **Expected risk:** French WordNet coverage is uneven and polysemy adds noise
  (*souris* → animal), so expansion may lower precision on type-A queries. That is a
  valid finding.

**Measured by:** ablation table, BM25 vs BM25+WordNet, nDCG@10 per query type (A/B/C/D).

## Optional component B — averaged word2vec baseline (`search/word2vec.py`)

- Default: train gensim `Word2Vec` (skip-gram, dim 100, window 5, min_count 2) on the
  **lexical form** of the corpus chunks. Runs in seconds, fully offline, no download.
- Optional: load pretrained French vectors (any `KeyedVectors` file) set by
  `W2V_PRETRAINED_PATH` in `search/config.py`. A model trained on only 150–300 documents is
  weak. Comparing it with pretrained vectors is itself a result.
- Chunk vector = mean of its word vectors (optionally TF-IDF weighted); query the same way;
  cosine similarity; top-k like the semantic branch.
- Shown as mode "Word2vec (moyenne)" in the interface and as row **S6** in the results table.
- Report extra: nearest neighbours of a few corpus words (`most_similar`) to illustrate
  the distributional hypothesis from the course.

**Measured by:** row S6 in the main table; expected between BM25 and the transformer on
type-C queries, and weak on type-A.

## Where they fit in the phases

- Phase 0: dependencies (`nltk`, `gensim`) and NLTK data downloaded with the rest.
- Phase 4 (stretch): implement A and B after S4 works.
- Phase 5: two extra rows/ablations; report section "Techniques du cours" maps each
  chapter concept to its module.
