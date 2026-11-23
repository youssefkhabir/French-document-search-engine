# Phase 2 — Technical review

**Project:** Moteur de recherche documentaire intelligent (French semantic search, TALN course)
**Phase:** 2 · Lexical branch and first interface
**Date:** 23 November 2026
**Status:** done. 61 tests pass (20 new: 18 in `test_phase2.py`, 2 in `test_phase1.py`). Both lexical modes run on the full Alloprof index
and in the Streamlit page.

---

## 1. What Phase 2 delivers

The plan's "done when" criterion: *you can type a query and see ranked documents in two
modes.*

```
chunks.text_index ──lexical_prep──▶ text_lex ──┬──▶ TF-IDF  (index/tfidf.pkl)
                                               └──▶ BM25    (index/bm25/)

query ──lexical_prep──▶ query_lex ──▶ score every chunk ──▶ filter ──▶ best chunk per document ──▶ snippet
```

| File | Role | Course technique |
|---|---|---|
| `search/lexical.py` | `lexical_prep`, TF-IDF, BM25, snippets | **Tokenisation, mots vides, lemmatisation, TF-IDF** |
| `search/engine.py` | `build_index()` adds the lexical indexes; `load_index()`, `search()` | — |
| `app.py` | Streamlit page: mode, filters, result cards | — |
| `search/clean.py` | Small fix: leftover HTML entities decoded | Expressions régulières |
| `tests/test_phase2.py` | 18 tests | — |

**Result on Alloprof:** 2,350 pages → 7,074 chunks → a vocabulary of **46,082 terms**. The
full build takes about **2 minutes** (124 s lexical, almost all of it spaCy). A search takes
**100–200 ms** once the index and spaCy are loaded.

---

## 2. `lexical_prep` — one function for documents and queries

### The pipeline

```
"Les développeurs travaillaient sur des applications web."
   │ spaCy tokenization + POS + lemmas (fr_core_news_sm, parser and NER disabled)
   ▼
 Les | développeurs | travaillaient | sur | des | applications | web | .
   │ drop punctuation and stopwords (Les, sur, des, .)
   ▼
 développeurs | travaillaient | applications | web
   │ lemma → lowercase → strip accents
   ▼
"developpeur travailler application web"
```

The one rule from the plan: **the documents and the query go through the same function**,
so a word in the query and the same word in a document always end up as the same term.

```python
def _token_terms(tok, lemmatize: bool, remove_stopwords: bool) -> list[str]:
    if tok.is_punct or tok.is_space or not any(c.isalnum() for c in tok.text):
        return []
    if remove_stopwords and tok.is_stop:
        return []
    word = tok.lemma_ if lemmatize and tok.lemma_.strip() else tok.text
    out = []
    for part in strip_accents(word.lower()).split():   # spaCy can return "à le"
        part = part.strip("'’-.")
        if remove_stopwords and part in _stopwords():  # "est-ce" -> "-ce" -> "ce"
            continue
        if len(part) >= 2 or part.isdigit():
            out.append(part)
    return out
```

### Design choices

| Choice | Why |
|---|---|
| **Lemma before lowercase** | spaCy's tagger uses capitals to guess the part of speech; lowercasing first gives worse lemmas. |
| **Accents stripped last** | Students often leave out accents. Stripping them on both sides makes *"équivalent"* and *"equivalent"* the same term **once lemmatized** (see the limitation in §8). |
| **Stopwords checked twice** | spaCy keeps *"-ce"* (from *est-ce*) as a non-stopword token. Checking again after stripping `'-.` removes it. The stopword list is stored without accents to match. |
| **Numbers kept** | Years, article numbers and norms (*ISO 27001*, *1759*) are useful search terms. |
| **One-letter terms dropped** | *l'*, *d'*, *a* carry no meaning once the stopwords are gone. |
| **`parser` and `ner` disabled** | Lemmas need only the tagger, morphologizer and lemmatizer. Measured on 300 Alloprof chunks: 5.5 s instead of 7.9 s (about 1.4× faster). |
| **`lexical_prep_many` with `nlp.pipe`** | Batches of 64 texts; this is what indexing uses. |

### Options for the comparisons

`lemmatize=False` keeps the surface words, and `remove_stopwords=False` keeps the
stopwords. Phase 5 can compare these settings by passing the options; the index code
doesn't change.

| Input: *"Les développeurs travaillaient"* | Output |
|---|---|
| default | `developpeur travailler` |
| `lemmatize=False` | `developpeurs travaillaient` |
| `remove_stopwords=False` | `le developpeur travailler` |

### Examples on real Alloprof questions

| Question | Lexical form |
|---|---|
| *Bonjour! Je voulais d'aide pour les fractions équivalentes* | `bonjour vouloir aide fraction equivalent` |
| *est-ce que un adverbe est quelque chose que tu peut pas mimer?* | `adverbe chose mimer` |
| *comment calculer l'aire d'un triangle* | `calculer aire triangle` |

The second line shows why stopword removal matters on chatty questions: 13 words become 3
meaningful terms. It also shows a cost: *pas* is a stopword, so the negation is lost.

---

## 3. S1 · TF-IDF

```python
vectorizer = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True)
matrix = vectorizer.fit_transform(texts_lex)          # chunks × terms, rows L2-normalized
...
q = vectorizer.transform([query_lex])
scores = (matrix @ q.T).toarray().ravel()             # = cosine similarity
```

- **`token_pattern=r"\S+"`, `lowercase=False`**: the text is already normalized by
  `lexical_prep`, so scikit-learn must only split on spaces. Its default pattern would
  drop one-character tokens and re-tokenize numbers.
- **`sublinear_tf=True`**: weight `1 + log(tf)` instead of `tf`. A term repeated 10 times in
  a chunk is not 10 times more relevant. This matters on Alloprof, where some pages repeat
  a sentence (§8).
- **IDF**: scikit-learn's smoothed `log((1+N)/(1+df)) + 1`.
- **Cosine**: rows are L2-normalized, so a sparse dot product with the query vector is the
  cosine. No extra normalization is needed.
- Saved with `pickle` as `index/tfidf.pkl` (vectorizer + sparse matrix).

## 4. S2 · BM25

```python
index = bm25s.BM25()                                  # Lucene variant, k1=1.5, b=0.75
index.index([t.split() for t in texts_lex])
...
ids = index.get_tokens_ids(query_lex.split())         # unknown terms are dropped
scores = index.get_scores(ids)                        # one score per chunk
```

- **bm25s** stores the BM25 weights of every (term, chunk) pair in a sparse matrix, so a
  query is a sum of a few columns. It's fast in plain NumPy.
- We give it **our own tokens**, not `bm25s.tokenize`. Its tokenizer defaults to English
  stopwords and would break the "same function for documents and queries" rule.
- **`get_scores` instead of `retrieve`**: we need the score of every chunk so the
  collection and file-type filters can be applied before taking the top k (§5).
- Unknown query terms are dropped by `get_tokens_ids`. A query with no known term returns
  no results instead of an error.
- Saved with `index.save()` under `index/bm25/` (NumPy arrays + JSON vocabulary).

**TF-IDF vs BM25, the difference the report should explain:** BM25 saturates term
frequency (parameter `k1`) and normalizes by chunk length relative to the average
(parameter `b`). TF-IDF uses a log for frequency and cosine normalization of *both* vectors.

---

## 5. `engine.search()` — from chunk scores to documents

```
query_lex ──▶ score all 7,074 chunks ──▶ mask (filters) ──▶ sort positive scores
          ──▶ keep the first chunk of each document (max-pooling) ──▶ top k documents
          ──▶ snippet from that chunk
```

### Filters are a mask, not a post-filter

```python
mask = np.ones(len(chunks), dtype=bool)
if collections: mask &= chunks["collection"].isin(collections).to_numpy()
if formats:     mask &= chunks["format"].isin(formats).to_numpy()
scores = np.where(mask, scores, 0.0)
```

Filtering *before* the top k means a filtered search still returns k results. Filtering
after would return fewer, or none.

### Every matching chunk is ranked

Lexical scoring is cheap, so `search()` ranks **all** chunks with a positive score, not only
the top 50 (`BRANCH_TOP_K`). Without this, a long document like *Candide* (178 chunks) can
fill the 50 slots by itself and push every other document out of the top 10. Phase 4's
fusion will still use `BRANCH_TOP_K` for the semantic branch.

### One result per document

A document's score is the score of its **best chunk** (max-pooling). That chunk also gives
the page number (PDF) and the snippet. Summing chunk scores instead would favour long
documents just for being long.

### The result format

```python
{"query", "query_lex", "corrected_query", "corrections", "mode", "n_chunks_matched",
 "results": [{"doc_id", "title", "collection", "format", "page", "score", "snippet",
              "highlights", "chunk_id", "meta", "rank_bm25", "rank_semantic"}],
 "timings": {"load", "prep", "search", "snippets"}}       # milliseconds
```

The fields of later phases (`corrected_query`, `corrections`, `rank_semantic`) are already
in place, so the page and the evaluation won't need to change their reading code.
`search()` raises `NotImplementedError` for the modes that aren't ready yet.

### Index loading

`load_index()` keeps the loaded index in memory and reloads it when `meta.json` changes, so
a rebuild is picked up without restarting the app. The page wraps it in
`st.cache_resource` and loads spaCy at the same time, so the first query no longer waits
10 s for spaCy.

---

## 6. Snippets and highlighting

```python
def snippet(text, terms, n_words=45) -> tuple[str, list[tuple[int, int]]]:
```

1. Run spaCy on the chunk's `text_raw` (the original text, not the lexical form).
2. Mark each token whose lexical form is a query term. This is why *"développeurs"*,
   *"triangles"* or *"Calculer"* are highlighted for a query that says *développeur*,
   *triangle*, *calculer*.
3. Choose the 45-token window that holds the most marked tokens, starting 8 tokens before a
   match so the reader sees the context.
4. Return the text and the **character spans** of the matches. `…` marks a cut.

The engine returns spans, not HTML: the page escapes the text, wraps the spans in `<mark>`,
and writes `$` as `&#36;` (otherwise `st.markdown` would read it as LaTeX). The same result
can be printed in a terminal or saved by the evaluation without any HTML.

Cost: one spaCy pass on 10 chunks per query, about 30–150 ms.

---

## 7. The Streamlit page

| Area | Content |
|---|---|
| Sidebar · Index | Chooses any project folder holding a `meta.json` (`index`, `index_test`) |
| Sidebar · Mode | TF-IDF, BM25 active; Sémantique, Hybride and Word2vec say which phase brings them |
| Sidebar · Filtres | Collection, file type (multi-select, empty = all), number of results |
| Sidebar · Options | Typo correction, WordNet, reranker: shown, disabled until Phase 4 |
| Sidebar · Corpus indexé | Documents and passages per collection, vocabulary size, build date |
| Results | Count, matching passages, time, the query's lexical form; then one card per document: title, collection, format, PDF page, Alloprof subject (in French), score, `doc_id`, highlighted snippet |

Showing the **lexical form of the query** on the page is on purpose: it explains a result
during the demo (*"your question became `adverbe chose mimer`"*).

---

## 8. Results and findings

### Quick check on the 300 Alloprof dev questions

Not the Phase 5 evaluation (that one uses the 2,281 test questions and nDCG); a sanity check
that the branch works. Script kept outside the project.

| Mode | Recall@10 | MRR@10 | Time / query |
|---|---|---|---|
| S1 · TF-IDF | **0.620** | **0.398** | ~150 ms |
| S2 · BM25 | 0.590 | 0.383 | ~105 ms |

- About 6 in 10 relevant pages appear in the top 10 with words alone. That's the baseline
  the semantic branch has to beat on type-C (no shared word) and type-D (noisy) questions.
- **TF-IDF is slightly ahead of BM25.** That's unexpected and must be checked on the full
  test set with a significance test in Phase 5. A likely reason: Alloprof questions are
  long and chatty, and TF-IDF's cosine normalizes the query vector, while BM25 adds up
  every term's contribution, including weak ones like *bonjour* or *aide*. Possible
  follow-ups: tune `k1` and `b` on the dev questions, or compare with the stopwords kept.

### Example queries on the test documents

| Query | Mode | Top result | Highlighted |
|---|---|---|---|
| *carte d'identité pour un enfant* | BM25 | Service-Public F1342 | carte, identité |
| *reconnaissance d'entités nommées* | BM25 | TALN 2023 paper, page 3 | entités nommées, reconnaissance |
| *festival de jazz en Bretagne* | BM25 | Panorama des festivals (*Bain de Blues*) | Bretagne, Jazz |
| same, filter PDF + HTML | BM25 | no result: correct, no such document | — |
| *Pangloss optimisme* | TF-IDF | *Candide* | Pangloss, optimisme |

### Bugs found and fixed during the phase

| Bug | How it was found | Fix |
|---|---|---|
| `&amp;` shown in a snippet; Alloprof has 653 `&amp;` and ~120 `&lt;`/`&gt;` | Reading the first result card | `clean_text` decodes the common HTML entities (with their `;`); `R&D`, `AT&T` unchanged. 2 tests |
| *"est-ce"* left a term `ce` | Reading the lexical form of a student question | Stopword check after stripping `-` |
| First query took 12 s | Timing shown on the page | spaCy loads together with the index |
| A long document could fill the 50 candidate slots | Reasoning on *Candide* (178 chunks) | Lexical modes rank every matching chunk |
| Port 8501 already in use | Starting the preview | `.claude/launch.json` uses 8510 |

### Known limitations

| # | Limitation | Effect | Where it gets fixed |
|---|---|---|---|
| 1 | **A word typed without accents isn't lemmatized.** spaCy knows *"infirmières"* → `infirmier`, but *"infirmiere"* stays `infirmiere`. | Missed matches on noisy questions, which are common in Alloprof. A test documents the gap. | Phase 4: typo/accent correction against the index vocabulary |
| 2 | **spaCy's small model mangles some lemmas**: *Pangloss* → `panglos`, *optimisme* → `optimism`. | Harmless for matching (query and documents are changed the same way), visible in the lexical form shown on the page. | Accepted; `fr_core_news_md` is an option |
| 3 | **Negation is a stopword** (*pas*, *ne*). | *"ce qu'on ne peut pas mimer"* matches pages about what you *can* mime. | Accepted for a bag-of-words model; the semantic branch handles it better |
| 4 | **Alloprof pages contain LaTeX** (`\color{blue}{\text{…}}`, `\begin{align}`) in 1,222 of the 7,074 chunks. | LaTeX commands become terms (`color`, `text`, `begin`) and appear in snippets. | Possible cleaning step; to decide with Phase 5 numbers |
| 5 | **10 Alloprof pages repeat one sentence** (*"Pour retourner au menu en haut de la page, clique ici"*). | `sublinear_tf` dampens it; only 10 of 2,350 pages. | Accepted |
| 6 | **Indexing time is spaCy.** 124 of 129 s. | Fine for 7,000 chunks; 10 times more would need `n_process` or a bigger batch. | If needed |

---

## 9. Tests

`tests/test_phase2.py`, 18 tests (plus 2 entity cases added to `test_phase1.py`):

| Group | What it checks |
|---|---|
| `lexical_prep` | Lemmas, stopwords incl. `-ce`, numbers kept, the two options, accents stripped, same term from a document and a query |
| Known gap | A query without accents does **not** match yet (limitation 1). This test flips when Phase 4 fixes it |
| TF-IDF and BM25 | Only rows with the term, scores ordered, two terms beat one, unknown and empty queries, filter mask; save/load gives identical results |
| Snippet | Inflected forms highlighted (*développeurs* for `developpeur`), `…` on both sides, no match → start of the text |
| End to end | Index built on the fixtures; *"développeur backend Python"* finds the backend offer first in both modes; one entry per document; scores sorted; file-type and collection filters; no match; unready and unknown modes |

The end-to-end tests build a real index on the 23 fixtures once per test run (module-scope
fixture), in about a second.

---

## 10. How to review the code yourself

1. Run the tests: `python -m pytest`.
2. Rebuild the index (about 2 min): `python scripts/build_index.py`.
3. Look at lexical forms:
   `python -c "from search.lexical import lexical_prep; print(lexical_prep('Les élèves étudiaient les fractions'))"`
4. Start the page: `streamlit run app.py`, then compare TF-IDF and BM25 on the same
   question, try the filters, and switch to `index_test` for the mixed-format documents.
5. Try a question without accents (*"fraction equivalente"*) and with accents
   (*"fraction équivalente"*) to see limitation 1.

## 11. Handoff to Phase 3

Phase 3 builds the semantic branch on the same `chunks.parquet`:
- embed `text_index` with BGE-M3 (about 7 minutes on the RTX 2060), saved to `index/`
- `search_semantic()` with cosine similarity, top `BRANCH_TOP_K`
- mode "Sémantique" switched on in `MODES_READY` and in the page

Done when the semantic mode answers questions that share no word with the page (type C),
where TF-IDF and BM25 fail.
