# Progress

**Project:** Moteur de recherche documentaire intelligent (French semantic search, TALN course)
**Last update:** 24 November 2026
**Last commit:** `e6ddf6e` on `main` (today's work not committed yet)
**Tests:** 104 pass (`python -m pytest`)

---

## Where we are

| Phase | Status | Review |
|---|---|---|
| 0 · Skeleton, environment, embedding benchmark | ✅ Done | `docs/phase0_review.md` |
| 1 · Extraction, cleaning, chunking | ✅ Done, plus fixes from real web documents | `docs/phase1_review.md` |
| 2 · Lexical branch (TF-IDF, BM25) and first page | ✅ Done | `docs/phase2_review.md` |
| 3 · Semantic branch (BGE-M3) | ✅ Done | `docs/phase3_review.md` |
| 4 · Hybrid fusion, reranker, typo correction | ✅ Core done · ✅ WordNet · ⏳ word2vec (optional) not started | `docs/phase4_review.md` |
| 5 · Evaluation and report | 🔄 Metrics and `eval/run_eval.py` done; test-set run not started | — |

Decisions that override the original plan are in `docs/plan_addendum.md`.

---

## Results so far (300 Alloprof dev questions)

Tuning set only. The 2,281 test questions are kept for Phase 5.

| System | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|
| S1 · TF-IDF | 0.618 | 0.393 | 0.435 |
| S2 · BM25 + typo correction | 0.606 | 0.395 | 0.434 |
| S3 · Semantic (BGE-M3) | 0.733 | 0.553 | 0.584 |
| S4 · Hybrid, RRF | 0.741 | 0.529 | 0.564 |
| **S4 · Hybrid, weighted α = 0.7** | **0.765** | **0.573** | **0.605** |
| **S5 · Hybrid + reranker** | **0.787** | **0.584** | **0.622** |
| S2 + WordNet (k=1, originals ×2) | 0.598 | — | 0.436 |

Main findings to keep for the report:
- The semantic branch is the big step (+15 nDCG points over the lexical modes). It handles
  typos and questions that share no word with the page.
- BM25 still wins on some questions (exact words, polysemous terms), so the hybrid adds
  2 points. α = 0.7 is best, on a flat top from 0.6 to 0.8; the plan's 0.5 was 2 points lower.
- Weighted fusion beats RRF by 4 points.
- The reranker adds about 2 points but makes queries 14× slower (0.5 s on GPU).
- Typo correction helps the lexical modes (+2 points), not the semantic one.
- TF-IDF is slightly ahead of BM25 on these chatty student questions. To confirm on the
  test set.
- WordNet expansion is neutral at best (k=1) and hurts with more synonyms (−8.7 points at
  k=3): polysemy, and the semantic branch already covers synonymy.
- Cutting chunks at 512 tokens (7% of Alloprof chunks, because of LaTeX) made no measurable
  difference; the cap is 1,024 now anyway.

---

## Data and index (local, not in git)

| What | Where | Content |
|---|---|---|
| Alloprof corpus | `data/corpus/alloprof/` | 2,350 course pages, 12 subjects (`scripts/fetch_alloprof.py`) |
| Team corpus | `data/corpus/divers/` | **10 documents only** (public web files: 2 PDF, 3 DOCX, 2 HTML, 1 TXT, 2 CSV) |
| Alloprof questions | `eval/alloprof/` | 2,281 test + 300 dev questions with relevance judgments |
| Team questions | `eval/queries.json`, `eval/qrels.csv` | **Templates only** (2 example queries, no judgments) |
| Index | `index/` | 2,360 documents → 10,295 chunks, BGE-M3 embeddings (built in ~7 min on the RTX 2060) |
| Test documents | `test_docs/`, `index_test/` | The same 10 web files, for quick checks |

Rebuild the index after changing the corpus: `python scripts/build_index.py`.

---

## Next steps

### Phase 4, optional (Chapter 3 of the course)
- [x] **WordNet query expansion** (`search/wordnet_expand.py`): done, measured, Wu–Palmer
      examples in `docs/phase4_review.md` §3b.
- [ ] **Averaged word2vec, S6** (`search/word2vec.py`): gensim trained on the lexical forms;
      one more row between BM25 and the transformer.

### Phase 5: evaluation and report
- [x] `eval/metrics.py` (Recall, MRR, nDCG by hand) and `eval/run_eval.py`
      (`python -m eval.run_eval --split dev --systems S1 S2 S3 S4`); reproduces the dev table.
- [ ] `eval/run_eval.py` on the **2,281 test questions**: S1 to S5 (and S6), Recall@10,
      MRR@10, nDCG@10, per subject.
- [ ] Paired significance test (Wilcoxon or bootstrap): S4 vs S3, S5 vs S4, BM25 vs TF-IDF.
- [ ] Ablations: title prefix on/off, lemmas vs raw words, stopwords on/off.
- [ ] Error analysis: five failures, from both collections.
- [ ] Report section mapping each course concept to its module.

### Team work (data track)
- [ ] Collect **50–100 documents** for `data/corpus/divers/`, from 5–8 themes, in the five
      formats. Consider replacing `panorama-des-festivals.csv`, which is 2,895 of the
      collection's 3,221 chunks.
- [ ] Write the **~40 queries** (types A/B/C/D) in `eval/queries.json` and judge the pooled
      top 10 (0/1/2) in `eval/qrels.csv`.
- [ ] Rebuild the index once the corpus is complete.

---

## Known limitations (details in the phase reviews)

- Alloprof pages are flat text with LaTeX: no paragraph breaks, and LaTeX commands become
  search terms.
- Typos 3+ edits away aren't corrected (the semantic branch covers them).
- The reranker needs a GPU to be practical.
- Building with BGE-M3 needs a GPU; a CPU machine uses the shared `index/` folder.
- α is tuned on Alloprof; the team corpus may prefer another value.

## How to resume

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8510
```

Port 8501 is used by another local project. `.claude/launch.json` starts the page on 8510.
