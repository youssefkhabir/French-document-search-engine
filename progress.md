# Progress

**Project:** Moteur de recherche documentaire intelligent (French semantic search, TALN course)
**Last update:** 24 November 2026
**Last commit:** see `git log` (`main`, local commits not pushed)
**Tests:** 111 pass (`python -m pytest`)

---

## Where we are

| Phase | Status | Review |
|---|---|---|
| 0 · Skeleton, environment, embedding benchmark | ✅ Done | `docs/phase0_review.md` |
| 1 · Extraction, cleaning, chunking | ✅ Done, plus fixes from real web documents | `docs/phase1_review.md` |
| 2 · Lexical branch (TF-IDF, BM25) and first page | ✅ Done | `docs/phase2_review.md` |
| 3 · Semantic branch (BGE-M3) | ✅ Done | `docs/phase3_review.md` |
| 4 · Hybrid fusion, reranker, typo correction | ✅ Done, including optional WordNet and word2vec | `docs/phase4_review.md` |
| 5 · Evaluation and report | ✅ Done on Alloprof (100 test questions) · ⏳ team corpus part waits for its queries | `docs/phase5_review.md` |

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
| S6 · word2vec, IDF-weighted mean | 0.395 | 0.250 | 0.273 |

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
- Averaged word2vec (S6) is the weakest system (0.273), 16 points under BM25, but finds 12
  questions BM25 misses. Its neighbours show relatedness (fraction → dénominateur), not
  synonymy.
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
- [x] **Averaged word2vec, S6** (`search/word2vec.py`): done, measured, neighbours in
      `docs/phase4_review.md` §3c. Below BM25, not between BM25 and the transformer.

### Phase 5: evaluation and report
- [x] `eval/metrics.py`, `eval/run_eval.py` (`--sample 100`), `eval/significance.py`,
      `eval/ablations.py`.
- [x] All systems on **100 test questions** (seeded sample, team decision instead of 2,281):
      S5 0.587, S4 0.576, S3 0.561, S1 0.442, S2 0.427, S2W 0.424, S6 0.330 nDCG@10.
- [x] Significance: only semantic vs BM25 (+13.4) and word2vec vs BM25 (−9.7) are
      significant on 100 questions; hybrid, reranker, TF-IDF vs BM25 are not.
- [x] Ablations: stopwords −7/−8, lemmas −1.4/−3.7, title prefix −1.5 (BM25) and −4.3 (semantic).
- [x] Error analysis: five Alloprof failures. Team-corpus failures wait for its queries.
- [x] Course concept map: `docs/phase5_review.md` §6.
- [ ] Optional: full 2,281 test run (~25 min) to settle the 1–2-point effects.
- [ ] Team corpus evaluation (`python -m eval.run_eval --dataset team`) once judged.

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
