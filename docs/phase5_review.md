# Phase 5 review — evaluation

**Status:** done on Alloprof (100 test questions, by team decision). The team corpus part
waits for its queries and judgments. 111 tests pass.

---

## 1. What Phase 5 delivers

| File | Role |
|---|---|
| `eval/metrics.py` | Recall@10, MRR@10, nDCG@10, written by hand (graded relevance 0/1/2) |
| `eval/run_eval.py` | Runs systems S1–S6 on Alloprof or the team queries; `--sample N` draws N judged questions with a fixed seed (42); writes per-question, summary and per-subject tables |
| `eval/significance.py` | Paired bootstrap (10,000 resamples) and Wilcoxon signed-rank on per-question nDCG@10 |
| `eval/ablations.py` | Title prefix, lemmas and stopwords ablations, rebuilt in memory from the stored chunks |
| `tests/test_phase5.py` | 7 tests (metrics, sampling, significance) |

Output: `eval/results/` (not in git; rerun the commands in §8 to regenerate).

**Why 100 test questions and not 2,281.** Team decision to keep Phase 5 short. The 100 are
drawn at random with a fixed seed, so anyone gets the same sample. The cost is statistical
power: on 100 questions, only differences of about 5 nDCG points or more can be shown to be
significant (§3). The 300 dev questions give a second, larger view, but they were used for
tuning (α, WordNet settings), so they slightly favour the tuned systems.

---

## 2. Main results: 100 Alloprof test questions

| System | Recall@10 | MRR@10 | nDCG@10 | ms / query* | dev nDCG@10 (300) |
|---|---|---|---|---|---|
| S1 · TF-IDF | 0.614 | 0.405 | 0.442 | 11 | 0.435 |
| S2 · BM25 + typo | 0.591 | 0.385 | 0.427 | 8 | 0.434 |
| S2W · BM25 + typo + WordNet | 0.561 | 0.390 | 0.424 | 15 | 0.436 |
| S6 · word2vec (IDF-weighted mean) | 0.456 | 0.296 | 0.330 | 15 | 0.273 |
| S3 · Semantic (BGE-M3) | 0.715 | 0.519 | 0.561 | 42 | 0.587 |
| S4 · Hybrid RRF | 0.739 | 0.533 | 0.571 | 33 | 0.564 |
| **S4 · Hybrid weighted, α = 0.7** | 0.715 | **0.544** | 0.576 | 33 | 0.605 |
| **S5 · Hybrid + reranker** | **0.775** | 0.539 | **0.587** | 506 | 0.622 |

\* Median, without snippets, RTX 2060.

**The ranking of the families is the one the course predicts**, and the same on dev and
test: word2vec < sparse (TF-IDF, BM25) < contextual (BGE-M3) ≤ hybrid ≤ hybrid + reranker.

- **The contextual model is the big step:** +13.4 nDCG points over BM25, significant
  (p < 0.001). Student questions are long, chatty and full of typos; the pages are
  textbook French. Exact words don't match, meaning does.
- **Hybrid and reranker add 1–2 points each**, on dev and on test, but on 100 questions the
  gain is within noise (§3). The reranker mainly raises recall (+6 points); MRR doesn't move.
- **TF-IDF ≥ BM25** on these questions (+1.5 test, +0.1 dev, not significant). Long questions
  repeat words; TF-IDF's sublinear tf and cosine normalization handle that slightly better
  than BM25's saturation. The plan's assumption "BM25 > TF-IDF" doesn't hold here.
- **RRF vs weighted:** the 4-point dev gap (0.564 → 0.605) shrinks to 0.5 point on test. α
  was tuned on dev, so part of that gap was tuning.
- **WordNet** is neutral; **word2vec** is significantly worse than BM25 (−9.7 points,
  p = 0.004). See `docs/phase4_review.md` §3b–3c.

### nDCG@10 by subject (100 test questions)

| Subject | n | S1 | S2 | S3 | S4 | S5 | S6 |
|---|---|---|---|---|---|---|---|
| math | 32 | 0.386 | 0.378 | 0.513 | 0.532 | 0.573 | 0.226 |
| science | 30 | 0.578 | 0.544 | 0.685 | 0.675 | 0.644 | 0.465 |
| french | 19 | 0.318 | 0.250 | 0.338 | 0.366 | 0.464 | 0.169 |
| history | 8 | 0.445 | 0.352 | 0.730 | 0.773 | 0.640 | 0.352 |
| physics | 4 | 0.504 | 0.597 | 0.750 | 0.750 | 0.533 | 0.408 |
| chemistry | 3 | 0.315 | 0.581 | 0.144 | 0.273 | 0.544 | 0.462 |
| geography | 3 | 0.662 | 0.662 | 0.810 | 0.810 | 0.767 | 0.710 |

- **French grammar is the hardest subject** for every system (best 0.464). Questions quote
  the student's sentence ("pour lequel ou pour lesquelles ?"), whose words are not the
  grammar rule's name ("pronom relatif"). The reranker helps most there (+10 points), as it
  reads the question and the page together.
- **Math** needs the semantic branch (+13 points over BM25): numbers and formulas in the
  question match nothing lexically.
- Subjects with under 10 questions are anecdotal.

---

## 3. Significance (paired, per-question nDCG@10, 100 test questions)

| Comparison | Diff | 95% CI (bootstrap) | Wins / losses / ties | p bootstrap | p Wilcoxon | Significant |
|---|---|---|---|---|---|---|
| S3 vs S2 (semantic vs BM25) | **+0.134** | [+0.066, +0.202] | 40 / 11 / 49 | <0.001 | <0.001 | **yes** |
| S6 vs S2 (word2vec vs BM25) | **−0.097** | [−0.164, −0.033] | 15 / 32 / 53 | 0.004 | 0.007 | **yes** |
| S4 vs S3 (hybrid vs semantic) | +0.015 | [−0.005, +0.038] | 12 / 9 / 79 | 0.16 | 0.20 | no |
| S5 vs S4 (reranker) | +0.010 | [−0.042, +0.063] | 25 / 25 / 50 | 0.72 | 0.92 | no |
| S1 vs S2 (TF-IDF vs BM25) | +0.015 | [−0.031, +0.063] | 23 / 21 / 56 | 0.54 | 0.69 | no |
| S4 vs S4 RRF | +0.005 | [−0.035, +0.046] | 17 / 15 / 68 | 0.80 | 0.84 | no |
| S2W vs S2 (WordNet) | −0.003 | [−0.024, +0.021] | 5 / 10 / 85 | 0.78 | 0.38 | no |

"Significant" = both tests under 0.05.

- Only the two large effects are significant. The 1–2-point effects (hybrid, reranker,
  TF-IDF vs BM25) are consistent in sign on dev and test, but 100 questions cannot separate
  them from chance. Running the full 2,281 would be needed to settle them
  (`python -m eval.run_eval --split test`, ~25 min with the reranker).
- **The reranker is the most unstable component:** 25 questions better, 25 worse. It changes
  many rankings, in both directions; the hybrid changes few (79 ties) and mostly helps.

---

## 4. Ablations

Each row changes one preprocessing choice; everything else as in S2 / S3. The baselines
reproduce S2 and S3 exactly, which checks the ablation script.

| Branch | Variant | nDCG@10 dev (300) | Δ | nDCG@10 test (100) | Δ |
|---|---|---|---|---|---|
| BM25 + typo | **baseline** (lemmas, no stopwords, title prefix) | **0.434** | | **0.427** | |
| BM25 + typo | raw words (no lemmatization) | 0.397 | −3.7 | 0.413 | −1.4 |
| BM25 + typo | stopwords kept | 0.365 | **−6.9** | 0.344 | **−8.3** |
| BM25 + typo | no title prefix | 0.416 | −1.8 | 0.412 | −1.5 |
| Semantic | **baseline** (title prefix) | **0.587** | | **0.561** | |
| Semantic | no title prefix | 0.542 | **−4.5** | 0.519 | **−4.2** |

- **Every choice helps, in the same direction on both question sets.**
- **Stopword removal matters most for BM25** (−7 to −8 points without it). The questions
  are chatty ("Bonjour, je voudrais savoir si…"): kept stopwords make every page match
  every question a little, and BM25's IDF doesn't fully cancel that.
- **Lemmatization** is worth 1.4–3.7 points: *conjugue / conjuguer*, *fractions / fraction*
  meet in one term.
- **The title prefix** (an addition to the original plan) is worth 1.5–1.8 points for BM25
  and **4.2–4.5 for the semantic branch**. A chunk from the middle of a page often doesn't
  say what the page is about; the title gives the transformer that context. It's the
  cheapest gain in the project: one line in `chunk.py`.

---

## 5. Error analysis: five failures

20 of the 100 test questions have no relevant page in S5's top 10. Five of them, chosen to
show different causes:

**1. Judgments are incomplete — "C'est quoi la différence entre la forme canonique et la
forme factorisée"** (math, test-22081). Every system returns *Les formes d'écriture de la
fonction polynomiale de degré 2* first or second: exactly the answer. The only page judged
relevant is the broader *La fonction polynomiale de degré 2*. Alloprof's judgments are the
links a teacher put in the answer, so a correct page that wasn't linked counts as wrong.
Same pattern in test-22501 ("comment conjuguer les verbes aimer, jouer, aller…"): the
systems return *Trucs pour conjuguer* and *Conjugaison*; only *Le verbe aller* is judged.
**Cause: the measure, not the system.** The real scores are somewhat higher than reported.

**2. The reranker removes a good page — ppm concentration** (science, test-14558). A long
word problem: prepare 1 L of fertiliser solution at 200 ppm. S3 and S4 rank the relevant
*Les calculs de la concentration en ppm* 4th; the reranker pushes it out of the top 10 and
puts *Les techniques de préparation des solutions* first. It reads the story ("préparer une
solution") rather than the calculation asked for. **Cause: reranker misled by the
narrative.** It is the 25-losses side of §3.

**3. Chemistry notation — "Un échantillon de Ca(ClO2)2 contient 5,7E+19 atomes
d'oxygène…"** (chemistry, test-16879). The answer needs Avogadro's number, which the
question never names. BM25 matches *échantillon* to statistics pages (*méthodes
d'échantillonnage*): a polysemy error. The semantic branch matches "masse" and returns
*masse molaire*, *masse volumique*: close, wrong. Fusion weights semantic at 0.7, so BM25's
rank-5 hit is lost. Oddly, word2vec ranks the right page 1st. **Cause: the link
atoms → mole is reasoning, not similarity.**

**4. A riddle, not a question — "qui est plus petit qu'une maison"** (French, test-6632).
Expected: *Le suffixe* (maison → maisonnette). Every system takes it literally: villages,
atoms, comparing solids. **Cause: the question is an exercise statement**; finding its
topic needs to know it's a vocabulary exercise. No retrieval system here can do that.

**5. No text at all — "Comment on fait se probleme? Merci d'Avance https://…png"**
(science, test-2292; also physics test-4914). The question is in a screenshot. **Cause: no
information in the query.** 2 of the 20 failures are like this. Unsolvable without OCR.

**Summary of the 20 failures.** Most are not ranking bugs: questions whose words are the
exercise and not the topic (grammar sentences, word problems, a riddle), image-only
questions, and correct pages missing from the judgments. The team corpus failures will
complete this section once its queries are judged.

---

## 6. Course concepts and where they are in the project

| Course concept | Module | What it does here | Measured in |
|---|---|---|---|
| Tokenization, lemmatization (spaCy) | `search/lexical.py` `lexical_prep` | Same pipeline for chunks and queries | Ablation §4 |
| Stopwords | `search/lexical.py` | Removed from the lexical form | Ablation §4 |
| Normalization (case, accents) | `search/lexical.py` `strip_accents` | *développé* → *developpe* | — |
| Edit distance (Levenshtein) | `search/spelling.py` | Typo correction against the corpus vocabulary | +1.6–1.8 pts lexical (phase 4 §2) |
| Bag of words, term–document matrix, TF-IDF | `search/lexical.py` `build_tfidf` | S1, cosine on sublinear TF-IDF | S1 |
| Probabilistic weighting (BM25) | `search/lexical.py` `build_bm25` | S2 | S1 vs S2 |
| Synonymy, semantic database (WordNet) | `search/wordnet_expand.py` | Query expansion with French WordNet | S2W, phase 4 §3b |
| Taxonomy similarity (Wu–Palmer) | `wordnet_expand.wu_palmer_fr` | Report examples (guerre/conflit 1.0) | phase 4 §3b |
| Distributional hypothesis, static dense vectors (word2vec) | `search/word2vec.py` | S6, skip-gram trained on the corpus; neighbours | S6, phase 4 §3c |
| Contextual dense vectors (transformers) | `search/semantic.py` | S3, BGE-M3 embeddings, cosine | S3 |
| Combining representations | `search/fusion.py` | S4, weighted min-max fusion and RRF | S4 vs S3 |
| Cross-encoder (query and passage read together) | `search/rerank.py` | S5, bge-reranker-v2-m3 | S5 vs S4 |
| Evaluation (Recall, MRR, nDCG), significance | `eval/` | This document | §2–3 |

---

## 7. Known limitations

- 100 test questions: only large effects are significant (§3).
- Alloprof judgments are incomplete (§5, failure 1): scores are a lower bound.
- The team corpus (10 documents) has no judged queries yet: part B of the evaluation and
  the per-type (A/B/C/D) tables wait for them. `run_eval --dataset team` is ready.
- α and WordNet settings were tuned on dev; dev numbers are slightly optimistic.

## 8. How to reproduce

```powershell
.\.venv\Scripts\python.exe -m eval.run_eval --split test --sample 100 --systems S1 S2 S2W S3 S4rrf S4 S5 S6
.\.venv\Scripts\python.exe -m eval.significance eval/results/alloprof_test_sample100
.\.venv\Scripts\python.exe -m eval.ablations
```
