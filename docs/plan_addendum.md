# Plan addendum

This file records the changes to the original plan (`Plan_Moteur_Recherche_Documentaire.pdf`).
Where they disagree, **this file wins**.

- **Part 1** — Optional components from TALN Chapter 3 (WordNet, word2vec)
- **Part 2** — Multi-domain scope after the professor's feedback: corpus, datasets,
  evaluation

---

# Part 1 — Optional components from TALN Chapter 3

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

---

# Part 2 — Multi-domain scope (after the professor's feedback)

The professor confirmed the subject is a **general** *moteur de recherche documentaire
intelligent*: it must search any kind of document, not one domain such as CVs and job
offers. The architecture already fits; this part changes the corpus, the test data, the
evaluation, and adds one indexing feature.

## 2.1 Summary of changes

| Topic | Original plan | Now |
|---|---|---|
| Corpus | 150–300 documents from **one** coherent domain (e.g. CVs and offers) | Documents from **several domains**, short and long, mixed formats |
| Main data source | Collected or written by the team | **Alloprof** benchmark (public, 13 subjects), plus the team's own mixed-format corpus |
| Relevance judgments | ~40 queries judged by hand | **Alloprof's existing judgments** (2,316 test questions) + ~40 hand-judged queries on the team corpus |
| Chunk context | Chunk text alone | **Document title prefixed** to each chunk for indexing (new option + ablation) |
| Index size | ~1,500 chunks | ~6,000–7,000 chunks: build on the GPU machine, share `index/` |
| Interface | Mode, α, options | + filter by collection/folder and file type |
| CV fixtures | The whole test corpus | One theme among several; more document types added |

Unchanged: the hybrid architecture (TF-IDF, BM25, BGE-M3, fusion, reranker), typo
correction, the optional WordNet/word2vec components, the phases, and the Phase 0–1 code.

## 2.2 Datasets evaluated on Hugging Face

Searched on 23 November 2026, via the Hugging Face API, for French datasets with documents,
queries and relevance judgments.

| Dataset | Content | Licence | Decision |
|---|---|---|---|
| **`lyon-nlp/alloprof`** | 2,556 French course pages (13 subjects); 2,316 test + 9,264 train student questions with relevant pages | Apache 2.0 | **Main dataset**: corpus and evaluation |
| `mteb/MIRACLRetrieval_fr_top_250_only_w_correct-v2` | 75,357 French Wikipedia passages; 343 real questions with judgments | Apache 2.0 (MIRACL) | **Optional** second benchmark (general knowledge) |
| `Shitao/MLDR` (`mldr-v1.0-fr`) | Long Wikipedia articles; questions generated by an LLM | MIT | Rejected: 130 MB compressed, synthetic questions |
| `lyon-nlp/mteb-fr-retrieval-syntec-s2p` | 90 articles of a collective agreement; 100 questions | Not stated | Rejected: too small, unclear licence |
| `vidore/vidore_v3_finance_fr_mteb_format` | Financial report pages, 320 questions | CC BY 4.0 | Rejected: pages are **images** (needs OCR, out of scope) |
| `maastrichtlawtech/bsard` | 22,633 Belgian law articles, 222 test questions | CC BY-NC-SA 4.0 | Rejected for now: single domain, non-commercial share-alike licence |
| `Data-Gouv-ML/les-offres-diffusees-sur-choisir-le-service-public` | Public-sector job offers | Licence Ouverte 2.0 | Rejected: 98% of offers have no description text |

### Alloprof in numbers

Measured on the full document set:

| Subject | Pages | Subject | Pages |
|---|---:|---|---:|
| French | 582 | Geography | 84 |
| Math | 498 | Chemistry | 71 |
| History | 435 | Financial education | 29 |
| Science | 422 | Tips | 23 |
| English | 206 | Revision | 21 |
| Physics | 93 | Native communities | 4 |
| Contemporary world | 88 | | |

- 1.39 million words in total; median page **398 words**, 90th percentile 1,152, maximum
  about 2,400. Long enough to exercise chunking, unlike the CV fixtures.
- Expected index: **~6,000–7,000 chunks** at 250 words. From the Phase 0 benchmark, BGE-M3
  needs about **6–7 minutes on the RTX 2060** and about **2 hours on CPU**. The index is
  built once on the GPU machine and shared.
- Questions are written by students and are **noisy**: typos, missing accents, chatty
  openings (*"Bonjour! Je voulais d'aide…"*), and different wording from the page
  (*"est-ce que un adverbe est quelque chose que tu peut pas mimer?"*). They test exactly
  the plan's type-C (no shared word) and type-D (noisy) cases, on real data.
- Each question lists one or more relevant page ids (binary relevance), and its subject.
- **"English" subject:** these pages teach English and are partly written in English,
  while the lexical branch lemmatizes in French. They are **excluded by default**
  (option to keep them), which leaves **2,350 pages**.

## 2.3 Corpus: what the system indexes

Two collections, kept in separate folders so the interface can filter by collection:

| Collection | Folder | Content | Purpose |
|---|---|---|---|
| **Alloprof** | `data/corpus/alloprof/` | 2,350 course pages, written as `.txt` files (title + text) | Scale, multi-domain, real evaluation |
| **Team corpus** | `data/corpus/divers/` | 50–100 documents in PDF, DOCX, TXT, HTML, CSV from 5–8 themes: administrative procedures, news, reports, scientific abstracts, laws, CVs and offers… | Mixed formats, long multi-page PDFs, the demo |

`scripts/fetch_alloprof.py` (new) downloads Alloprof once, writes the pages as files, and
converts the questions into our evaluation format (`eval/alloprof/queries.json`,
`qrels.csv`).

The synthetic fixtures (`tests/fixtures/corpus/`) gain **5–6 non-CV documents**, among them
one long multi-page report (10+ pages), so chunking and header removal are tested on long
documents too.

## 2.4 New indexing option: title prefix

In a long document, a chunk often starts mid-topic (*"Cette méthode permet…"*, *"Il ou elle
conseille…"*) and alone it has lost its subject. Each chunk gets a context line built from
the document title:

```
text_index = f"{title}\n{text_raw}"     # used for BM25, TF-IDF and embeddings
text_raw                                # unchanged, used for snippets
```

- New setting `CHUNK_TITLE_PREFIX = True` in `search/config.py`.
- Ablation: with vs without prefix, nDCG@10 on Alloprof.
- Titles are cleaned first: a PDF metadata title that looks like a software name or a path
  (*"Microsoft Word - cv_final.docx"*) falls back to the file name.

## 2.5 Evaluation protocol

| | **A · Alloprof** | **B · Team corpus** |
|---|---|---|
| Queries | 2,316 test questions (minus those whose relevant pages are all in the excluded "English" subject) | ~40 written by the team, types A/B/C/D as in the plan |
| Judgments | Provided (binary) | Pooled top-10 of every system, judged 0/1/2 by hand |
| Tuning (α, title prefix) | **A sample of 300 train questions**, never the test questions | 15 dev queries |
| Metrics | Recall@10, MRR@10, nDCG@10, + per subject | Recall@10, MRR@10, nDCG@10, + per query type |
| What it proves | The engine works at scale on real, noisy questions across 12 subjects | Mixed formats, long PDFs, the demo cases |

Notes:
- Alloprof's train questions give a **clean tuning set**: α and options are tuned on train,
  measured once on test. That fixes the plan's weakest point, tuning on only 15 dev queries.
- With 2,316 test queries, differences between systems become measurable. A paired test
  (per-query nDCG, Wilcoxon or bootstrap) can say whether S4 beats S3 for real.
- Running five systems × 2,316 queries with the reranker is the slowest step (cross-encoder
  on 20 chunks per query). If too slow, report the reranker row on a fixed random sample of
  500 test queries, and say so.
- **Error analysis** keeps the plan's five hand-analysed failures, now drawn from both
  collections.

## 2.6 Formats (decided)

"Any type of document" means **any topic**. The supported file formats are fixed to five:

| Format | Extensions | Library | How it becomes text |
|---|---|---|---|
| PDF | `.pdf` | PyMuPDF | Page by page, text blocks as paragraphs; scans skipped |
| Text | `.txt` | standard library | UTF-8, then Windows-1252, then Latin-1 |
| Word | `.docx` | python-docx | Body paragraphs and tables, in order |
| HTML | `.html`, `.htm` | BeautifulSoup | Navigation, scripts, footers dropped |
| **CSV** | `.csv` | standard library `csv` | **New.** One paragraph per row, written as `column : value` pairs |

Other formats (`.md`, `.odt`, `.pptx`, `.rtf`, `.doc`, `.xlsx`…) are listed as skipped
("unsupported format") in the interface, not indexed.

### CSV design

A CSV file is a table, not prose, so it needs its own conversion:
- **Delimiter** detected with `csv.Sniffer` among `;` `,` tab `|`. French Excel exports
  use `;`.
- **Encoding**: same fallback as TXT, since Excel on Windows writes Windows-1252.
- **Header row** detected with `csv.Sniffer.has_header`. When present, each row becomes
  `Colonne1 : valeur1 ; Colonne2 : valeur2 ; …`. The column names give each value its
  meaning, for both BM25 and the embedding model. Empty cells are skipped.
- **One row = one paragraph.** The chunker then packs rows into ~250-word chunks, so a
  result points to a group of neighbouring rows.
- **Page** is 1 for the whole file.
- **Limit:** a CSV of numbers (statistics, measurements) carries little searchable text.
  The engine finds it through its column names and labels, not through its figures.
  Stated in the report.

## 2.7 Language

The target stays **French**. BGE-M3 is multilingual, so an English document is still found
by the semantic branch, but the lexical branch (French spaCy lemmas and stopwords) handles
it poorly. This is stated as a limit in the report. Per-document language detection is an
optional extra.

## 2.8 Interface additions

- Sidebar filter: **collection** (Alloprof / team corpus / all) and **file type**.
- Result card: collection, file type, and, for Alloprof, the subject.
- The corpus panel shows counts per collection.

## 2.9 Phase 1 follow-up: done

Implemented and tested (38 tests):

| Item | Result |
|---|---|
| `scripts/fetch_alloprof.py` | 2,350 pages in 12 subjects; **2,281 test** questions (35 dropped: all their relevant pages are in the excluded "English" subject) and **300 dev** questions sampled from train (seed 42); 3,297 judgments |
| Index on Alloprof | 2,350 documents → **7,074 chunks** in ~5 s; median chunk 236 words; longest page (7,525 words) → 36 chunks |
| Folder metadata | Optional `.metadata.json` per folder: title + extra fields (`meta`), e.g. the Alloprof subject |
| Chunk record | Gains `collection` (first sub-folder), `format`, `text_index`, `meta` |
| Title prefix | `CHUNK_TITLE_PREFIX = True`: `text_index = title + "\n" + text_raw`; `text_raw` unchanged for snippets |
| Title clean-up | Junk metadata titles (*Microsoft Word - …*, paths, *untitled*) fall back to the file name; a TXT file's first line becomes the title when it stands alone |
| Fixtures | `divers/`: 11-page report, DOCX procedure with a table, news article, scientific abstract, decree excerpt, 2 CSVs |

**Dataset finding: Alloprof pages are flat text.** No page has paragraph breaks, and
headings run into the text without punctuation (*"…Univers vivant Les systèmes
biologiques L'anatomie…"*). Chunking therefore relies on sentence boundaries, with
250-word windows where punctuation is missing. Snippets from Alloprof will show no
paragraph structure. This is a property of the dataset, to mention in the report.

## 2.10 Phase 2: done

Lexical branch and first page, reviewed in `docs/phase2_review.md`. Alloprof: 7,074 chunks,
46,082 terms. Quick check on the 300 dev questions: Recall@10 **0.620 (TF-IDF)** and
**0.590 (BM25)**. BM25 being behind is to be confirmed on the test questions in Phase 5.

Two items move to Phase 4's typo correction:
- **accents**: a word typed without accents is not lemmatized by spaCy (*infirmiere* ≠
  `infirmier`), and Alloprof questions often lack accents
- possibly **LaTeX** left in Alloprof pages, if Phase 5 shows it hurts

## 2.11 Phase 3: done

Semantic branch with BGE-M3, reviewed in `docs/phase3_review.md`. The index now holds both
collections: 2,360 documents → 10,295 chunks, embedded in 214 s on the RTX 2060.

| Dev questions (300) | Recall@10 | MRR@10 |
|---|---|---|
| TF-IDF | 0.618 | 0.393 |
| BM25 | 0.582 | 0.379 |
| **Semantic** | **0.733** | **0.553** |

Relevant page in the top 10 from the semantic mode only: 65 questions; BM25 only: 18. This
is the gain the Phase 4 hybrid has to capture.

`EMBED_MAX_TOKENS` raised to 1,024: Alloprof's LaTeX made 7% of chunks exceed 512 tokens.
It made no measurable difference to the results, but no text is lost now.

The team corpus (`data/corpus/divers/`) has its first 10 documents.

## 2.12 Phase 4 (core): done

Hybrid fusion, reranker and typo correction, reviewed in `docs/phase4_review.md`. Tuned on
the 300 dev questions:

| System | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|
| S2 · BM25 + typo | 0.606 | 0.395 | 0.434 |
| S3 · Semantic | 0.733 | 0.553 | 0.584 |
| S4 · Hybrid RRF | 0.741 | 0.529 | 0.564 |
| **S4 · Hybrid weighted, α = 0.7** | 0.765 | 0.573 | 0.605 |
| **S5 · + reranker** | 0.787 | 0.584 | 0.622 |

Decisions: **α = 0.7** (flat top 0.6–0.8; the plan's 0.5 is 2 points lower); weighted
fusion over RRF; typo correction applied to the BM25 query only. WordNet and word2vec
(optional) not started.

## 2.13 Impact on the phases

| Phase | Change |
|---|---|
| 1 · Foundations | **Small follow-up:** non-CV fixtures (incl. a 10+ page report), title clean-up, `CHUNK_TITLE_PREFIX`, `scripts/fetch_alloprof.py`; the chunk record gains `collection` and `format` |
| 2 · Lexical + interface | Index Alloprof + team corpus; collection filter in the page |
| 3 · Semantic | Embed ~7,000 chunks on the GPU machine; share `index/` |
| 4 · Hybrid + polish | Tune α and the title prefix on Alloprof **train** questions |
| 5 · Evaluation + report | Two-part evaluation (A Alloprof, B team corpus), per-subject and per-type tables, significance test |
| Data track | Lighter: ~40 hand-judged queries on the team corpus instead of all judgments; collect the 50–100 mixed-format documents |
