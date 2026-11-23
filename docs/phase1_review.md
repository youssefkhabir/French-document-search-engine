# Phase 1 — Technical review

**Project:** Moteur de recherche documentaire intelligent (French semantic search, TALN course)
**Phase:** 1 · Foundations: extraction, cleaning, chunking
**Date:** 23 November 2026
**Commit:** `d5eed64` on `main`
**Status:** done on the synthetic test corpus; 23 tests pass. Chunk sizes and
header detection still need a check on the professor's real data.

---

## 1. What Phase 1 delivers

The plan's "done when" criterion: *a folder of files becomes a table of clean chunks you
can read.*

```
data/corpus/  ──extract──▶  pages  ──remove headers/footers──▶  ──clean──▶  ──chunk──▶  index/chunks.parquet
(PDF DOCX TXT HTML)         (raw text, per page)                 (regex)       (sentences)   + skipped.json + meta.json
```

| File | Role | Course technique |
|---|---|---|
| `search/extract.py` | Read the four formats, page by page for PDF | — |
| `search/clean.py` | Normalize text, rejoin hyphens, drop headers/footers | **Expressions régulières** |
| `search/chunk.py` | Split into sentences, pack into overlapping chunks | **Segmentation en phrases** |
| `search/engine.py` | `ingest()` chains the three; `build_index()` saves the files | — |
| `scripts/build_index.py` | Command line: `python scripts/build_index.py --corpus <folder>` | — |
| `scripts/make_fixtures.py` | Generates the synthetic test corpus | — |
| `tests/test_phase1.py` | 23 tests | — |

**Result on the test corpus:** 16 files → 15 documents, 18 pages → 16 chunks, 1 file
skipped (the scan). The run takes about 2 seconds.

---

## 2. Test data

### Why synthetic

The search for real data found nothing usable:
- **CVs:** no French CV dataset exists on Hugging Face, which makes sense, since real CVs
  are personal data.
- **Job offers:** the French government's open data on public-sector offers
  (*Choisir le Service Public*, Licence Ouverte 2.0) has about 130,000 offers, but only
  **2 in 100** have any description text. The rest is metadata (title, place, dates),
  about 30 words per offer: too thin to test retrieval.

So `scripts/make_fixtures.py` writes **10 fictional CVs and 6 fictional job offers**. All
names, companies and addresses are invented (*Camille Fictif*, *Société Exemple
Numérique*…), so the files can be public.

### What the corpus tests on purpose

| File | Format | Hard case it carries |
|---|---|---|
| `cv_scanne_agent_accueil.pdf` | PDF | Image only, no text layer → must be **skipped** |
| `offre_rssi_ministere.pdf` | PDF, 3 pages | Repeated header + `Page n / 3` footer, 2 line-end hyphens, 345 words → **2 chunks** |
| `cv_technicien_reseaux.pdf` | PDF, 2 pages | Header and footer with the page number inside, 3 line-end hyphens |
| `offre_developpeur_backend.html` | HTML | `<nav>`, `<script>`, `<style>`, `<aside>`, `<footer>`, bold labels inside sentences |
| `cv_chargee_communication.html` | HTML | Same, as a CV web page |
| `cv_professeure_maths.txt` | TXT | Encoded in **Windows-1252**, not UTF-8 |
| `cv_auditrice_iso27001.docx` | DOCX | A page header that must not be indexed |
| the other one-page CVs/offers | all | Must stay **one chunk** each |

It also plants **retrieval pairs** for Phases 2–5. For example,
`cv_ingenieur_logiciel.pdf` says *ingénieur logiciel* and never *développeur*, while
`offre_developpeur_backend.html` asks for a *développeur*. That is the plan's type-C
case. `cv_auditrice_iso27001.docx` vs `offre_rssi_ministere.pdf` is the exact-term case
(*ISO 27001*).

### How the generator makes the hard cases

```python
def write_pdf(path, pages, title, fontsize=10.5):
    doc = pymupdf.open()
    for text in pages:
        text = text.replace("œ", "oe")        # built-in Helvetica has no œ glyph
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        page.insert_textbox(rect, text, fontsize=fontsize, fontname="helv")
```

`insert_textbox` breaks lines where the source text has `\n`. So writing `traite-\nment`
in the content produces a real hyphen at the end of a PDF line, which extraction must
undo.

```python
def write_scanned_pdf(path, text):
    ...                                              # draw the text on a page
    pix = page.get_pixmap(dpi=80, colorspace=pymupdf.csGRAY)
    p.insert_image(p.rect, stream=pix.tobytes("jpeg", jpg_quality=60))
```

The scan is the page rendered to an image and pasted alone into a new PDF: it looks like
a CV but contains no text. JPEG at 80 dpi brings it from 2.9 MB to 13 KB.

---

## 3. `extract.py` — reading the files

### Output contract

```python
{"doc_id": "cv/cv_ingenieur_logiciel.pdf",   # path relative to the corpus folder
 "title":  "CV - Ingénieur logiciel",
 "path":   "...", "format": "pdf",
 "page":   1,                                  # one dict per PDF page; 1 for other formats
 "text":   "Paragraph one\nwrapped line\n\nParagraph two"}
```

The raw text keeps **two levels of structure**: `\n\n` between paragraphs and `\n` between
lines inside a paragraph. Cleaning needs the line breaks to find hyphenated words, and
chunking needs the paragraph breaks.

### The folder loop: one bad file never stops the build

```python
for path in sorted(p for p in folder.rglob("*") if p.is_file()):
    rel = path.relative_to(folder).as_posix()
    if path.name.startswith((".", "~$")):        # .gitkeep, Office lock files
        continue
    if path.suffix.lower() not in SUPPORTED:
        skipped.append((rel, f"unsupported format ({path.suffix or 'no extension'})"))
        continue
    try:
        doc_pages = extract_file(path)
    except SkipFile as e:                         # scan, empty document
        skipped.append((rel, str(e)))
        continue
    except Exception as e:                        # corrupt file
        skipped.append((rel, f"unreadable: {type(e).__name__}: {e}"))
        continue
```

- `sorted(...)` makes the order, and so the chunk ids, reproducible between runs.
- Two kinds of failure are kept apart. `SkipFile` is an expected decision (a scan) with a
  clear reason. Any other exception is a broken file. Both go to `skipped.json`, so the
  interface can list them, as the plan requires for scans.
- `doc_id` is the path relative to the corpus folder (`cv/…`, `offres/…`). It is unique
  even if two folders hold a file with the same name.

### PDF: blocks are paragraphs, scans are detected by character count

```python
blocks = page.get_text("blocks", sort=True)
paras = [b[4].strip() for b in blocks if b[6] == 0 and b[4].strip()]
texts.append("\n\n".join(paras))
...
n_chars = sum(len(t) for t in texts)
if doc.page_count and n_chars / doc.page_count < MIN_CHARS_PER_PAGE:   # 30
    raise SkipFile(f"scanned PDF: {n_chars} characters on {doc.page_count} page(s), "
                   "OCR is out of scope")
```

- PyMuPDF groups text into **blocks** by layout. On our files one block is one paragraph,
  and a header or a page number is its own block. That last point is what makes header
  removal possible later.
- `b[6] == 0` keeps text blocks and drops image blocks.
- `sort=True` orders blocks top to bottom, then left to right.
- A scan has 0 characters. The threshold is an average of 30 characters per page, so a
  real PDF with one nearly empty page (a cover) is not mistaken for a scan.

### DOCX: walk the body in order, tables included

```python
for child in doc.element.body.iterchildren():
    tag = child.tag.rsplit("}", 1)[-1]
    if tag == "p":
        text = Paragraph(child, doc).text.strip()
        ...
    elif tag == "tbl":
        for row in Table(child, doc).rows:
            cells = []
            for cell in row.cells:
                t = cell.text.strip()
                if t and t not in cells:      # merged cells repeat their text
                    cells.append(t)
            if cells:
                paras.append(" | ".join(cells))
```

- `doc.paragraphs` alone would **miss tables**, and many CV templates are built with
  tables. Walking the XML body keeps paragraphs and tables in reading order.
- In python-docx, a merged cell returns the same text once per grid column, so
  duplicates are dropped.
- Section headers and footers are not in the body, so *"Curriculum vitae - document
  confidentiel"* is never indexed. A test checks this.

### HTML: drop the noise, then merge inline text

```python
HTML_DROP = ["script", "style", "noscript", "nav", "footer", "aside", "form", "iframe",
             "svg", "button", "template"]
HTML_INLINE = ["a", "b", "strong", "i", "em", "u", "span", ...]

for tag in soup.find_all(HTML_DROP):
    tag.decompose()                    # remove the element and everything inside
for tag in soup.find_all(HTML_INLINE):
    tag.unwrap()                       # keep the text, drop the tag
soup.smooth()                          # merge the text pieces left side by side
lines = [" ".join(line.split()) for line in body.get_text("\n").split("\n")]
```

`get_text("\n")` puts a newline between **every** piece of text. Without the last two
steps, `<p><b>Entreprise :</b> Startup Fictive</p>` came out as two lines,
*"Entreprise :"* and *"Startup Fictive"*. That was a real bug found on the fixture.
`unwrap()` removes the `<b>`, but leaves two separate text pieces; `smooth()` merges them.
After that, each line comes from a block element (`<p>`, `<li>`, `<h2>`), so each becomes
a paragraph.

`<header>` is deliberately **not** dropped: on a CV page it holds the person's name and
job title.

### TXT: try encodings in order

```python
TXT_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
for enc in TXT_ENCODINGS:
    try:
        return raw.decode(enc)
    except UnicodeDecodeError:
        continue
```

- `utf-8-sig` reads UTF-8 and removes the byte-order mark that Notepad adds.
- A Windows-1252 file is **invalid UTF-8** as soon as it has an accent (`é` is byte
  `0xE9`), so it falls through to `cp1252`. The professor's TXT files may well come from
  Windows.
- `latin-1` accepts any byte, so it is the last resort that never fails.

---

## 4. `clean.py` — regular expressions

### `clean_text`

```python
CHAR_MAP = str.maketrans({
    " ": " ", " ": " ", ...,         # non-breaking / thin spaces
    "­": "",                               # soft hyphen
    "’": "'", "‘": "'", "ʼ": "'",  # typographic apostrophes
    "ﬁ": "fi", "ﬂ": "fl", ...,        # PDF ligatures
})

def clean_text(text):
    text = unicodedata.normalize("NFC", text).translate(CHAR_MAP)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _rejoin_hyphens(text)
    paragraphs = []
    for para in re.split(r"\n\s*\n", text):
        para = RE_SPACES.sub(" ", para.replace("\n", " ")).strip()
        if para:
            paragraphs.append(para)
    return "\n\n".join(paragraphs)
```

Each step exists because it breaks matching otherwise:

| Step | Problem it prevents |
|---|---|
| NFC | `é` can be stored as one character or as `e` + an accent mark. They look the same but are different strings, so BM25 would treat them as different words. |
| `’` → `'` | French uses both. *l’expérience* and *l'expérience* must give the same tokens. |
| Non-breaking spaces → space | French typography puts them before `:` `?` `!` and inside numbers (*10 000 €*). |
| Ligatures | PDFs often store *fi* as one character `ﬁ`, so *filière* would not match. |
| Single `\n` → space | Line breaks inside a paragraph are layout, not meaning. |

Accents and capitals are **kept**: the embedding model is trained on natural text. The
lexical form (lowercase, lemmas, no accents) is Phase 2.

### Rejoining hyphenated line ends

```python
RE_HYPHEN_BREAK = re.compile(r"(\w+)-\n[ \t]*(\w+)")

def _rejoin_hyphens(text):
    lowered = text.lower()
    def repl(m):
        left, right = m.group(1), m.group(2)
        compound = f"{left}-{right}".lower()
        if right[0].isupper() or lowered.count(compound) > 0:
            return f"{left}-{right}"          # real compound: keep the hyphen
        return left + right                    # word cut by the layout: join
    return RE_HYPHEN_BREAK.sub(repl, text)
```

The plan's regex `(\w+)-\n(\w+) → \1\2` is right for *traite-/ment* but wrong for a
compound that happens to break at its own hyphen: *aides-/soignants* would become
*aidessoignants*. Two cheap rules separate the cases:
1. **Capital after the cut** (*Wi-/Fi*, *Jean-/Pierre*): a compound, keep the hyphen.
2. **The hyphenated form appears elsewhere in the text on one line**: a compound, keep it.

| Input | Output | Rule |
|---|---|---|
| `traite-\nment` | `traitement` | joined |
| `aides-soignants … aides-\nsoignants` | `aides-soignants … aides-soignants` | rule 2 |
| `Wi-\nFi` | `Wi-Fi` | rule 1 |

**Limit:** a lowercase compound that is cut at its hyphen and appears nowhere else in the
text is still joined wrongly. That is rare, and the error only affects one token.

### Removing headers, footers and page numbers

```python
RE_PAGE_NUMBER = re.compile(r"^[-–\s]*(page\s*)?\d{1,4}(\s*(/|sur|of)\s*\d{1,4})?[-–\s]*$",
                            re.IGNORECASE)

def _key(paragraph):
    return RE_SPACES.sub(" ", RE_DIGITS.sub("#", paragraph.lower())).strip()

def remove_repeated_lines(pages):
    ...
    if n >= 2:
        for paras in split:
            for key in {_key(p) for p in paras}:        # a set: count pages, not lines
                counts[key] = counts.get(key, 0) + 1
        threshold = max(2, math.ceil(REPEAT_MIN_SHARE * n))   # 60% of pages, at least 2
        repeated = {k for k, c in counts.items() if c >= threshold}
    ...
            if _key(p) in repeated or RE_PAGE_NUMBER.match(p.strip()):
                removed.append(p.strip())
```

- **The key replaces digits with `#`.** *"Thomas Fictif - CV - page 1"* and *"… page 2"*
  both become `thomas fictif - cv - page #`, so a header that contains the page number is
  still recognized as repeated.
- **A set per page** means that a line repeated three times on the same page counts once.
  What is counted is "on how many pages does this appear".
- **The 60% threshold** handles documents whose first page has no header (a cover page).
- **Bare page numbers** (`3`, `Page 3`, `3 / 12`, `3 sur 12`, `- 3 -`) are removed even on
  one-page documents.
- The removed lines are **returned**, not just thrown away. `ingest()` keeps them per
  document, and `build_index.py` prints them. That is the before/after evidence for the
  regex section of the report.

On the test corpus:

```
offres/offre_rssi_ministere.pdf: removed 6 lines, e.g. "Ministère Imaginaire de l'Exemple - Offre n° 2026-114", "Page 1 / 3"
cv/cv_technicien_reseaux.pdf:    removed 4 lines, e.g. "Thomas Fictif - CV - page 1", "thomas.fictif@example.org - 1"
```

---

## 5. `chunk.py` — sentence segmentation and chunking

### The sentence splitter

```python
ABBREVIATIONS = {"m", "mme", "dr", "etc", "cf", "ex", "art", "n", "no", "tél", "réf",
                 "janv", "févr", "sept", "oct", ...}

RE_BOUNDARY = re.compile(r"([.!?…]+[»\")\]]*)\s+(?=[«\"(\[]?[A-ZÀ-ÖØ-Þ0-9])")
RE_WORD_BEFORE_DOT = re.compile(r"(\w+)\.$")

def split_sentences(text):
    sentences, start = [], 0
    for m in RE_BOUNDARY.finditer(text):
        candidate = text[start:m.end(1)]
        if m.group(1).startswith("."):
            w = RE_WORD_BEFORE_DOT.search(candidate)
            if w and (w.group(1).lower() in ABBREVIATIONS
                      or (len(w.group(1)) == 1 and w.group(1).isupper())):   # "J. Dupont"
                continue                     # not a boundary: keep going
        sentences.append(candidate.strip())
        start = m.end()
    ...
```

How to read the boundary regex:
- `[.!?…]+`: one or more end marks (handles `?!` and `...`)
- `[»\")\]]*`: an optional closing quote or bracket that belongs to the sentence
- `\s+`: at least one space, which also covers the French space before `?` and `!`
- `(?=[«\"(\[]?[A-ZÀ-ÖØ-Þ0-9])`: a lookahead, so the next sentence must **start** with a
  capital (accented capitals included) or a digit, optionally after an opening quote

Then the **abbreviation filter** rejects false boundaries after *M.*, *etc.*, *art.*, and
single initials like *J.*. From the test:

```
"M. Dupont travaille avec Mme Martin, etc. Il est arrivé en 2020. J. Durand est parti ! Pourquoi ? Voir l'art. 3 du contrat."
→ ["M. Dupont travaille avec Mme Martin, etc. Il est arrivé en 2020.",
   "J. Durand est parti !", "Pourquoi ?", "Voir l'art. 3 du contrat."]
```

The first sentence shows a known ambiguity. *etc.* followed by a capital can end a
sentence, but the splitter treats it as an abbreviation and does not split there. For
chunking that is harmless: the worst case is a slightly longer sentence.

**Why a regex and not spaCy's sentence splitter?** spaCy is loaded anyway in Phase 2, but
its parser is slow on a whole corpus and hard to explain. The regex is transparent, fast,
tested, and it *is* the course technique. spaCy remains the tokenizer and lemmatizer for
the lexical form.

### Units: sentences that remember their paragraph

```python
def _units(text, page, max_words):
    """Cut one page into (sentence, n_words, page, starts_paragraph) units."""
    units = []
    for para in text.split("\n\n"):
        first = True
        for sent in split_sentences(para.strip()):
            words = sent.split()
            for i in range(0, len(words), max_words):  # one window unless the sentence is huge
                piece = words[i:i + max_words]
                units.append((" ".join(piece), len(piece), page, first))
                first = False
    return units
```

Each unit is a sentence, carrying its **word count**, its **page** and whether it **starts
a paragraph**. A "sentence" longer than 250 words (a list without punctuation) is cut into
fixed windows, so no unit can exceed the chunk size.

### Packing with overlap

```python
for unit in units:
    size = sum(u[1] for u in current)
    if current and size + unit[1] > max_words and len(current) > n_overlap:
        groups.append(current)                    # close the chunk
        firsts.append(n_overlap)
        tail, tail_words = [], 0                  # take whole sentences from the end...
        for u in reversed(current):
            if tail_words + u[1] > overlap_words:   # ...up to 40 words
                break
            tail.insert(0, u)
            tail_words += u[1]
        while tail and tail_words + unit[1] > max_words:   # leave room for the new sentence
            tail_words -= tail.pop(0)[1]
        current, n_overlap = tail, len(tail)      # next chunk starts with the overlap
    current.append(unit)
```

How it works:
- Sentences are added until the next one would pass **250 words**. Then the chunk is
  closed and the next one **starts with the last sentences of the previous one**, up to
  **40 words**. A requirement stated across a chunk boundary is then complete in at least
  one chunk.
- The overlap is made of **whole sentences**, never half a sentence.
- `len(current) > n_overlap` blocks a chunk made only of overlap, which would just repeat
  its neighbour.
- `firsts` remembers where the *new* content of each chunk starts. A chunk's `page` is the
  page of that new content, not of the overlap, so the page shown in the results points
  where the reader should look.

**Why sentences and not paragraphs?** The first version packed paragraphs. On the RSSI
offer, every paragraph had 60–70 words, which is more than the 40-word overlap, so no
paragraph ever fitted and **chunks had no overlap at all**. A test now checks that each
chunk starts with a sentence of the previous one.

### Two finishing rules

```python
if groups and new_words < min_tail_words:     # 50
    groups[-1].extend(current[n_overlap:])    # tiny last piece: merge into the previous chunk
```

A last chunk of a few words (a closing line such as *"Candidature : lettre de motivation"*)
is useless alone, so it is appended to the previous chunk. That chunk can then reach
**up to ~300 words**, a deliberate trade-off.

```python
def _join(units):
    out = ""
    for k, u in enumerate(units):
        out += u[0] if k == 0 else ("\n" if u[3] else " ") + u[0]
    return out
```

When the chunk text is rebuilt, sentences of the same paragraph are joined with a space
and paragraphs with a newline. Snippets in the interface then read like the original.

### Chunk record

```python
{"chunk_id": "offres/offre_rssi_ministere.pdf#1",   # doc_id + position
 "doc_id":   "offres/offre_rssi_ministere.pdf",
 "title":    "Offre - Responsable de la sécurité des systèmes d'information",
 "page":     2,
 "n_words":  126,
 "text_raw": "Il ou elle conseille la direction ...\nProfil recherché Formation ..."}
```

`text_lex`, the lexical form, is added in Phase 2.

---

## 6. `engine.py` — `ingest()` and `build_index()`

```python
def ingest(folder):
    pages, skipped = extract_folder(folder)
    ...
    for doc_id, doc_pages in by_doc.items():
        doc_pages.sort(key=lambda p: p["page"])
        texts, dropped = remove_repeated_lines([p["text"] for p in doc_pages])   # 1. per document
        if dropped:
            removed[doc_id] = dropped
        for p, text in zip(doc_pages, texts):
            text = clean_text(text)                                             # 2. per page
            if text:
                cleaned.append({**p, "text": text})
    chunks = pd.DataFrame(chunk_pages(cleaned), columns=[...])                  # 3. per document
```

**Order matters.** Headers are removed **before** `clean_text`, because detection compares
paragraph blocks, and cleaning would otherwise merge a header into text. They are removed
per **document**, because "repeated" means repeated across the pages of one file.

`build_index()` writes:

| File | Content |
|---|---|
| `index/chunks.parquet` | The chunk table |
| `index/skipped.json` | `[{"path", "reason"}]` for scans and broken files, for the interface |
| `index/meta.json` | Build time, counts, chunk settings, duration |

The comment `# Phases 2-4 add: lexical forms, TF-IDF, BM25, embeddings, ...` marks where
the next phases extend it. The function's signature does not change.

---

## 7. Tests

`python -m pytest` runs 23 tests in about 4 seconds.

| Group | Tests | What they lock in |
|---|---:|---|
| `clean_text` | 10 | hyphen join, compound kept, capital kept, apostrophe, non-breaking spaces, NFC, ligature, wrapped lines, spaces, paragraphs and accents kept |
| `remove_repeated_lines` | 2 | header + `Page n / 3` removed on 3 pages; one-page document keeps its content, loses only the page number |
| `split_sentences` | 1 | abbreviations, initials, `!` `?` with the French space |
| Chunking | 3 | short document → 1 chunk; long document → size ≤ 300, unique ids, **overlap**, pages; unpunctuated text → 250 / 250 / 100 windows |
| Extraction | 6 | scan skipped; PDF pages and title; HTML noise gone and bold labels merged; cp1252 decoded; DOCX header excluded; unsupported and corrupt files reported |
| End to end | 1 | the whole fixture corpus: 15 documents, 1 skipped, no header/footer text left, hyphenated words joined, RSSI offer = 2 chunks |

The tests use the committed fixture files, so a teammate can run them without generating
anything.

---

## 8. Bugs found during the phase

| Bug | How it was found | Fix |
|---|---|---|
| HTML: *"Entreprise :"* separated from its value | Reading the extracted HTML | `soup.smooth()` after `unwrap()` |
| Chunks had **no overlap** | Comparing the two RSSI chunks | Pack sentences instead of paragraphs |
| Sentences of one paragraph separated by newlines | Reading a CV chunk | `_join()` was written but not called; now used |
| `œ` became `?` in generated PDFs | Reading extracted PDF text | Generator writes `oe` (font limit, not an extraction bug) |
| Scanned fixture weighed 2.9 MB | File listing | Grayscale JPEG at 80 dpi: 13 KB |
| `import fitz` deprecation warning | Generator output | `import pymupdf` everywhere |

---

## 9. Known limitations: check on the real data

These cannot be settled with synthetic files. Each is a risk for the professor's corpus:

| # | Limitation | Why it matters for real CVs | What to do |
|---|---|---|---|
| 1 | **Two-column PDF layouts** | Many CV templates put skills in a side column. `sort=True` orders blocks by position, which can interleave the two columns. | Read the chunks of 3–4 real two-column CVs. If mixed, sort blocks by column (x position) before y. |
| 2 | **DOCX text boxes** | Word CV templates often put content in text boxes or shapes. Their text sits inside drawing elements nested in a paragraph, and python-docx's `Paragraph.text` does not read it, so the current walk misses it. | Count words per real DOCX; a CV with very few words is a sign. Fix by also reading `w:txbxContent` elements. |
| 3 | **PDF metadata titles** | Real PDFs often have titles like *"Microsoft Word - cv_final.docx"* or none. | Fall back to the file name when the title looks like a software name or a path. |
| 4 | **Header detection on 2-page documents** | The threshold is 2 pages, so any paragraph that appears on both pages is removed, a real repeated heading included. | Look at the `removed` lines printed by `build_index.py` for real files. |
| 5 | **Headers inside a paragraph block** | If a PDF puts the header in the same block as body text, it is not detected. | Same check. |
| 6 | **Compound check is per page** | `clean_text` runs per page, so "appears elsewhere" means elsewhere on the same page. | Acceptable; could pass the whole document if needed. |
| 7 | **Chunk size on short documents** | Test documents have a median of 82 words, so most are one chunk, and the 250 / 40 settings are barely exercised. | Re-check the chunk length distribution on the real corpus. |
| 8 | **`doc_id` is the file path** | Renaming or moving a file after relevance judgments breaks `qrels.csv`. | Freeze file names before writing the judgments (the plan already says to freeze the corpus). |

---

## 10. How to review the code yourself

1. Run the tests: `python -m pytest`.
2. Build the index on the test corpus and print chunks:
   `python scripts/build_index.py --corpus tests/fixtures/corpus --show 5`
3. Open a fixture and its chunks side by side. The RSSI offer is the richest: compare
   `tests/fixtures/corpus/offres/offre_rssi_ministere.pdf` with its two chunks.
4. Try your own file: drop any French PDF, DOCX or HTML into a temporary folder and
   build with `--corpus` pointing at it. That's the quickest preview of limitations 1–3.

## 11. Handoff to Phase 2

Phase 2 builds the lexical branch on `chunks.parquet`:
- `lexical_prep()` (spaCy: tokens, lowercase, stopwords, lemmas, accent removal), added
  as a `text_lex` column
- the TF-IDF baseline and the BM25 index, saved to `index/`
- the first Streamlit page with ranked results and snippets

Done when you can type a query and see ranked documents in two modes.
