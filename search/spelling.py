"""Phase 4 · Query typo correction with Levenshtein distance (course technique: distance d'édition).

Only words of >= TYPO_MIN_LEN letters, absent from the corpus vocabulary (or present but
missing accents that a 20x more frequent spelling has), without digits
or capitals (protects acronyms like ISO and names) are corrected. Comparison is done
accent-free; ties go to the more frequent word. Returns the corrected query and the
list of corrections to show in the interface ("Recherche corrigée : ...").

Two steps before the edit distance:
- a query written mostly in capitals ("BONJOUR J'AURAIS BESOIN…") is lowercased first:
  the capitals rule would otherwise protect every word, and the embedding model handles
  capitals badly (Phase 3 finding)
- a word that only lacks its accents ("infirmiere") gets them back at distance 0: spaCy
  cannot lemmatize the accent-free form (Phase 2 finding)
"""
import json
import re
from collections import Counter
from pathlib import Path

from search import config
from search.lexical import strip_accents

RE_WORD = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*")   # letters, with inner ' or -
RE_TOKEN = re.compile(r"\w+|\W+")                       # words and the text between them
CAPS_MIN_LETTERS = 8      # a query needs this many letters to be judged "in capitals"
CAPS_SHARE = 0.6          # share of capital letters above which the query is lowercased


def build_vocabulary(texts_raw: list[str]) -> dict[str, int]:
    """Word -> frequency over the corpus, lowercased, accents kept. Words of 2+ letters."""
    counts: Counter = Counter()
    for text in texts_raw:
        for w in re.findall(r"[^\W\d_]{2,}", text.lower()):
            counts[w] += 1
    return dict(counts)


def save_vocabulary(vocab: dict[str, int], path: Path) -> None:
    path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")


def load_vocabulary(path: Path) -> dict[str, int]:
    return json.loads(path.read_text(encoding="utf-8"))


class Speller:
    """Holds the vocabulary in the forms the correction needs. Build once per index."""

    def __init__(self, vocab: dict[str, int], min_freq: int = config.TYPO_MIN_FREQ):
        self.vocab = vocab
        # Accent-free form -> its most frequent accented spelling ("infirmiere" -> "infirmière").
        best: dict[str, tuple[int, str]] = {}
        for w, n in vocab.items():
            key = strip_accents(w)
            if n > best.get(key, (0, ""))[0]:
                best[key] = (n, w)
        self.by_plain = {k: w for k, (_, w) in best.items()}
        # Candidates for the edit distance: frequent enough not to be typos themselves.
        self.candidates = [k for k, (n, _) in best.items()
                           if n >= min_freq and len(k) >= config.TYPO_MIN_LEN - 1]
        self.freq = {k: n for k, (n, _) in best.items()}

    def correct_word(self, word: str) -> str | None:
        """Corrected spelling of one lowercase word, or None to keep it."""
        from rapidfuzz import process
        from rapidfuzz.distance import Levenshtein

        plain = strip_accents(word)
        if word in self.vocab:
            # The corpus has its own accent-free misspellings ("francaise" 6 times against
            # 347 "française"): keep the word unless the accented spelling dominates.
            fixed = self.by_plain.get(plain, word)
            if fixed != word and self.vocab[fixed] >= config.TYPO_ACCENT_RATIO * self.vocab[word]:
                return fixed
            return None
        if plain in self.by_plain:                           # only the accents differ
            fixed = self.by_plain[plain]
            return fixed if fixed != word else None
        max_d = (config.TYPO_MAX_DIST_SHORT if len(plain) < 8 else config.TYPO_MAX_DIST_LONG)
        matches = process.extract(plain, self.candidates, scorer=Levenshtein.distance,
                                  score_cutoff=max_d, limit=20)
        if not matches:
            return None
        # Smallest distance first, then the most frequent word.
        key, _, _ = min(matches, key=lambda m: (m[1], -self.freq[m[0]]))
        return self.by_plain[key]


def _mostly_capitals(query: str) -> bool:
    letters = [c for c in query if c.isalpha()]
    return (len(letters) >= CAPS_MIN_LETTERS
            and sum(c.isupper() for c in letters) / len(letters) >= CAPS_SHARE)


def fix_capitals(query: str) -> str:
    """The query lowercased if it is mostly capitals, else unchanged."""
    return query.lower() if _mostly_capitals(query) else query


def correct_query(query: str, speller: "Speller | dict[str, int]") -> tuple[str, list[tuple[str, str]]]:
    """Return (corrected query, [(original, corrected)]). A lowercased capitals query is
    reported as one correction of the whole query."""
    if isinstance(speller, dict):
        speller = Speller(speller)
    corrections: list[tuple[str, str]] = []
    if _mostly_capitals(query):
        lowered = query.lower()
        corrections.append((query, lowered))
        query = lowered

    out = []
    for tok in RE_TOKEN.findall(query):
        fixed = None
        if (RE_WORD.fullmatch(tok) and len(tok) >= config.TYPO_MIN_LEN
                and not any(c.isupper() for c in tok)):
            fixed = speller.correct_word(tok)
        if fixed:
            corrections.append((tok, fixed))
            out.append(fixed)
        else:
            out.append(tok)
    return "".join(out), corrections
