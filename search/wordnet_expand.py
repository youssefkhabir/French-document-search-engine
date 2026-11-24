"""OPTIONAL (course Ch. 3) · WordNet query expansion for the lexical branch.

Uses NLTK WordNet + Open Multilingual WordNet (French: lang="fra").
See docs/plan_addendum.md, component A.

French WordNet is a translation of the English synsets, so a word has many senses and
each sense drags in loosely related words (travailler -> accouchement, calculer -> comte).
Four filters keep the noise down:
  1. only synsets of the query word's part of speech (spaCy NOUN/VERB/ADJ);
  2. candidates ranked by the number of the word's senses they share (a synonym present in
     5 senses of "calculer" is safer than one present in a single rare sense);
  3. no multi-word entries ("faire_la_guerre") and no proper names;
  4. optionally, only terms the corpus contains (an unknown term can't change BM25 scores,
     so it would only waste one of the `max_terms` slots).
"""
from collections import Counter
from functools import lru_cache

from search import config, lexical

# spaCy part of speech -> WordNet POS codes ("s" = satellite adjective).
POS_MAP = {"NOUN": ("n",), "VERB": ("v",), "ADJ": ("a", "s")}


@lru_cache(maxsize=1)
def _wn():
    from nltk.corpus import wordnet as wn
    wn.ensure_loaded()
    return wn


@lru_cache(maxsize=4096)
def _synsets(lemma: str, pos: str | None):
    wn = _wn()
    synsets = wn.synsets(lemma, lang=config.WORDNET_LANG)
    if pos:
        synsets = [s for s in synsets if s.pos() in POS_MAP.get(pos, ())]
    return tuple(synsets)


def synonyms_fr(lemma: str, max_terms: int, pos: str | None = None, known=None) -> list[str]:
    """French synonyms of a lemma from its WordNet synsets, most shared first.

    pos: spaCy part of speech of the word ("NOUN", "VERB", "ADJ"), None = every synset.
    known: optional predicate on the lexical form of a candidate (e.g. "is in the index").
    """
    own = lexical.strip_accents(lemma.lower())
    counts: Counter[str] = Counter()
    for s in _synsets(lemma.lower(), pos):
        for name in set(s.lemma_names(config.WORDNET_LANG)):
            if "_" in name or not name.islower() or not name.isalpha():
                continue
            counts[name] += 1
    out, seen = [], {own}
    # most_common keeps the first-seen order among ties.
    for name, _ in counts.most_common():
        form = lexical.strip_accents(name)
        if (form in seen or form in lexical._stopwords()
                or (known is not None and not known(form))):
            continue
        seen.add(form)
        out.append(name)
        if len(out) >= max_terms:
            break
    return out


def query_words(text: str) -> list[tuple[str, str]]:
    """(lemma with accents, POS) of the content words of a query: nouns, verbs, adjectives."""
    return [(tok.lemma_.lower(), tok.pos_) for tok in lexical._nlp()(text)
            if tok.pos_ in POS_MAP and not tok.is_stop and tok.lemma_.isalpha()]


def expand_query(text: str, max_terms: int | None = None, original_repeat: int | None = None,
                 known=None) -> tuple[str, dict[str, list[str]]]:
    """Expanded lexical query for BM25 / TF-IDF.

    Returns (query_lex, {word: [synonyms added]}). The original lexical form is repeated
    `original_repeat` times so it keeps more weight than the expansions (bm25s has no
    per-term weights, and a repeated term counts again in the score).
    """
    max_terms = config.WORDNET_MAX_EXPANSIONS if max_terms is None else max_terms
    original_repeat = config.WORDNET_ORIGINAL_REPEAT if original_repeat is None else original_repeat
    base = lexical.lexical_prep(text)
    added: dict[str, list[str]] = {}
    extra: list[str] = []
    in_query = set(base.split())
    for lemma, pos in query_words(text):
        if lemma in added:
            continue
        syns = [s for s in synonyms_fr(lemma, max_terms, pos, known)
                if lexical.strip_accents(s) not in in_query]
        if syns:
            added[lemma] = syns
            extra.extend(lexical.strip_accents(s) for s in syns)
    query_lex = " ".join([base] * max(1, original_repeat) + extra)
    return query_lex, added


def wu_palmer_fr(word1: str, word2: str) -> float | None:
    """Max Wu-Palmer similarity over the synset pairs of two French words (report illustration).

    None when either word is not in French WordNet or no pair is comparable (different POS).
    """
    best = None
    for s1 in _synsets(word1.lower(), None):
        for s2 in _synsets(word2.lower(), None):
            sim = s1.wup_similarity(s2)
            if sim is not None and (best is None or sim > best):
                best = sim
    return best
