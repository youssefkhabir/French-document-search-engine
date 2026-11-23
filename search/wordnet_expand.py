"""OPTIONAL (course Ch. 3) · WordNet query expansion for the lexical branch.

Uses NLTK WordNet + Open Multilingual WordNet (French: lang="fra").
See docs/plan_addendum.md, component A.
"""


def synonyms_fr(lemma: str, max_terms: int) -> list[str]:
    """French synonyms of a lemma from its WordNet synsets."""
    raise NotImplementedError("Phase 4 stretch")


def expand_query(query_lemmas: list[str], max_terms: int, original_repeat: int) -> list[str]:
    """Return original lemmas (repeated) + expansion terms, ready for lexical_prep."""
    raise NotImplementedError("Phase 4 stretch")


def wu_palmer_fr(word1: str, word2: str) -> float | None:
    """Max Wu-Palmer similarity over the synset pairs of two French words (report illustration)."""
    raise NotImplementedError("Phase 4 stretch")
