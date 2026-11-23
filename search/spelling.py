"""Phase 4 · Query typo correction with Levenshtein distance (course technique: distance d'édition).

Only words of >= TYPO_MIN_LEN letters, absent from the corpus vocabulary, without digits
or capitals (protects acronyms like ISO and names) are corrected. Comparison is done
accent-free; ties go to the more frequent word. Returns the corrected query and the
list of corrections to show in the interface ("Recherche corrigée : ...").
"""


def build_vocabulary(texts_raw: list[str]) -> dict[str, int]:
    raise NotImplementedError("Phase 4")


def correct_query(query: str, vocab: dict[str, int]) -> tuple[str, list[tuple[str, str]]]:
    raise NotImplementedError("Phase 4")
