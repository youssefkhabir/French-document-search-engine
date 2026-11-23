"""Phase 4 · Fuse lexical and semantic candidate lists.

weighted: min-max each list over its own candidates (guard max == min -> all 0),
          absent chunk -> 0, score = alpha * sem + (1 - alpha) * lex.
rrf:      reciprocal rank fusion, kept as one comparison row.
"""


def minmax(scores: dict[int, float]) -> dict[int, float]:
    raise NotImplementedError("Phase 4")


def weighted_fusion(lex: list[tuple[int, float]], sem: list[tuple[int, float]],
                    alpha: float) -> list[tuple[int, float]]:
    raise NotImplementedError("Phase 4")


def rrf_fusion(lex: list[tuple[int, float]], sem: list[tuple[int, float]],
               k: int = 60) -> list[tuple[int, float]]:
    raise NotImplementedError("Phase 4")
