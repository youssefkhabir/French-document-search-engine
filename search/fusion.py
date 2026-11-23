"""Phase 4 · Fuse lexical and semantic candidate lists.

weighted: min-max each list over its own candidates (guard max == min -> all 1),
          absent chunk -> 0, score = alpha * sem + (1 - alpha) * lex.
rrf:      reciprocal rank fusion, kept as one comparison row.

Both take lists of (chunk row, score) sorted best first and return the same shape.
"""


def minmax(scores: dict[int, float]) -> dict[int, float]:
    """Rescale to [0, 1] over these candidates. When every score is equal (one candidate,
    or ties) there is no spread to rescale: all get 1, so they still count as found."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def weighted_fusion(lex: list[tuple[int, float]], sem: list[tuple[int, float]],
                    alpha: float) -> list[tuple[int, float]]:
    """alpha = weight of the semantic branch: 0 -> lexical only, 1 -> semantic only.

    A chunk found by one branch only gets 0 from the other: being in both lists is a
    signal of its own.
    """
    nl, ns = minmax(dict(lex)), minmax(dict(sem))
    fused = {row: alpha * ns.get(row, 0.0) + (1 - alpha) * nl.get(row, 0.0)
             for row in nl.keys() | ns.keys()}
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def rrf_fusion(lex: list[tuple[int, float]], sem: list[tuple[int, float]],
               k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: sum of 1 / (k + rank) over the lists. Uses ranks only, so
    it needs no score normalization and no alpha."""
    fused: dict[int, float] = {}
    for hits in (lex, sem):
        for rank, (row, _) in enumerate(hits, 1):
            fused[row] = fused.get(row, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
