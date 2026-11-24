"""Phase 5 · Paired significance tests on per-query nDCG@10 (from eval/run_eval.py).

Two tests per pair of systems, both paired (same questions):
- paired bootstrap: resample the questions 10,000 times; p = share of resamples where the
  mean difference has the opposite sign of the observed one (x2, two-sided); 95% CI.
- Wilcoxon signed-rank (scipy): ranks the per-question differences; ties (diff = 0) dropped.

Usage: python -m eval.significance eval/results/alloprof_test_sample100
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PAIRS = [("S4", "S3"), ("S5", "S4"), ("S1", "S2"), ("S4", "S4rrf"), ("S2W", "S2"),
         ("S3", "S2"), ("S6", "S2")]
N_BOOT = 10_000
SEED = 42


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT,
                     seed: int = SEED) -> tuple[float, float, float]:
    """(p-value, CI low, CI high) for mean(a - b)."""
    d = a - b
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    observed = d.mean()
    if observed == 0:
        p = 1.0
    else:
        p = min(1.0, 2 * float(np.mean(means <= 0 if observed > 0 else means >= 0)))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return p, float(lo), float(hi)


def compare(per_query: pd.DataFrame, pairs=PAIRS, metric: str = "ndcg@10") -> pd.DataFrame:
    wide = per_query.pivot(index="qid", columns="system", values=metric)
    rows = []
    for x, y in pairs:
        if x not in wide or y not in wide:
            continue
        a, b = wide[x].to_numpy(), wide[y].to_numpy()
        p_boot, lo, hi = paired_bootstrap(a, b)
        d = a - b
        p_wil = float(wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0
        rows.append({"comparison": f"{x} vs {y}", "mean_a": a.mean(), "mean_b": b.mean(),
                     "diff": d.mean(), "ci95": f"[{lo:+.3f}, {hi:+.3f}]",
                     "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                     "ties": int((d == 0).sum()), "p_bootstrap": p_boot, "p_wilcoxon": p_wil,
                     "significant": "yes" if max(p_boot, p_wil) < 0.05 else "no"})
    return pd.DataFrame(rows).round({"mean_a": 3, "mean_b": 3, "diff": 3,
                                     "p_bootstrap": 4, "p_wilcoxon": 4})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", help="folder written by eval.run_eval (has per_query.csv)")
    p.add_argument("--metric", default="ndcg@10")
    args = p.parse_args()
    folder = Path(args.results)
    table = compare(pd.read_csv(folder / "per_query.csv"), metric=args.metric)
    table.to_csv(folder / "significance.csv", index=False)
    (folder / "significance.md").write_text(
        f"# Paired tests on {args.metric} ({N_BOOT} bootstrap resamples, Wilcoxon)\n\n"
        f"{table.to_markdown(index=False)}\n", encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
