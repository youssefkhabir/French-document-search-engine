"""Build the index from a corpus folder and print a summary.

Usage:
    python scripts/build_index.py                                  # data/corpus -> index/
    python scripts/build_index.py --corpus tests/fixtures/corpus   # synthetic test corpus
    python scripts/build_index.py --show 3                         # also print 3 chunks
    python scripts/build_index.py --no-embed                       # lexical only, no GPU needed
    python scripts/build_index.py --model intfloat/multilingual-e5-small   # another model
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from search import config  # noqa: E402  (sets HF_HUB_CACHE before any HF import)
from search.engine import build_index  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=str(config.CORPUS_DIR))
    p.add_argument("--index", default=str(config.INDEX_DIR))
    p.add_argument("--show", type=int, default=0, help="print the first N chunks")
    p.add_argument("--no-embed", action="store_true", help="skip the embeddings (Phase 3)")
    p.add_argument("--model", default=None, help=f"embedding model (default {config.EMBED_MODEL})")
    args = p.parse_args()

    s = build_index(Path(args.corpus), Path(args.index), embed=not args.no_embed,
                    embed_model=args.model, show_progress=True)
    print(f"{s['n_documents']} documents, {s['n_pages']} pages -> {s['n_chunks']} chunks "
          f"in {s['seconds']} s")
    e = s["embeddings"]
    if e:
        print(f"  embeddings: {e['model']} on {e['device']}, {e['n']} x {e['dim']} in {e['seconds']} s "
              f"({e['n_truncated']} chunks over {e['max_tokens']} tokens truncated)")
    for path, reason in s["skipped"]:
        print(f"  skipped {path}: {reason}")
    for doc, lines in s["removed_headers_footers"].items():
        print(f"  {doc}: removed {len(lines)} header/footer lines, e.g. {lines[0]!r}")

    if args.show:
        import pandas as pd
        df = pd.read_parquet(Path(args.index) / "chunks.parquet")
        for _, row in df.head(args.show).iterrows():
            print(f"\n--- {row.chunk_id} (page {row.page}, {row.n_words} words)\n{row.text_raw[:400]}")


if __name__ == "__main__":
    main()
