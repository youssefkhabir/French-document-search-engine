"""Download every model and dataset once, while online. Afterwards set HF_HUB_OFFLINE=1.

Usage:
    python scripts/download_models.py               # spaCy, NLTK WordNet, BGE-M3, e5-small
    python scripts/download_models.py --reranker    # + bge-reranker-v2-m3 (~2.3 GB)
    python scripts/download_models.py --light      # skip BGE-M3 (weak laptops)
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from search import config  # noqa: E402


def step(msg: str) -> None:
    print(f"\n=== {msg}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reranker", action="store_true", help="also download the cross-encoder")
    p.add_argument("--light", action="store_true", help="skip BGE-M3, keep only e5-small")
    args = p.parse_args()

    step(f"spaCy {config.SPACY_MODEL}")
    subprocess.run([sys.executable, "-m", "spacy", "download", config.SPACY_MODEL], check=True)

    step("NLTK WordNet + Open Multilingual WordNet (optional component)")
    import nltk
    # NLTK >= 3.10 reads Open Multilingual WordNet from omw-2.0 (older NLTK: omw-1.4).
    for pkg in ("wordnet", "omw-2.0", "omw-1.4"):
        nltk.download(pkg, quiet=True)

    from huggingface_hub import snapshot_download
    models = [config.EMBED_MODEL_FALLBACK]
    if not args.light:
        models.insert(0, config.EMBED_MODEL_MAIN)
    if args.reranker:
        models.append(config.RERANK_MODEL)
    for name in models:
        step(f"Hugging Face {name}")
        # Skip duplicate ONNX / other-format weights to save disk space.
        snapshot_download(name, ignore_patterns=["*.onnx", "onnx/*", "*.msgpack", "*.h5",
                                                 "openvino/*", "imgs/*"])

    step("Done. Set HF_HUB_OFFLINE=1 and run scripts/check_env.py with Wi-Fi off.")


if __name__ == "__main__":
    main()
