"""Check that every library and offline model is usable on this machine.

Run with Wi-Fi off (or HF_HUB_OFFLINE=1) to prove the demo will work on presentation day.
"""
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from search import config  # noqa: E402

LIBS = ["streamlit", "sentence_transformers", "bm25s", "spacy", "sklearn", "rapidfuzz",
        "fitz", "docx", "bs4", "numpy", "pandas", "pyarrow", "nltk", "gensim"]

ok = True


def report(name: str, good: bool, detail: str = "") -> None:
    global ok
    ok &= good
    print(f"[{'OK ' if good else 'ERR'}] {name} {detail}")


for lib in LIBS:
    try:
        m = importlib.import_module(lib)
        report(lib, True, getattr(m, "__version__", ""))
    except Exception as e:  # noqa: BLE001
        report(lib, False, repr(e))

try:
    import torch
    dev = f"cuda ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "cpu"
    report("torch", True, f"{torch.__version__} device={dev}")
except Exception as e:  # noqa: BLE001
    report("torch", False, repr(e))

try:
    import spacy
    nlp = spacy.load(config.SPACY_MODEL)
    doc = nlp("Les développeurs ont développé des applications.")
    report("spaCy model", True, str([t.lemma_ for t in doc]))
except Exception as e:  # noqa: BLE001
    report("spaCy model", False, repr(e))

try:
    from nltk.corpus import wordnet as wn
    syn = sorted({l for s in wn.synsets("voiture", lang="fra") for l in s.lemma_names("fra")})
    report("WordNet fra", bool(syn), f"voiture -> {syn[:6]}")
except Exception as e:  # noqa: BLE001
    report("WordNet fra", False, repr(e))

from huggingface_hub import try_to_load_from_cache  # noqa: E402
for name in (config.EMBED_MODEL_MAIN, config.EMBED_MODEL_FALLBACK, config.RERANK_MODEL):
    cached = try_to_load_from_cache(name, "config.json")
    optional = name == config.RERANK_MODEL
    good = isinstance(cached, str)
    print(f"[{'OK ' if good else ('-- ' if optional else 'ERR')}] HF cache {name}"
          f"{'' if good else ' (not downloaded)'}")
    if not optional:
        ok &= good

print("\nAll required checks passed." if ok else "\nSome checks failed.")
sys.exit(0 if ok else 1)
