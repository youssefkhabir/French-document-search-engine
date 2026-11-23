"""Central settings: paths, model names, and tunable parameters.

Every module imports from here so a model swap (BGE-M3 -> e5-small) is a one-line change.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
INDEX_DIR = ROOT / "index"
EVAL_DIR = ROOT / "eval"

# Hugging Face models live in the project (models/hub), not in the user profile on C:.
# Set before any HF library is imported; an explicit HF_HUB_CACHE env var still wins.
# Models already downloaded elsewhere can be linked in (see README, "Reuse models").
MODELS_DIR = ROOT / "models" / "hub"
os.environ.setdefault("HF_HUB_CACHE", str(MODELS_DIR))

# --- Models -----------------------------------------------------------------
SPACY_MODEL = "fr_core_news_sm"

# Embedding model: first choice and fallback (see plan section 05).
EMBED_MODEL_MAIN = "BAAI/bge-m3"
EMBED_MODEL_FALLBACK = "intfloat/multilingual-e5-small"
EMBED_MODEL = EMBED_MODEL_MAIN  # switch here after the Phase 0 benchmark

# e5 models need these prefixes; BGE-M3 needs none.
EMBED_PREFIXES = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
}
E5_MAX_TOKENS = 480

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# --- Chunking ---------------------------------------------------------------
CHUNK_MIN_WORDS = 200
CHUNK_MAX_WORDS = 250
CHUNK_OVERLAP_WORDS = 40
# Prefix each chunk's indexed text with the document title (ablation: set to False).
CHUNK_TITLE_PREFIX = True

# --- Search -----------------------------------------------------------------
BRANCH_TOP_K = 50        # candidates kept by each branch
RERANK_TOP_K = 20        # chunks sent to the cross-encoder
RESULTS_TOP_K = 10       # documents shown
DEFAULT_ALPHA = 0.5      # weight of the semantic branch in fusion

# Typo correction: max Levenshtein distance by word length.
TYPO_MIN_LEN = 4
TYPO_MAX_DIST_SHORT = 1  # words of 4-7 letters
TYPO_MAX_DIST_LONG = 2   # words of 8+ letters

# --- Optional: course Chapter 3 components (see docs/plan_addendum.md) -------
WORDNET_LANG = "fra"
WORDNET_MAX_EXPANSIONS = 3     # new terms per query word
WORDNET_ORIGINAL_REPEAT = 2    # repeat original terms to keep them dominant

W2V_DIM = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_PRETRAINED_PATH = None     # e.g. ROOT / "models" / "frwac.bin" to use pretrained vectors
