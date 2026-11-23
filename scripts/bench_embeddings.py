"""Phase 0 risk check: time embedding models on 100 French chunks and extrapolate.

Answers the plan's first risk: "Is BGE-M3 too slow or too heavy on your laptops?"
Each teammate runs this once and posts the summary line.

Usage:
    python scripts/bench_embeddings.py                       # both models, auto device
    python scripts/bench_embeddings.py --device cpu          # force CPU (worst case)
    python scripts/bench_embeddings.py --models intfloat/multilingual-e5-small
    python scripts/bench_embeddings.py --corpus-chunks 3000  # extrapolate to 3000 chunks
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from search import config  # noqa: E402

# Varied French sentences; mixed into ~220-word chunks like real corpus chunks.
SENTENCES = [
    "Ingénieur logiciel avec cinq ans d'expérience, j'ai développé des applications web en Python.",
    "J'ai encadré une équipe de trois développeurs sur un projet de migration vers le cloud.",
    "Nous recherchons un développeur backend pour rejoindre notre équipe produit à Lyon.",
    "Le candidat maîtrise les bases de données relationnelles et les API REST.",
    "L'auditeur est certifié ISO 27001 depuis 2022 et intervient sur la sécurité des systèmes.",
    "La mission comprend la rédaction de la documentation technique et le support aux utilisateurs.",
    "Les inscriptions administratives se déroulent au secrétariat de la faculté jusqu'au 15 octobre.",
    "Le conseil municipal a voté le budget annuel consacré à la rénovation des écoles.",
    "Une expérience en apprentissage automatique et en traitement du langage serait appréciée.",
    "Le poste est basé à Paris avec deux jours de télétravail par semaine.",
    "Elle a obtenu un master en informatique et a effectué un stage dans une start-up.",
    "Les étudiants doivent fournir une attestation de stage signée par l'entreprise d'accueil.",
    "Le rapport présente les résultats trimestriels ainsi que les perspectives pour l'année prochaine.",
    "Compétences : Java, Spring, Docker, Kubernetes, intégration continue et tests automatisés.",
    "La réunion de rentrée aura lieu dans l'amphithéâtre principal le lundi à dix heures.",
]
QUERIES = ["candidat pour un poste de développeur", "certification sécurité informatique",
           "date limite des inscriptions", "expérience en encadrement d'équipe"]


def make_chunks(n: int, words: int = 220, seed: int = 0) -> list[str]:
    rnd = random.Random(seed)
    chunks = []
    for _ in range(n):
        out, count = [], 0
        while count < words:
            s = rnd.choice(SENTENCES)
            out.append(s)
            count += len(s.split())
        chunks.append(" ".join(out))
    return chunks


def rss_gb() -> float:
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def bench(name: str, device: str, chunks: list[str], batch: int, target: int) -> dict:
    import torch
    from sentence_transformers import SentenceTransformer

    q_pre, p_pre = config.EMBED_PREFIXES.get(name, ("", ""))
    t0 = time.perf_counter()
    model = SentenceTransformer(name, device=device)
    if name in config.EMBED_PREFIXES:
        model.max_seq_length = min(model.max_seq_length, 512)
    load_s = time.perf_counter() - t0

    passages = [p_pre + c for c in chunks]
    tok = model.tokenizer(passages, add_special_tokens=True)["input_ids"]
    max_tok = max(len(t) for t in tok)

    model.encode(passages[:4], batch_size=batch)  # warm-up
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    E = model.encode(passages, batch_size=batch, normalize_embeddings=True)
    if device == "cuda":
        torch.cuda.synchronize()
    enc_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for q in QUERIES:
        model.encode([q_pre + q], normalize_embeddings=True)
    query_ms = (time.perf_counter() - t0) / len(QUERIES) * 1000

    res = {
        "model": name, "device": device, "dim": E.shape[1], "load_s": load_s,
        "per_chunk_ms": enc_s / len(chunks) * 1000,
        "extrapolated_min": enc_s / len(chunks) * target / 60,
        "query_ms": query_ms, "max_tokens": max_tok, "ram_gb": rss_gb(),
        "vram_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0,
    }
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+",
                   default=[config.EMBED_MODEL_MAIN, config.EMBED_MODEL_FALLBACK])
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--n", type=int, default=100, help="chunks to time")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--corpus-chunks", type=int, default=1500,
                   help="expected corpus size for extrapolation (plan: ~1,430 passages)")
    args = p.parse_args()

    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    chunks = make_chunks(args.n)

    rows = []
    for name in args.models:
        print(f"Timing {name} on {device} ({args.n} chunks)...", flush=True)
        r = bench(name, device, chunks, args.batch, args.corpus_chunks)
        rows.append(r)
        warn = ""
        if name in config.EMBED_PREFIXES and r["max_tokens"] > config.E5_MAX_TOKENS:
            warn = f"  WARNING: chunks reach {r['max_tokens']} tokens > {config.E5_MAX_TOKENS}"
        print(f"  load {r['load_s']:.1f}s | {r['per_chunk_ms']:.0f} ms/chunk | "
              f"{args.corpus_chunks} chunks ~ {r['extrapolated_min']:.1f} min | "
              f"query {r['query_ms']:.0f} ms | RAM {r['ram_gb']:.1f} GB"
              f"{' | VRAM %.1f GB' % r['vram_gb'] if device == 'cuda' else ''}"
              f"{warn}", flush=True)

    print("\nSummary (paste in the team chat):")
    for r in rows:
        print(f"  {r['model']} on {r['device']}: index {args.corpus_chunks} chunks in "
              f"~{r['extrapolated_min']:.1f} min, query {r['query_ms']:.0f} ms, dim {r['dim']}")


if __name__ == "__main__":
    main()
