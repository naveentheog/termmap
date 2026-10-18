"""Dense retrieval with dictionary embeddings precomputed on Kaggle (kaggle/embed_dictionaries.py).

Queries (proprietary terms) are embedded locally on CPU with the same models, so no clinical
data leaves the machine. Dictionary matrices are memory-mapped and scored in blocks, so RAM
stays around 1 GB even for SNOMED (627k x 768).
"""
import os
import re

import numpy as np

MODELS = {"sapbert": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext", "gte": "thenlper/gte-base"}
SYSTEMS = ("ICD10CM", "SNOMEDCT_US")


def clean(t):
    t = str(t or "").lower().replace("<br>", " ")
    t = re.sub(r"[^a-z0-9+]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def load_model(key, device="cpu"):
    from sentence_transformers import SentenceTransformer, models
    if key == "sapbert":
        w = models.Transformer(MODELS[key], max_seq_length=32)
        return SentenceTransformer(modules=[w, models.Pooling(w.get_word_embedding_dimension(),
                                                              pooling_mode="cls")], device=device)
    m = SentenceTransformer(MODELS[key], device=device)
    m.max_seq_length = 64
    return m


def embed(key, texts, model=None):
    model = model or load_model(key)
    return model.encode([clean(t) for t in texts], batch_size=128, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=len(texts) > 500).astype(np.float32)


def blockwise_search(emb_path, q, k, block=50_000):
    """Exact inner-product top-k over a memory-mapped float16 matrix (same result as FAISS FlatIP)."""
    emb = np.load(emb_path, mmap_mode="r")
    best_s = np.full((len(q), k), -np.inf, dtype=np.float32)
    best_i = np.zeros((len(q), k), dtype=np.int64)
    for start in range(0, emb.shape[0], block):
        chunk = np.asarray(emb[start:start + block], dtype=np.float32)
        s = q @ chunk.T
        kk = min(k, s.shape[1])
        part = np.argpartition(-s, kk - 1, axis=1)[:, :kk]
        all_s = np.concatenate([best_s, np.take_along_axis(s, part, axis=1)], axis=1)
        all_i = np.concatenate([best_i, part + start], axis=1)
        top = np.argpartition(-all_s, k - 1, axis=1)[:, :k]
        best_s = np.take_along_axis(all_s, top, axis=1)
        best_i = np.take_along_axis(all_i, top, axis=1)
    order = np.argsort(-best_s, axis=1, kind="stable")
    return np.take_along_axis(best_s, order, axis=1), np.take_along_axis(best_i, order, axis=1)


def dict_search(key, system, q_emb, cache_dir="cache", k=100):
    """-> list of {code: best cosine} (top-k distinct codes) per query."""
    codes = np.load(os.path.join(cache_dir, f"codes_{system}.npy"), allow_pickle=True)
    D, I = blockwise_search(os.path.join(cache_dir, f"emb_{key}_{system}.npy"), q_emb, k * 3)
    out = []
    for d, ii in zip(D, I):
        best = {}
        for sc, j in zip(d, ii):
            c = str(codes[j])
            if sc > best.get(c, -1):
                best[c] = float(sc)
        out.append(dict(sorted(best.items(), key=lambda kv: -kv[1])[:k]))
    return out
