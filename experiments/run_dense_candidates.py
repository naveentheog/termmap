"""Run on YOUR LAPTOP (CPU is fine, ~10-20 min). Needs the Kaggle outputs in cache/:
    cache/codes_ICD10CM.npy  cache/codes_SNOMEDCT_US.npy
    cache/emb_sapbert_ICD10CM.npy  cache/emb_sapbert_SNOMEDCT_US.npy
    (optional) cache/emb_gte_ICD10CM.npy  cache/emb_gte_SNOMEDCT_US.npy

    pip install sentence-transformers faiss-cpu pandas numpy
    python run_dense_candidates.py --train data/hackathon_train.csv

What it does (all locally, no data leaves the machine):
  1. embeds every training entity with SapBERT (and GTE if its files exist)
  2. searches the dictionary embeddings -> top-100 codes per row per model
  3. saves cache/dense_cands.pkl  (row index -> {model: {code: score}})     ~ few MB
     and     cache/entity_emb_<model>.npy (entity embeddings, for "train synonyms" dense search)
Upload those small files back to the chat; the 2.5 GB embeddings stay on your laptop.
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

MODELS = {"sapbert": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext", "gte": "thenlper/gte-base"}
SYSTEMS = ("ICD10CM", "SNOMEDCT_US")


def clean(t):
    import re
    t = str(t or "").lower().replace("<br>", " ")
    t = re.sub(r"[^a-z0-9+]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def load_model(key):
    from sentence_transformers import SentenceTransformer, models
    if key == "sapbert":
        w = models.Transformer(MODELS[key], max_seq_length=32)
        return SentenceTransformer(modules=[w, models.Pooling(w.get_word_embedding_dimension(), pooling_mode="cls")],
                                   device="cpu")
    m = SentenceTransformer(MODELS[key], device="cpu"); m.max_seq_length = 64
    return m


def blockwise_search(emb_path, q, k, block=50_000):
    """Exact inner-product top-k without loading the whole matrix: the float16 file is
    memory-mapped and scored in blocks. Same results as FAISS IndexFlatIP, ~1 GB RAM."""
    emb = np.load(emb_path, mmap_mode="r")
    best_s = np.full((len(q), k), -np.inf, dtype=np.float32)
    best_i = np.zeros((len(q), k), dtype=np.int64)
    for start in range(0, emb.shape[0], block):
        chunk = np.asarray(emb[start:start + block], dtype=np.float32)
        s = q @ chunk.T
        kk = min(k, s.shape[1])
        part = np.argpartition(-s, kk - 1, axis=1)[:, :kk]
        ps = np.take_along_axis(s, part, axis=1)
        all_s = np.concatenate([best_s, ps], axis=1)
        all_i = np.concatenate([best_i, part + start], axis=1)
        top = np.argpartition(-all_s, k - 1, axis=1)[:, :k]
        best_s = np.take_along_axis(all_s, top, axis=1)
        best_i = np.take_along_axis(all_i, top, axis=1)
    order = np.argsort(-best_s, axis=1, kind="stable")
    return np.take_along_axis(best_s, order, axis=1), np.take_along_axis(best_i, order, axis=1)


def main():
    import faiss
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/hackathon_train.csv")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--low-mem", action="store_true", help="memory-mapped blockwise search (~1 GB RAM)")
    a = ap.parse_args()

    df = pd.read_csv(a.train, dtype=str, keep_default_na=False)
    queries = [clean(e) for e in df.entity]
    out = {i: {} for i in range(len(df))}
    for key in MODELS:
        if not all(os.path.exists(f"{a.cache}/emb_{key}_{s}.npy") for s in SYSTEMS):
            print(f"[dense] skipping {key}: embedding files not found"); continue
        t0 = time.time()
        model = load_model(key)
        q = model.encode(queries, batch_size=128, show_progress_bar=True, normalize_embeddings=True,
                         convert_to_numpy=True).astype(np.float32)
        np.save(f"{a.cache}/entity_emb_{key}.npy", q.astype(np.float16))
        for s in SYSTEMS:
            codes = np.load(f"{a.cache}/codes_{s}.npy", allow_pickle=True)
            rows = np.flatnonzero(df.coding_system.values == s)
            if a.low_mem:
                D, I = blockwise_search(f"{a.cache}/emb_{key}_{s}.npy", q[rows], a.k * 3)
            else:
                emb = np.load(f"{a.cache}/emb_{key}_{s}.npy").astype(np.float32)
                index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb); del emb
                D, I = index.search(q[rows], a.k * 3)
                del index
            for r, d, ii in zip(rows, D, I):
                best = {}
                for sc, j in zip(d, ii):
                    c = str(codes[j])
                    if sc > best.get(c, -1):
                        best[c] = float(sc)
                out[r][key] = dict(sorted(best.items(), key=lambda kv: -kv[1])[: a.k])
            print(f"[dense] {key} {s}: {len(rows)} rows done ({time.time() - t0:.0f}s)", flush=True)
    with open(f"{a.cache}/dense_cands.pkl", "wb") as fh:
        pickle.dump(out, fh)
    print("[dense] saved", f"{a.cache}/dense_cands.pkl", "and entity_emb_*.npy -> upload these to the chat")


if __name__ == "__main__":
    main()
