"""Hybrid candidate fusion shared by CV (run_hybrid_cv.py) and inference (predict.py)."""
import numpy as np
import pandas as pd

from .data import SYSTEMS

K_KEEP = 100
RRF_K = 30
LEX_COLS = ("dict_char", "dict_word", "train_char")


def _neigh(sims, codes, k):
    k = min(k, sims.shape[1])
    top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    out = []
    for qi in range(sims.shape[0]):
        agg, votes = {}, {}
        for j in top[qi]:
            c, sc = codes[j], float(sims[qi, j])
            agg[c] = max(agg.get(c, -1), sc)
            votes[c] = votes.get(c, 0) + 1
        out.append((agg, votes))
    return out


def train_neighbours_oof(train, emb, n_folds, k=20):
    """CV: nearest training entities from OTHER folds, same coding system."""
    emb = emb.astype(np.float32)
    out = {}
    for s in SYSTEMS:
        for f in range(n_folds):
            va = train.index[(train.coding_system == s) & (train.fold == f)].values
            tr = train.index[(train.coding_system == s) & (train.fold != f)].values
            if len(va) == 0 or len(tr) == 0:
                continue
            res = _neigh(emb[va] @ emb[tr].T, train.code.values[tr], k)
            out.update(zip(va, res))
    return out


def train_neighbours_query(q_emb, q_system, train, train_emb, k=20):
    """Inference: nearest entities in the FULL training data, same coding system."""
    q_emb, train_emb = q_emb.astype(np.float32), train_emb.astype(np.float32)
    out = [None] * len(q_emb)
    for s in SYSTEMS:
        qi = np.flatnonzero(np.asarray(q_system) == s)
        ti = np.flatnonzero(train.coding_system.values == s)
        if len(qi) == 0:
            continue
        res = _neigh(q_emb[qi] @ train_emb[ti].T, train.code.values[ti], k)
        for i, r in zip(qi, res):
            out[i] = r
    return out


def fuse(lex_df, dense_row, sap_tr, gte_tr, extra=None):
    """One query -> candidate DataFrame with per-source scores/ranks, ordered by RRF."""
    src = {}
    if lex_df is not None and len(lex_df):
        for col in LEX_COLS:
            if col in lex_df:
                src[col] = dict(zip(lex_df.code, lex_df[col]))
    for m, name in (("sapbert", "sap_dict"), ("gte", "gte_dict")):
        src[name] = (dense_row or {}).get(m, {})
    src["sap_train"], sap_votes = sap_tr or ({}, {})
    src["gte_train"], gte_votes = gte_tr or ({}, {})
    for name, d in (extra or {}).items():
        src[name] = d or {}
    rrf = {}
    for name, d in src.items():
        ranked = sorted(((c, s) for c, s in d.items() if s > 0), key=lambda kv: (-kv[1], kv[0]))
        for r, (c, _) in enumerate(ranked):
            rrf[c] = rrf.get(c, 0.0) + 1.0 / (RRF_K + r)
    top = sorted(rrf.items(), key=lambda kv: (-kv[1], kv[0]))[:K_KEEP]
    df = pd.DataFrame({"code": [c for c, _ in top], "rrf": [v for _, v in top]})
    for name, d in src.items():
        df[name] = df.code.map(d).fillna(0.0).astype(np.float32)
        df[f"{name}_rank"] = df[name].rank(ascending=False, method="min").where(df[name] > 0, 999).astype(np.float32)
    df["sap_votes"] = df.code.map(sap_votes).fillna(0).astype(np.float32)
    df["gte_votes"] = df.code.map(gte_votes).fillna(0).astype(np.float32)
    df["n_sources"] = (df[list(src)] > 0).sum(axis=1).astype(np.float32)
    df["rank"] = np.arange(len(df), dtype=np.float32)
    df["fused"] = df["rrf"]
    return df
