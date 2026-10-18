"""Couple the two systems (suggested as unexplored by the problem statement).

At assessment time every entity gets BOTH an ICD-10-CM and a SNOMED code. In training, 2,124 entity+chunk
pairs carry both. For a candidate in one system we look at the OTHER system's prediction for the same
entity (out-of-fold in CV) and add:
  cp_pair_p   : P(candidate | other system's predicted code), from training pairs of the other folds
  cp_pair_n   : how often that predicted code was seen in pairs
  cp_desc_sim : char n-gram similarity between the other system's predicted code description and this
                candidate's descriptions (works for unseen codes too)
Rows without a partner prediction get NaN.
"""
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .text import clean


def pair_key(df):
    return list(zip(df.chunk_text.values, df.entity.map(clean).values))


def pair_counts(rows, a, b):
    """{code_in_a: Counter(code_in_b)} from rows that have both systems for the same entity+chunk."""
    ra = rows[rows.coding_system == a]
    rb = rows[rows.coding_system == b]
    mb = dict(zip(pair_key(rb), rb.code.values))
    out = defaultdict(Counter)
    for k, c in zip(pair_key(ra), ra.code.values):
        if k in mb:
            out[c][mb[k]] += 1
    return out


def add_coupling(F, queries, partner_pred, pair_rows, descs, folds_n=5, fold_of=None):
    """F: candidate table (qid, code). queries: qid-indexed frame with entity, chunk_text, coding_system
    (+ fold in CV). partner_pred: {qid: predicted code of the OTHER system for the same entity} (OOF in CV).
    pair_rows: training rows used to count pairs (CV: per fold, rows of other folds).
    descs: {(system, code): [cleaned descriptions]}"""
    F = F.copy()
    other = {"ICD10CM": "SNOMEDCT_US", "SNOMEDCT_US": "ICD10CM"}
    sys_of = queries.coding_system
    p_col, n_col = np.full(len(F), np.nan, np.float32), np.full(len(F), np.nan, np.float32)
    folds = range(folds_n) if fold_of is not None else [None]
    for f in folds:
        rows = pair_rows if f is None else pair_rows[pair_rows.fold != f]
        cnt = {s: pair_counts(rows, other[s], s) for s in other}
        for i, (q, c) in enumerate(zip(F.qid.values, F.code.values)):
            if f is not None and fold_of[q] != f:
                continue
            pp = partner_pred.get(q)
            if pp is None:
                continue
            ctr = cnt[sys_of[q]].get(pp)
            n = sum(ctr.values()) if ctr else 0
            n_col[i] = n
            p_col[i] = (ctr.get(c, 0) / n) if n else 0.0
    F["cp_pair_p"], F["cp_pair_n"] = p_col, n_col
    texts = sorted({t for v in descs.values() for t in v[:3]})
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True).fit(texts)
    cache = {}

    def emb(key):
        if key not in cache:
            ts = descs.get(key, [])[:3] or [str(key[1])]
            cache[key] = vec.transform(ts)
        return cache[key]

    sim = np.full(len(F), np.nan, np.float32)
    for i, (q, c) in enumerate(zip(F.qid.values, F.code.values)):
        pp = partner_pred.get(q)
        if pp is None:
            continue
        a, b = emb((other[sys_of[q]], pp)), emb((sys_of[q], c))
        sim[i] = float((a @ b.T).max())
    F["cp_desc_sim"] = sim
    g = F.groupby("qid").cp_desc_sim
    F["cp_desc_sim_gap"] = (g.transform("max") - F.cp_desc_sim).astype(np.float32)
    F["cp_desc_sim_rank"] = g.rank(ascending=False, method="min").astype(np.float32)
    return F
