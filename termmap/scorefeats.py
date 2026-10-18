"""Derived features from neural scores. Shared by CV (run_fast.py) and inference (engine_final.py),
so both paths compute exactly the same columns.

scores: {qid: {code: score}}
"""
import numpy as np


def add_ce(F, scores, tag):
    """Cross-encoder logits over the shortlist -> x_<tag>, _rank, _gap, _prob."""
    x = f"x_{tag}"
    F[x] = np.asarray([scores.get(q, {}).get(c, np.nan) for q, c in zip(F.qid.values, F.code.values)],
                      dtype=np.float32)
    g = F.groupby("qid")[x]
    F[f"{x}_rank"] = g.rank(ascending=False, method="min")
    F[f"{x}_gap"] = g.transform("max") - F[x]
    e = np.exp(F[x] - g.transform("max"))
    F[f"{x}_prob"] = e / e.groupby(F.qid).transform("sum")
    return F


def add_bi(F, scores, tag):
    """Fine-tuned SapBERT retriever: top-100 dictionary codes per query (dict in descending score order)."""
    x = f"x_{tag}"
    brank = {q: {c: r for r, c in enumerate(d)} for q, d in scores.items()}
    F[x] = np.asarray([scores.get(q, {}).get(c, np.nan) for q, c in zip(F.qid.values, F.code.values)],
                      dtype=np.float32)
    F[f"{x}_rank"] = np.asarray([brank.get(q, {}).get(c, 999) for q, c in zip(F.qid.values, F.code.values)],
                                dtype=np.float32)
    top = F.qid.map({q: (max(d.values()) if d else np.nan) for q, d in scores.items()})
    F[f"{x}_gap"] = (top - F[x]).astype(np.float32)
    return F


def add_llm(F, scores, tag):
    """LLM letter probabilities over the first-stage top-10 -> x_<tag>, _gap, _is_top."""
    x = f"x_{tag}"
    F[x] = np.asarray([scores.get(q, {}).get(c, np.nan) for q, c in zip(F.qid.values, F.code.values)],
                      dtype=np.float32)
    top = F.qid.map({q: (max(d.values()) if d else np.nan) for q, d in scores.items()})
    F[f"{x}_gap"] = (top - F[x]).astype(np.float32)
    F[f"{x}_is_top"] = (F[f"{x}_gap"] <= 1e-6).astype(np.float32)
    return F
