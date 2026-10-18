"""The team leaderboard: 5-fold GroupKFold by chunk_text (no sentence leaks between train/val).

Every experiment reports the same numbers so tracks A/B can be compared fairly:
  acc per system, headline = min(ICD, SNOMED), long-tail acc (gold code unseen in train fold),
  accuracy by subconcept, and for retrievers Recall@k.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .data import ICD, SNO, SYSTEMS

N_FOLDS = 5


def folds(train: pd.DataFrame, n=N_FOLDS):
    """Deterministic fold id per row."""
    fold = np.zeros(len(train), dtype=int)
    for f, (_, va) in enumerate(GroupKFold(n_splits=n).split(train, groups=train.chunk_text)):
        fold[va] = f
    return fold


def accuracy_report(df, pred_col="pred", title=""):
    """df needs: coding_system, code, subconcept, seen (gold code seen in train fold), pred_col."""
    ok = df[pred_col].astype(str) == df.code.astype(str)
    rows = {}
    for s in SYSTEMS:
        m = df.coding_system == s
        rows[s] = {"n": int(m.sum()), "acc": ok[m].mean(),
                   "tail_n": int((m & ~df.seen).sum()), "tail_acc": ok[m & ~df.seen].mean()}
    head = min(rows[ICD]["acc"], rows[SNO]["acc"])
    print(f"\n===== {title} =====")
    print(f"HEADLINE (min of systems): {head:.4f}")
    for s, r in rows.items():
        print(f"  {s:<12} n={r['n']:<5} acc={r['acc']:.4f}   long-tail n={r['tail_n']:<5} acc={r['tail_acc']:.4f}")
    by = df.assign(ok=ok).groupby(["coding_system", "subconcept"]).ok.agg(["size", "mean"])
    print("  by subconcept:")
    for (s, sc), r in by.iterrows():
        if r["size"] >= 20:
            print(f"    {s:<12} {sc:<15} n={int(r['size']):<5} acc={r['mean']:.4f}")
    return head, rows


def recall_report(df, ks=(1, 5, 10, 20, 50), title=""):
    """df needs: coding_system, gold_rank (-1 = not retrieved), seen, subconcept."""
    print(f"\n===== {title} =====")
    for s in SYSTEMS:
        d = df[df.coding_system == s]
        r = d.gold_rank
        line = "  ".join(f"R@{k}={((r >= 0) & (r < k)).mean():.3f}" for k in ks)
        tail = d[~d.seen].gold_rank
        print(f"  {s:<12} n={len(d):<5} {line}   | long-tail R@50={((tail >= 0) & (tail < 50)).mean():.3f} (n={len(tail)})")
