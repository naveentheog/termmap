"""v5: v4 features + MedCPT cross-encoder scores -> LightGBM. Out-of-fold leaderboard.

python run_rerank_ce.py      (needs cache/features_v4.pkl, cache/cands_hybrid.pkl, cache/ce_scores.pkl)

Candidates outside the cross-encoder's top-K get NaN (LightGBM treats it as "not scored").
"""
import pickle

import numpy as np
import pandas as pd

from termmap.cv import accuracy_report
from termmap.data import SYSTEMS, ICD, SNO
import run_rerank_v4 as v4


def add_ce(F, ce):
    e = [ce.get(q, {}).get(c, [np.nan, np.nan]) for q, c in zip(F.qid.values, F.code.values)]
    e = np.asarray(e, dtype=np.float32)
    F = F.copy()
    F["x_ce_ent"], F["x_ce_ctx"] = e[:, 0], e[:, 1]
    for col in ("x_ce_ent", "x_ce_ctx"):
        g = F.groupby("qid")[col]
        F[f"{col}_rank"] = g.rank(ascending=False, method="min")
        F[f"{col}_gap"] = g.transform("max") - F[col]
    F["x_ce_mean"] = F[["x_ce_ent", "x_ce_ctx"]].mean(axis=1)
    return F


def main():
    with open("cache/cands_hybrid.pkl", "rb") as fh:
        train = pickle.load(fh)["train"]
    with open("cache/ce_scores.pkl", "rb") as fh:
        ce = pickle.load(fh)
    print(f"[ce] scores for {len(ce)} of {len(train)} queries")
    F = add_ce(pd.read_pickle("cache/features_v4.pkl"), ce)
    F["fold"] = F.qid.map(train.fold)
    F["system"] = F.qid.map(train.coding_system)
    F["is_snomed"] = (F.system == SNO).astype(float)
    feats_v4 = [c for c in F.columns if c not in ("qid", "code", "label", "fold", "system") and not c.startswith("x_ce")]
    feats_ce = feats_v4 + [c for c in F.columns if c.startswith("x_ce")]

    top = F.dropna(subset=["x_ce_ctx"]).sort_values(["qid", "x_ce_ctx"], ascending=[True, False]).groupby("qid").head(1)
    train["pred_ce_only"] = train.index.map(top.set_index("qid").code)
    accuracy_report(train, "pred_ce_only", "MedCPT cross-encoder alone (top-1 of its candidates)")

    params, rounds, _ = v4.VARIANTS["lr .03, 900r, 63 leaves"]
    res = {}
    for label, feats in (("v4", feats_v4), ("v4 + cross-encoder", feats_ce)):
        for pooled in (False, True):
            pred, conf = v4.run(F, train, feats, params, rounds, pooled)
            ok = pred == train.code
            res[(label, pooled)] = (pred, conf, {s: ok[train.coding_system == s].mean() for s in SYSTEMS})
            print(label, "pooled" if pooled else "per-system", {s: round(v, 4) for s, v in res[(label, pooled)][2].items()},
                  flush=True)
    for label in ("v4", "v4 + cross-encoder"):
        icd_pred = res[(label, False)][0]
        sno_pred = res[(label, True)][0]
        train[f"pred_{label}"] = np.where(train.coding_system == ICD, icd_pred, sno_pred)
        accuracy_report(train, f"pred_{label}", f"{label}  (ICD per-system, SNOMED pooled)")
    train.to_pickle("cache/oof_v5.pkl")


if __name__ == "__main__":
    main()
