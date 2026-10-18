"""v5 experiments (all out-of-fold, final policy: ICD per-system model, SNOMED pooled model):
  A  v4 features                          (reference)
  B  v4 + context features
  C  B without any subconcept-based features (organizers: subconcept may be absent)
"""
import os
import pickle
import time

import numpy as np
import pandas as pd

import run_rerank_v4 as v4
from termmap.cv import accuracy_report
from termmap.data import dictionaries, SYSTEMS, ICD, SNO
from termmap.features import CodeInfo
from termmap.features_context import add_context

SUB_FEATS = {"sub_id", "p_tag", "x_code_sub_prior"}


def policy(F, train, feats):
    params, rounds, _ = v4.VARIANTS["lr .03, 900r, 63 leaves"]
    icd_pred, _ = v4.run(F, train, feats, params, rounds, pooled=False)
    sno_pred, conf = v4.run(F, train, feats, params, rounds, pooled=True)
    return pd.Series(np.where(train.coding_system == ICD, icd_pred, sno_pred), index=train.index)


def main():
    with open("cache/cands_hybrid.pkl", "rb") as fh:
        train = pickle.load(fh)["train"]
    path = "cache/features_v5.pkl"
    if os.path.exists(path):
        F = pd.read_pickle(path)
    else:
        t0 = time.time()
        D = dictionaries("data/icd.xlsx", "data/snomed.xlsx")
        info = {s: CodeInfo(D[s], s) for s in SYSTEMS}
        F = add_context(pd.read_pickle("cache/features_v4.pkl"), train, info)
        F.to_pickle(path)
        print(f"[v5] context features built in {time.time() - t0:.0f}s", flush=True)
    F["fold"] = F.qid.map(train.fold)
    F["system"] = F.qid.map(train.coding_system)
    F["is_snomed"] = (F.system == SNO).astype(float)
    base = [c for c in F.columns if c not in ("qid", "code", "label", "fold", "system")]
    v4_feats = [c for c in base if not c.startswith("c_")]
    variants = {
        "A v4": v4_feats,
        "B v4 + context": base,
        "C v4 + context, no subconcept": [c for c in base if c not in SUB_FEATS],
    }
    rows = []
    for name, feats in variants.items():
        t0 = time.time()
        train["pred"] = policy(F, train, feats)
        ok = train.pred == train.code
        r = {"model": name, "n_feats": len(feats)}
        for s in SYSTEMS:
            m = train.coding_system == s
            r[s] = round(ok[m].mean(), 4)
            r[f"{s}_diag"] = round(ok[m & (train.subconcept == "DiagnosisCode")].mean(), 4)
        r["headline"] = min(r[ICD], r[SNO])
        r["sec"] = round(time.time() - t0)
        rows.append(r)
        print(r, flush=True)
        if name.startswith("B"):
            accuracy_report(train, "pred", "v5 B: v4 + context features")
            train.to_pickle("cache/oof_v5.pkl")
    print("\n===== LEADERBOARD v5 =====")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
