"""Quick test of the smoke cross-encoder scores (top-10, entity only): does MedCPT help the ranker?"""
import pickle

import numpy as np
import pandas as pd

import run_rerank_v4 as v4
from run_rerank_ce import add_ce
from termmap.data import SYSTEMS, ICD, SNO


def main():
    with open("cache/cands_hybrid.pkl", "rb") as fh:
        train = pickle.load(fh)["train"]
    with open("cache/ce_smoke.pkl", "rb") as fh:
        ce = pickle.load(fh)
    F = add_ce(pd.read_pickle("cache/features_v4.pkl"), ce)
    F = F.drop(columns=["x_ce_ctx", "x_ce_ctx_rank", "x_ce_ctx_gap", "x_ce_mean"])  # smoke run had no context pass
    F["fold"] = F.qid.map(train.fold)
    F["system"] = F.qid.map(train.coding_system)
    F["is_snomed"] = (F.system == SNO).astype(float)
    feats_v4 = [c for c in F.columns if c not in ("qid", "code", "label", "fold", "system") and not c.startswith("x_ce")]
    feats_ce = feats_v4 + [c for c in F.columns if c.startswith("x_ce")]

    top = F.dropna(subset=["x_ce_ent"]).sort_values(["qid", "x_ce_ent"], ascending=[True, False]).groupby("qid").head(1)
    ce_only = train.index.map(top.set_index("qid").code)
    print("CE alone:", {s: round((ce_only == train.code)[train.coding_system == s].mean(), 4) for s in SYSTEMS}, flush=True)

    params, rounds, _ = v4.VARIANTS["lr .03, 900r, 63 leaves"]
    for label, feats in (("v4", feats_v4), ("v4 + CE(smoke)", feats_ce)):
        icd, _ = v4.run(F, train, feats, params, rounds, pooled=False)
        sno, _ = v4.run(F, train, feats, params, rounds, pooled=True)
        pred = np.where(train.coding_system == ICD, icd, sno)
        ok = pred == train.code
        r = {s: round(ok[train.coding_system == s].mean(), 4) for s in SYSTEMS}
        print(label, r, "headline", min(r.values()), flush=True)
        if label != "v4":
            train["pred"] = pred
            train.to_pickle("cache/oof_ce_smoke.pkl")


if __name__ == "__main__":
    main()
