"""v4 + full MedCPT cross-encoder scores (top-20, entity and entity+context). Final policy only.
The v4 reference (65.05 / 50.61) is already known, so only the CE variant is trained here."""
import pickle

import numpy as np
import pandas as pd

import run_rerank_v4 as v4
from run_rerank_ce import add_ce
from termmap.data import SYSTEMS, ICD, SNO


def main():
    with open("cache/cands_hybrid.pkl", "rb") as fh:
        train = pickle.load(fh)["train"]
    with open("cache/ce_scores.pkl", "rb") as fh:
        ce = pickle.load(fh)
    F = add_ce(pd.read_pickle("cache/features_v4.pkl"), ce)
    F["fold"] = F.qid.map(train.fold)
    F["system"] = F.qid.map(train.coding_system)
    F["is_snomed"] = (F.system == SNO).astype(float)
    feats = [c for c in F.columns if c not in ("qid", "code", "label", "fold", "system")]
    params, rounds, _ = v4.VARIANTS["lr .03, 900r, 63 leaves"]
    icd, _ = v4.run(F, train, feats, params, rounds, pooled=False)
    sno, _ = v4.run(F, train, feats, params, rounds, pooled=True)
    pred = np.where(train.coding_system == ICD, icd, sno)
    ok = pred == train.code
    r = {s: round(ok[train.coding_system == s].mean(), 4) for s in SYSTEMS}
    print("v4 + CE(full)", r, "headline", min(r.values()), flush=True)
    train["pred"] = pred
    train.to_pickle("cache/oof_ce_full.pkl")


if __name__ == "__main__":
    main()
