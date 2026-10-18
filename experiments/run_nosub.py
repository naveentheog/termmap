"""v4 without any subconcept-based features (robustness: organizers said subconcept can be ignored)."""
import pickle

import numpy as np
import pandas as pd

import run_rerank_v4 as v4
from termmap.data import SYSTEMS, ICD, SNO

SUB_FEATS = {"sub_id", "p_tag", "x_code_sub_prior"}


def main():
    with open("cache/cands_hybrid.pkl", "rb") as fh:
        train = pickle.load(fh)["train"]
    F = pd.read_pickle("cache/features_v4.pkl")
    F["fold"] = F.qid.map(train.fold)
    F["system"] = F.qid.map(train.coding_system)
    F["is_snomed"] = (F.system == SNO).astype(float)
    feats = [c for c in F.columns if c not in ("qid", "code", "label", "fold", "system") and c not in SUB_FEATS]
    params, rounds, _ = v4.VARIANTS["lr .03, 900r, 63 leaves"]
    icd, _ = v4.run(F, train, feats, params, rounds, pooled=False)
    sno, _ = v4.run(F, train, feats, params, rounds, pooled=True)
    pred = np.where(train.coding_system == ICD, icd, sno)
    ok = pred == train.code
    r = {s: round(ok[train.coding_system == s].mean(), 4) for s in SYSTEMS}
    print("v4 no-subconcept", r, "headline", min(r.values()), flush=True)
    train["pred"] = pred
    train.to_pickle("cache/oof_nosub.pkl")


if __name__ == "__main__":
    main()
