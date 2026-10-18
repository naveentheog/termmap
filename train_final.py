"""Train the FINAL rankers + helpers used by predict.py, from the out-of-fold feature tables.

python train_final.py [--out models_final] [--exclude-fold K] [--coupling]

Stage A : ft4_tagpred_bi_bi3_truesub          (no LLM) -> its top-10 goes to the LLM, its code is the partner
Stage B : ft4_tagpred_bi_bi3[_cp]_truesub_llmft (+ LLM probabilities, + coupling with stage A's partner code)
Each stage: one ICD model (ICD rows) + one pooled model (used for SNOMED), 3 seeds averaged.
The neural scores in the tables are out-of-fold, so the rankers learn how much to trust them on unseen rows.
"""
import argparse
import json
import os
import pickle

import sys

import lightgbm as lgb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments"))

import run_fast
import run_rerank_v4 as v4
from termmap.data import ICD

STAGE_A = "ft4_tagpred_bi_bi3_truesub"
PARTNER_A = "cache/oof_fast_ft4_tagpred_bi_bi3_truesub_k40_lr0.03_r900_l63_s3_v7.pkl"


def train_stage(variant, out, stage, seeds, exclude_fold, topk=40):
    train, F, feats = run_fast.load(variant, topk, "v7")
    if exclude_fold is not None:
        F = F[F.fold != exclude_fold]
    params = {**v4.BASE, "learning_rate": 0.03, "num_leaves": 63}
    for group in (ICD, "pooled"):
        tr = F[F.system == ICD] if group == ICD else F
        tr = tr[tr.groupby("qid").label.transform("max") == 1]
        ds = lgb.Dataset(tr[feats], tr.label, group=tr.groupby("qid", sort=False).size().values)
        for sd in range(seeds):
            lgb.train({**params, "seed": sd}, ds, 900).save_model(os.path.join(out, f"ranker_{stage}_{group}_s{sd}.txt"))
        print(f"[train] stage {stage} {group}: {seeds} rankers, {tr.qid.nunique()} queries, {len(feats)} feats", flush=True)
    return train, feats


def main(out="models_final", seeds=3, exclude_fold=None, coupling=False):
    os.makedirs(out, exist_ok=True)
    train, fa = train_stage(STAGE_A, out, "A", seeds, exclude_fold)
    if coupling:
        os.environ["CP_PARTNER"] = PARTNER_A
    vb = "ft4_tagpred_bi_bi3_cp_truesub_llmft" if coupling else "ft4_tagpred_bi_bi3_truesub_llmft"
    _, fb = train_stage(vb, out, "B", seeds, exclude_fold)
    tag_of = pickle.load(open("cache/sno_tag_of.pkl", "rb"))
    descs = pickle.load(open("cache/abbrev.pkl", "rb"))["descs"]
    pickle.dump({"tag_of": dict(tag_of), "descs": dict(descs)}, open(os.path.join(out, "aux.pkl"), "wb"))
    json.dump({"stage_A": STAGE_A, "stage_B": vb, "features_A": fa, "features_B": fb, "seeds": seeds, "topk": 40,
               "coupling": coupling, "exclude_fold": exclude_fold}, open(os.path.join(out, "config.json"), "w"), indent=1)
    print(f"[train] saved -> {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models_final")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--exclude-fold", type=int, default=None)
    ap.add_argument("--coupling", action="store_true")
    a = ap.parse_args()
    main(a.out, a.seeds, a.exclude_fold, a.coupling)
