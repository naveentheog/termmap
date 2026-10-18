"""Train the FINAL models (v6) used by predict.py and save them to models_v6/.

python train_models_v6.py [--variant ft2_subpred_tagpred] [--seeds 3] [--exclude-fold K]

What gets trained (all on the full training data unless --exclude-fold is given):
  - LightGBM rankers on the out-of-fold feature table (top-40 candidates, no true subconcept):
      ICD10CM model (ICD rows only) and a pooled model used for SNOMED, each with N seeds averaged
  - subconcept text classifier and SNOMED semantic-tag text classifier (termmap/subpred.py)
The fine-tuned cross-encoder itself (models/ce2_full) was trained on Kaggle (kaggle/finetune_ce_v2.py).
"""
import argparse
import json
import os
import pickle

import lightgbm as lgb
import numpy as np

import run_fast
import run_rerank_v4 as v4
from termmap.data import ICD, SNO
from termmap.subpred import SubconceptModel, TagModel


def main(variant="ft2_subpred_tagpred", seeds=3, out="models_v6", exclude_fold=None, topk=40):
    train, F, feats = run_fast.load(variant, topk)
    if exclude_fold is not None:
        F = F[F.fold != exclude_fold]
        train = train[train.fold != exclude_fold]
    os.makedirs(out, exist_ok=True)
    params = {**v4.BASE, "learning_rate": 0.03, "num_leaves": 63}
    for group in (ICD, "pooled"):
        tr = F[F.system == ICD] if group == ICD else F
        tr = tr[tr.groupby("qid").label.transform("max") == 1]
        ds = lgb.Dataset(tr[feats], tr.label, group=tr.groupby("qid", sort=False).size().values)
        for sd in range(seeds):
            m = lgb.train({**params, "seed": sd}, ds, 900)
            m.save_model(os.path.join(out, f"ranker_{group}_s{sd}.txt"))
        print(f"[train] {group}: {seeds} rankers on {tr.qid.nunique()} queries", flush=True)
    sub = SubconceptModel().fit(train.entity, train.chunk_text, train.subconcept)
    tag_of = pickle.load(open("cache/sno_tag_of.pkl", "rb"))
    sno = train[train.coding_system == SNO]
    tagm = TagModel().fit(sno.entity, sno.chunk_text, sno.code.map(tag_of).fillna("other"))
    pickle.dump({"sub": sub, "tag": tagm}, open(os.path.join(out, "text_models.pkl"), "wb"))
    json.dump({"variant": variant, "features": feats, "seeds": seeds, "topk": topk},
              open(os.path.join(out, "config.json"), "w"), indent=1)
    print(f"[train] saved models + config ({len(feats)} features) -> {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="ft2_subpred_tagpred")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="models_v6")
    ap.add_argument("--exclude-fold", type=int, default=None)
    a = ap.parse_args()
    main(a.variant, a.seeds, a.out, a.exclude_fold)
