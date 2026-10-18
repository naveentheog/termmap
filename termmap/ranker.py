"""Final LightGBM rankers (the configuration that won the v4 leaderboard):
  ICD10CM     : ICD-only model, lr 0.03, 900 rounds, 63 leaves
  SNOMEDCT_US : model trained on BOTH systems pooled (more data), same params
Trained on the out-of-fold feature table so train-time and test-time features match.
"""
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import ICD, SNO

PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[1], lambdarank_truncation_level=10,
              learning_rate=0.03, num_leaves=63, min_child_samples=20, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, seed=0, deterministic=True,
              force_row_wise=True, verbose=-1, num_threads=2)
ROUNDS = 900
NON_FEATURES = {"qid", "code", "label", "fold", "system", "score"}


def feature_columns(F):
    base = [c for c in F.columns if c not in NON_FEATURES and not c.startswith("x_") and c != "is_snomed"]
    return base + sorted(c for c in F.columns if c.startswith("x_")) + ["is_snomed"]


def _fit(F, feats):
    F = F[F.groupby("qid").label.transform("max") == 1]
    ds = lgb.Dataset(F[feats], F.label, group=F.groupby("qid", sort=False).size().values)
    return lgb.train(PARAMS, ds, ROUNDS)


def fit_rankers(F, systems_of_qid):
    """F: feature table with qid, code, label. systems_of_qid: Series qid -> coding_system."""
    F = F.sort_values(["qid", "rank"]).copy()
    F["system"] = F.qid.map(systems_of_qid)
    F["is_snomed"] = (F.system == SNO).astype(float)
    feats = feature_columns(F)
    return {ICD: _fit(F[F.system == ICD], feats), SNO: _fit(F, feats)}, feats


def save(models, feats, path):
    os.makedirs(path, exist_ok=True)
    for s, m in models.items():
        m.save_model(os.path.join(path, f"ranker_{s}.txt"))
    with open(os.path.join(path, "features.json"), "w") as fh:
        json.dump(feats, fh)


def load(path):
    with open(os.path.join(path, "features.json")) as fh:
        feats = json.load(fh)
    models = {s: lgb.Booster(model_file=os.path.join(path, f"ranker_{s}.txt")) for s in (ICD, SNO)}
    return models, feats


def score(models, feats, F, system_of_qid):
    F = F.copy()
    F["is_snomed"] = (F.qid.map(system_of_qid) == SNO).astype(float)
    F["score"] = 0.0
    for s, m in models.items():
        sel = F.qid.map(system_of_qid).values == s
        if sel.any():
            F.loc[sel, "score"] = m.predict(F.loc[sel, feats])
    return F
