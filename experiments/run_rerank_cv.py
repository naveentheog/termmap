"""Step 2 experiment: LightGBM LambdaRank over the retrieved candidates.

Uses the out-of-fold candidates from run_retrieval_cv.py. The ranker for fold f is trained
only on rows from the other folds, so every reported number is out-of-fold.
"""
import os
import pickle
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from termmap.data import dictionaries, SYSTEMS, SNO
from termmap.features import CodeInfo, FoldStats, build_features
from termmap.cv import accuracy_report, N_FOLDS

PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[1], learning_rate=0.05,
              num_leaves=31, min_child_samples=20, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambdarank_truncation_level=10, seed=0, deterministic=True,
              force_row_wise=True, verbose=-1, num_threads=2)
ROUNDS = 400


def make_feature_table(train, cands, D):
    info = {s: CodeInfo(D[s], s) for s in SYSTEMS}
    stats = {f: FoldStats(train[train.fold != f], info[SNO]) for f in range(N_FOLDS)}
    parts = []
    t0 = time.time()
    for n, (idx, row) in enumerate(train.iterrows()):
        c = cands[idx]
        if len(c) == 0:
            continue
        f = build_features(row.entity, row.chunk_text, row.subconcept, c, info[row.coding_system], stats[row.fold])
        f.insert(0, "qid", idx)
        f.insert(1, "code", c.code.values)
        f["label"] = (c.code.values == row.code).astype(int)
        parts.append(f)
        if n % 2000 == 0:
            print(f"[feat] {n}/{len(train)} ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(parts, ignore_index=True)


def main(cands_path="cache/cands.pkl", feat_path="cache/features.pkl", tag="TRACK A: retrieval + LightGBM reranker"):
    with open(cands_path, "rb") as fh:
        blob = pickle.load(fh)
    cands, train = blob["cands"], blob["train"]
    if os.path.exists(feat_path):
        F = pd.read_pickle(feat_path)
    else:
        D = dictionaries("data/icd.xlsx", "data/snomed.xlsx")
        F = make_feature_table(train, cands, D)
        F.to_pickle(feat_path)
    F["fold"] = F.qid.map(train.fold)
    F["system"] = F.qid.map(train.coding_system)
    feats = [c for c in F.columns if c not in ("qid", "code", "label", "fold", "system")]
    print(f"[rerank] {F.qid.nunique():,} queries, {len(F):,} pairs, {len(feats)} features")

    top = F[F["rank"] == 0].set_index("qid").code
    train["pred_retrieval"] = train.index.map(top)
    accuracy_report(train, "pred_retrieval", "BASELINE: retrieval top-1 (no reranker)")

    train["pred_lgb"], train["conf"] = None, 0.0
    imp = []
    for s in SYSTEMS:
        for f in range(N_FOLDS):
            tr = F[(F.system == s) & (F.fold != f)]
            tr = tr[tr.groupby("qid").label.transform("max") == 1]        # need a positive per query
            va = F[(F.system == s) & (F.fold == f)]
            ds = lgb.Dataset(tr[feats], tr.label, group=tr.groupby("qid", sort=False).size().values)
            model = lgb.train(PARAMS, ds, ROUNDS)
            va = va.assign(score=model.predict(va[feats]))
            va = va.sort_values(["qid", "score", "code"], ascending=[True, False, True])
            best = va.groupby("qid").head(2)
            for qid, g in best.groupby("qid"):
                train.at[qid, "pred_lgb"] = g.code.iloc[0]
                sc = g.score.values
                train.at[qid, "conf"] = float(sc[0] - sc[1]) if len(sc) > 1 else 9.0
            imp.append(pd.Series(model.feature_importance("gain"), index=feats, name=f"{s}{f}"))
    accuracy_report(train, "pred_lgb", tag)
    imp = pd.concat(imp, axis=1).mean(axis=1).sort_values(ascending=False)
    print("\ntop features (gain):", (imp / imp.sum()).round(3).head(12).to_dict())
    train.to_pickle(feat_path.replace("features", "oof"))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "hybrid":
        main("cache/cands_hybrid.pkl", "cache/features_hybrid.pkl",
             "TRACK A v2: hybrid retrieval (lexical+SapBERT+GTE) + LightGBM reranker")
    else:
        main()
