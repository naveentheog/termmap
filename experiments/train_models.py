"""Train the final rankers from the out-of-fold feature table and save them to models/.

python train_models.py          (needs cache/cands_hybrid.pkl and cache/features_v4.pkl)
"""
import pickle
import sys

import pandas as pd

from termmap import ranker


def main(model_dir="models", exclude_fold=None):
    with open("cache/cands_hybrid.pkl", "rb") as fh:
        train = pickle.load(fh)["train"]
    F = pd.read_pickle("cache/features_v4.pkl")
    if exclude_fold is not None:                       # used by the held-out test
        keep = train.index[train.fold != exclude_fold]
        F = F[F.qid.isin(keep)]
    models, feats = ranker.fit_rankers(F, train.coding_system)
    ranker.save(models, feats, model_dir)
    print(f"[train] saved {len(models)} rankers, {len(feats)} features -> {model_dir}/")


if __name__ == "__main__":
    main()
