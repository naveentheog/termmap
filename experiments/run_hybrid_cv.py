"""Step 3 experiment: hybrid retrieval (lexical + dense) recall, out-of-fold.

Sources per (row, code):
  dict_char, dict_word, train_char   lexical (cache/cands.pkl from run_retrieval_cv.py)
  sap_dict, gte_dict                 SapBERT / GTE over dictionary descriptions (cache/dense_cands.pkl)
  sap_train, gte_train               SapBERT / GTE nearest TRAINING entities of other folds -> their code
Candidates = union ordered by reciprocal-rank fusion, top-100 kept -> cache/cands_hybrid.pkl
"""
import pickle

import numpy as np

from termmap.cv import recall_report, N_FOLDS
from termmap.hybrid import fuse, train_neighbours_oof


def main():
    with open("cache/cands.pkl", "rb") as fh:
        blob = pickle.load(fh)
    lex, train = blob["cands"], blob["train"]
    with open("cache/dense_cands.pkl", "rb") as fh:
        dense = pickle.load(fh)
    sap_tr = train_neighbours_oof(train, np.load("cache/entity_emb_sapbert.npy"), N_FOLDS)
    gte_tr = train_neighbours_oof(train, np.load("cache/entity_emb_gte.npy"), N_FOLDS)
    cands = {idx: fuse(lex.get(idx), dense[idx], sap_tr.get(idx), gte_tr.get(idx)) for idx in train.index}

    train = train.copy()
    for idx, c in cands.items():
        hit = np.flatnonzero(c.code.values == train.at[idx, "code"])
        train.at[idx, "gold_rank"] = int(hit[0]) if len(hit) else -1
    recall_report(train, ks=(1, 5, 10, 20, 50, 100), title="RETRIEVAL: hybrid (lexical + SapBERT + GTE, dict + train)")
    with open("cache/cands_hybrid.pkl", "wb") as fh:
        pickle.dump({"cands": cands, "train": train}, fh)
    print("[hybrid] saved cache/cands_hybrid.pkl")


if __name__ == "__main__":
    main()
