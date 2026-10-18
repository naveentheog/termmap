"""v7 candidates: hybrid fusion + the fine-tuned SapBERT retriever (Kaggle, out-of-fold) as an 8th RRF source.
Rebuilds the candidate lists and the full feature table:
  cache/cands_v7.pkl, cache/features_v7.pkl   (same format as cands_hybrid.pkl / features_v4.pkl)
"""
import pickle
import time

import numpy as np

from run_rerank_cv import make_feature_table
from termmap.cv import recall_report, N_FOLDS
from termmap.data import dictionaries, SYSTEMS
from termmap.features import CodeInfo
from termmap.features_extra import add_extra
from termmap.hybrid import fuse, train_neighbours_oof


def main():
    t0 = time.time()
    with open("cache/cands.pkl", "rb") as fh:
        blob = pickle.load(fh)
    lex, train = blob["cands"], blob["train"]
    dense = pickle.load(open("cache/dense_cands.pkl", "rb"))
    bi = pickle.load(open("cache/bienc_scores.pkl", "rb"))
    sap_tr = train_neighbours_oof(train, np.load("cache/entity_emb_sapbert.npy"), N_FOLDS)
    gte_tr = train_neighbours_oof(train, np.load("cache/entity_emb_gte.npy"), N_FOLDS)
    cands = {idx: fuse(lex.get(idx), dense[idx], sap_tr.get(idx), gte_tr.get(idx), {"bi_dict": bi.get(idx, {})})
             for idx in train.index}
    train = train.copy()
    for idx, c in cands.items():
        hit = np.flatnonzero(c.code.values == train.at[idx, "code"])
        train.at[idx, "gold_rank"] = int(hit[0]) if len(hit) else -1
    recall_report(train, ks=(1, 5, 10, 20, 40, 50, 100), title="RETRIEVAL v7: hybrid + fine-tuned SapBERT")
    pickle.dump({"cands": cands, "train": train}, open("cache/cands_v7.pkl", "wb"))
    print(f"[v7] candidates done ({time.time() - t0:.0f}s)", flush=True)
    D = dictionaries("data/icd.xlsx", "data/snomed.xlsx")
    F = make_feature_table(train, cands, D)
    info = {s: CodeInfo(D[s], s) for s in SYSTEMS}
    F = add_extra(F, train, info, N_FOLDS)
    F.to_pickle("cache/features_v7.pkl")
    print(f"[v7] features {F.shape} saved ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
