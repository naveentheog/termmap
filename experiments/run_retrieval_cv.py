"""Step 1 experiment: how often is the gold code among the retrieved candidates?

python run_retrieval_cv.py  -> prints Recall@k per system and saves out-of-fold candidates
(cache/cands.pkl) that the reranker trains on.
"""
import os
import pickle
import time

import numpy as np
import pandas as pd

from termmap.data import dictionaries, load_train, SYSTEMS
from termmap.retrieval import DictRetriever, Retriever
from termmap.text import clean
from termmap.cv import folds, recall_report, N_FOLDS

TRAIN, ICD_X, SNO_X = "data/hackathon_train.csv", "data/icd.xlsx", "data/snomed.xlsx"


def main():
    t0 = time.time()
    train = load_train(TRAIN)
    train["fold"] = folds(train)
    D = dictionaries(ICD_X, SNO_X)
    dret = {s: DictRetriever(D[s]) for s in SYSTEMS}
    print(f"[cv] indexes ready in {time.time() - t0:.0f}s", flush=True)

    cands = {}            # row index -> candidate DataFrame
    train["gold_rank"], train["seen"] = -1, False
    for f in range(N_FOLDS):
        for s in SYSTEMS:
            tr = train[(train.fold != f) & (train.coding_system == s)]
            va = train[(train.fold == f) & (train.coding_system == s)]
            ret = Retriever(dret[s], tr)
            res = ret.candidates([clean(e) for e in va.entity], k=50)
            seen_codes = set(tr.code)
            for idx, gold, c in zip(va.index, va.code, res):
                cands[idx] = c
                hit = np.flatnonzero(c.code.values == gold)
                train.at[idx, "gold_rank"] = int(hit[0]) if len(hit) else -1
                train.at[idx, "seen"] = gold in seen_codes
        print(f"[cv] fold {f} done ({time.time() - t0:.0f}s)", flush=True)

    recall_report(train, title="RETRIEVAL (lexical: dict char+word, train synonyms)")
    os.makedirs("cache", exist_ok=True)
    with open("cache/cands.pkl", "wb") as fh:
        pickle.dump({"cands": cands, "train": train}, fh)


if __name__ == "__main__":
    main()
