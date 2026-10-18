"""End-to-end check of the FINAL inference path on held-out fold 0.

Rankers, tag classifier, lexical index and nearest-neighbour pools are built WITHOUT fold 0. Fold 0 is then
fed through FinalMapper.predict like an unseen file. The GPU models cannot run in this sandbox, so their
out-of-fold outputs for fold 0 are injected (they come from models that never saw fold 0):
dense dictionary search, SapBERT v2/v3, cross-encoder v4, LLM.
Expected: predict-path accuracy ~= out-of-fold CV accuracy of the same configuration on fold 0.
"""
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import train_final
from termmap.coupling import pair_key
from termmap.engine_final import FinalMapper
from termmap.hybrid import train_neighbours_query
from termmap.data import SYSTEMS

FOLD = 0


class Injected:
    def __init__(self, te, tr_df, emb):
        self.te, self.tr, self.emb = te, tr_df, emb
        self.dense_c = pickle.load(open("cache/dense_cands.pkl", "rb"))
        self.bi_s = {"bi": pickle.load(open("cache/bienc_scores.pkl", "rb")),
                     "bi3": pickle.load(open("cache/bienc3_scores.pkl", "rb"))}
        self.ft4 = pickle.load(open("cache/ft4_scores.pkl", "rb"))
        self.llm_s = pickle.load(open("cache/llm_ft_scores.pkl", "rb"))
        self.orig = te.index.values

    def dense(self, q):
        neigh = {k: train_neighbours_query(e[self.orig], q.coding_system.values, self.tr, e[self.tr_idx])
                 for k, e in self.emb.items()}
        return [self.dense_c[i] for i in self.orig], neigh

    def bi(self, q, name):
        return [self.bi_s[name].get(i, {}) for i in self.orig]

    def ce(self, q, codes):
        return {qid: self.ft4.get(self.orig[qid], {}) for qid in codes}

    def llm(self, q, top10):
        return {qid: self.llm_s.get(self.orig[qid], {}) for qid in top10}


def main(coupling=False, retrain=True):
    out = "models_final_ho" + ("_cp" if coupling else "")
    train = pickle.load(open("cache/cands_v7.pkl", "rb"))["train"]
    if retrain:
        train_final.main(out=out, exclude_fold=FOLD, coupling=coupling)
    tr, te = train[train.fold != FOLD], train[train.fold == FOLD]
    emb = {k: np.load(f"cache/entity_emb_{k}.npy").astype(np.float32) for k in ("sapbert", "gte")}
    tm = FinalMapper(tr.reset_index(drop=True), "data/icd.xlsx", "data/snomed.xlsx", out)
    inj = Injected(te, tm.train, emb)
    inj.tr_idx = tr.index.values
    keys = pd.Series(pair_key(te), index=te.index)
    pid = keys.map({k: i for i, k in enumerate(dict.fromkeys(keys))})
    inp = te[["entity", "chunk_text", "coding_system"]].assign(pair_id=pid.values).reset_index(drop=True)
    pred, _ = tm.predict(inp, inj)
    ok = pred.code.values == te.code.values
    variant = "ft4_tagpred_bi_bi3" + ("_cp" if coupling else "") + "_truesub_llmft"
    tag = "_cpA" if coupling else ""
    oof = pd.read_pickle(f"cache/oof_fast_{variant}{tag}_k40_lr0.03_r900_l63_s3_v7.pkl")
    oof = oof[oof.fold == FOLD]
    print(f"\n===== HELD-OUT fold {FOLD} through the FINAL predict path (coupling={coupling}) =====")
    for s in SYSTEMS:
        m = te.coding_system.values == s
        mo = (oof.coding_system == s).values
        print(f"  {s:<12} n={m.sum():<5} predict-path acc={ok[m].mean():.4f}   CV acc={(oof.pred.values[mo] == oof.code.values[mo]).mean():.4f}")
    agree = (pred.code.values == oof.pred.reindex(te.index).values).mean()
    print(f"  same prediction as CV for {agree:.1%} of rows")


if __name__ == "__main__":
    main(coupling="--coupling" in sys.argv, retrain="--no-train" not in sys.argv)
