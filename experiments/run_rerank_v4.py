"""v4: hybrid candidates + extra convention features + LightGBM variants. All numbers out-of-fold.

python run_rerank_v4.py        (needs cache/cands_hybrid.pkl and cache/features_hybrid.pkl)
"""
import os
import pickle
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from termmap.data import dictionaries, SYSTEMS, SNO, ICD
from termmap.features import CodeInfo
from termmap.features_extra import add_extra
from termmap.cv import accuracy_report, N_FOLDS

BASE = dict(objective="lambdarank", metric="ndcg", eval_at=[1], lambdarank_truncation_level=10,
            min_child_samples=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            seed=0, deterministic=True, force_row_wise=True, verbose=-1, num_threads=2)
VARIANTS = {
    "v3 params (lr .05, 400r, 31 leaves)": (dict(learning_rate=0.05, num_leaves=31), 400, False),
    "lr .03, 900r, 63 leaves":             (dict(learning_rate=0.03, num_leaves=63), 900, False),
    "pooled systems, lr .03, 900r":        (dict(learning_rate=0.03, num_leaves=63), 900, True),
}


def run(F, train, feats, params, rounds, pooled):
    pred = pd.Series(index=train.index, dtype=object)
    conf = pd.Series(0.0, index=train.index)
    groups = [None] if pooled else list(SYSTEMS)
    for s in groups:
        for f in range(N_FOLDS):
            sel = (F.system == s) if s else np.ones(len(F), bool)
            tr = F[sel & (F.fold != f)]
            tr = tr[tr.groupby("qid").label.transform("max") == 1]
            va = F[sel & (F.fold == f)]
            ds = lgb.Dataset(tr[feats], tr.label, group=tr.groupby("qid", sort=False).size().values)
            m = lgb.train({**BASE, **params}, ds, rounds)
            va = va.assign(score=m.predict(va[feats])).sort_values(["qid", "score", "code"],
                                                                  ascending=[True, False, True])
            va["pos"] = va.groupby("qid").cumcount()
            first = va[va.pos == 0].set_index("qid")
            second = va[va.pos == 1].set_index("qid").score
            pred.loc[first.index] = first.code
            conf.loc[first.index] = (first.score - second.reindex(first.index)).fillna(9.0).values
    return pred, conf


def main():
    with open("cache/cands_hybrid.pkl", "rb") as fh:
        train = pickle.load(fh)["train"]
    extra_path = "cache/features_v4.pkl"
    if os.path.exists(extra_path):
        F = pd.read_pickle(extra_path)
    else:
        t0 = time.time()
        F = pd.read_pickle("cache/features_hybrid.pkl")
        D = dictionaries("data/icd.xlsx", "data/snomed.xlsx")
        info = {s: CodeInfo(D[s], s) for s in SYSTEMS}
        F = add_extra(F, train, info, N_FOLDS)
        F.to_pickle(extra_path)
        print(f"[v4] extra features built in {time.time() - t0:.0f}s")
    F["fold"] = F.qid.map(train.fold)
    F["system"] = F.qid.map(train.coding_system)
    F["is_snomed"] = (F.system == SNO).astype(float)
    base_feats = [c for c in F.columns if c not in ("qid", "code", "label", "fold", "system") and not c.startswith("x_")
                  and c != "is_snomed"]
    all_feats = base_feats + [c for c in F.columns if c.startswith("x_")] + ["is_snomed"]
    print(f"[v4] {len(base_feats)} base features, {len(all_feats)} with extras")

    results = []
    for name, (params, rounds, pooled) in VARIANTS.items():
        for label, feats in (("base", base_feats), ("+extra", all_feats)):
            if name.startswith("v3") and label == "+extra":
                pass
            t0 = time.time()
            pred, conf = run(F, train, feats, params, rounds, pooled)
            train["pred"] = pred
            ok = train.pred == train.code
            a = {s: ok[train.coding_system == s].mean() for s in SYSTEMS}
            tail = {s: ok[(train.coding_system == s) & ~train.seen].mean() for s in SYSTEMS}
            results.append({"model": f"{name} | {label}", "ICD": a[ICD], "SNOMED": a[SNO],
                            "headline": min(a.values()), "tail_ICD": tail[ICD], "tail_SNO": tail[SNO],
                            "sec": round(time.time() - t0)})
            print(results[-1], flush=True)
            if not hasattr(main, "best") or results[-1]["headline"] > main.best[0]:
                main.best = (results[-1]["headline"], name, label, pred.copy(), conf.copy())
    print("\n===== LEADERBOARD v4 =====")
    print(pd.DataFrame(results).round(4).to_string(index=False))
    h, name, label, pred, conf = main.best
    train["pred_lgb"], train["conf"] = pred, conf
    accuracy_report(train, "pred_lgb", f"BEST: {name} | {label}")
    train.to_pickle("cache/oof_v4.pkl")


if __name__ == "__main__":
    main()
