"""Faster, resumable out-of-fold experiments (the 2-CPU sandbox kills jobs after 2 h).

python run_fast.py <variant> [--topk 40]
variants: v4 | ce | nosub | ce_nosub

- Keeps only the top-K hybrid candidates per query (K=40 by default) -> ~2.5x fewer rows.
- Final policy: ICD from an ICD-only model, SNOMED from the pooled model.
- Each (model, fold) result is checkpointed to cache/fast_<variant>_k<K>.pkl; rerun to resume.
"""
import argparse
import os
import pickle
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import run_rerank_v4 as v4
from run_rerank_ce import add_ce
from termmap.data import SYSTEMS, ICD, SNO

SUB_FEATS = {"sub_id", "p_tag", "x_code_sub_prior"}


def load(variant, topk, base="v4"):
    cpath, fpath = ("cache/cands_hybrid.pkl", "cache/features_v4.pkl") if base == "v4" else \
        (f"cache/cands_{base}.pkl", f"cache/features_{base}.pkl")
    with open(cpath, "rb") as fh:
        train = pickle.load(fh)["train"]
    F = pd.read_pickle(fpath)
    F = F[F["rank"] <= topk].reset_index(drop=True)
    toks = variant.split("_")
    for tag, path in (("ft", "cache/ft_scores.pkl"), ("ft2", "cache/ft2_scores.pkl"), ("ft3", "cache/ft3_scores.pkl"), ("ft4", "cache/ft4_scores.pkl")):
        if tag not in toks:
            continue  # fine-tuned cross-encoder scores from Kaggle (out-of-fold)
        with open(path, "rb") as fh:
            ft = pickle.load(fh)
        x = f"x_{tag}"
        F[x] = np.asarray([ft.get(q, {}).get(c, np.nan) for q, c in zip(F.qid.values, F.code.values)],
                          dtype=np.float32)
        g = F.groupby("qid")[x]
        F[f"{x}_rank"] = g.rank(ascending=False, method="min")
        F[f"{x}_gap"] = g.transform("max") - F[x]
        e = np.exp(F[x] - g.transform("max"))
        F[f"{x}_prob"] = e / e.groupby(F.qid).transform("sum")
    for tag, path in (("bi", "cache/bienc_scores.pkl"), ("bi3", "cache/bienc3_scores.pkl")):
        if tag not in toks:
            continue  # fine-tuned SapBERT retriever (Kaggle, out-of-fold): score + rank for each candidate
        with open(path, "rb") as fh:
            bi = pickle.load(fh)
        brank = {q: {c: r for r, c in enumerate(d)} for q, d in bi.items()}
        x = f"x_{tag}"
        F[x] = np.asarray([bi.get(q, {}).get(c, np.nan) for q, c in zip(F.qid.values, F.code.values)],
                          dtype=np.float32)
        F[f"{x}_rank"] = np.asarray([brank.get(q, {}).get(c, 999) for q, c in zip(F.qid.values, F.code.values)],
                                    dtype=np.float32)
        top = F.qid.map({q: (max(d.values()) if d else np.nan) for q, d in bi.items()})
        F[f"{x}_gap"] = (top - F[x]).astype(np.float32)
    for tag, path in (("llm", "cache/llm_scores.pkl"), ("llmft", "cache/llm_ft_scores.pkl")):
        if tag not in toks:
            continue  # LLM adjudicator probabilities over the ranker's top-10 (out-of-fold / zero-shot)
        sc = pickle.load(open(path, "rb"))
        x = f"x_{tag}"
        F[x] = np.asarray([sc.get(q, {}).get(c, np.nan) for q, c in zip(F.qid.values, F.code.values)],
                          dtype=np.float32)
        top = F.qid.map({q: (max(d.values()) if d else np.nan) for q, d in sc.items()})
        F[f"{x}_gap"] = (top - F[x]).astype(np.float32)
        F[f"{x}_is_top"] = (F[f"{x}_gap"] <= 1e-6).astype(np.float32)
    if "ce_" in variant or variant == "ce":
        with open("cache/ce_scores.pkl", "rb") as fh:
            F = add_ce(F, pickle.load(fh))
    F["fold"] = F.qid.map(train.fold)
    F["system"] = F.qid.map(train.coding_system)
    F["is_snomed"] = (F.system == SNO).astype(float)
    feats = [c for c in F.columns if c not in ("qid", "code", "label", "fold", "system")]
    if "subpred" in variant:
        from termmap.data import dictionaries
        from termmap.features import CodeInfo
        from termmap.subpred import add_subpred
        tag_of = pickle.load(open("cache/sno_tag_of.pkl", "rb"))
        F = add_subpred(F, train, pd.read_pickle("cache/subpred_oof.pkl"), tag_of)
        feats = [c for c in F.columns if c not in ("qid", "code", "label", "fold", "system")]
    if "abbr" in variant:
        from termmap.abbrev import add_abbrev
        ab = pickle.load(open("cache/abbrev.pkl", "rb"))
        F = add_abbrev(F, train, ab["abbr"], ab["descs"])
        feats = [c for c in F.columns if c not in ("qid", "code", "label", "fold", "system")]
    if "cp" in toks:  # couple ICD <-> SNOMED through the partner row's out-of-fold prediction
        from termmap.coupling import add_coupling, pair_key
        part = pd.read_pickle(os.environ.get("CP_PARTNER", "cache/oof_fast_ft2_subpred_tagpred_bi_k40_lr0.03_r900_l63_s3.pkl"))
        keys = pd.Series(pair_key(part), index=part.index)
        by = {(k, s): p for k, s, p in zip(keys.values, part.coding_system.values, part.pred.values)}
        other = {"ICD10CM": "SNOMEDCT_US", "SNOMEDCT_US": "ICD10CM"}
        partner = {q: by[(k, other[s])] for q, k, s in zip(part.index, keys.values, part.coding_system.values)
                   if (k, other[s]) in by}
        ab = pickle.load(open("cache/abbrev.pkl", "rb"))
        F = add_coupling(F, train, partner, train, ab["descs"], fold_of=train.fold)
        feats = [c for c in F.columns if c not in ("qid", "code", "label", "fold", "system")]
        print(f"[cp] partner prediction for {len(partner)} of {len(train)} queries", flush=True)
    if "tagpred" in variant:
        from termmap.subpred import add_tagpred
        tag_of = pickle.load(open("cache/sno_tag_of.pkl", "rb"))
        F = add_tagpred(F, pd.read_pickle("cache/tagpred_oof.pkl"), tag_of)
        feats = [c for c in F.columns if c not in ("qid", "code", "label", "fold", "system")]
    if (variant.endswith("nosub") or "subpred" in variant) and "truesub" not in toks:
        feats = [c for c in feats if c not in SUB_FEATS]
    return train, F, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--rounds", type=int, default=900)
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--seeds", type=int, default=1, help="average the scores of N differently-seeded models")
    ap.add_argument("--diag-only", action="store_true", help="train rankers on DiagnosisCode rows only")
    ap.add_argument("--diag-weight", type=float, default=1.0, help="row weight for DiagnosisCode queries")
    ap.add_argument("--base", default="v4", help="candidate/feature set: v4 (hybrid) or v7 (+ fine-tuned SapBERT)")
    a = ap.parse_args()
    tag = "" if (a.lr, a.rounds, a.leaves, a.seeds) == (0.03, 900, 63, 1) else f"_lr{a.lr}_r{a.rounds}_l{a.leaves}_s{a.seeds}"
    if a.base != "v4":
        tag += f"_{a.base}"
    if a.diag_only:
        tag += "_diagonly"
    if a.diag_weight != 1.0:
        tag += f"_dw{a.diag_weight}"
    ck_path = f"cache/fast_{a.variant}_k{a.topk}{tag}.pkl"
    ck = pickle.load(open(ck_path, "rb")) if os.path.exists(ck_path) else {}
    train, F, feats = load(a.variant, a.topk, a.base)
    gold_in = F.groupby("qid").label.max().reindex(train.index).fillna(0)
    print(f"[{a.variant}] top-{a.topk}: {len(F):,} rows, {len(feats)} feats, gold in shortlist "
          f"{gold_in[train.coding_system == ICD].mean():.3f} / {gold_in[train.coding_system == SNO].mean():.3f}",
          flush=True)
    params, rounds = dict(learning_rate=a.lr, num_leaves=a.leaves), a.rounds
    for group in (ICD, "pooled"):
        for f in range(v4.N_FOLDS):
            key = (group, f)
            if key in ck:
                continue
            t0 = time.time()
            sel = (F.system == ICD) if group == ICD else np.ones(len(F), bool)
            tr = F[sel & (F.fold != f)]
            tr = tr[tr.groupby("qid").label.transform("max") == 1]
            is_diag = tr.qid.map(train.subconcept).values == "DiagnosisCode"
            if a.diag_only:
                tr, is_diag = tr[is_diag], is_diag[is_diag]
            va = F[sel & (F.fold == f)]
            if group == "pooled":
                va = va[va.system == SNO]
            w = np.where(is_diag, a.diag_weight, 1.0) if a.diag_weight != 1.0 else None
            ds = lgb.Dataset(tr[feats], tr.label, weight=w, group=tr.groupby("qid", sort=False).size().values)
            score = np.zeros(len(va))
            for sd in range(a.seeds):
                m = lgb.train({**v4.BASE, **params, "seed": sd}, ds, rounds)
                score += m.predict(va[feats]) / a.seeds
            va = va.assign(score=score).sort_values(["qid", "score", "code"],
                                                                  ascending=[True, False, True])
            va["pos"] = va.groupby("qid").cumcount()
            first = va[va.pos == 0].set_index("qid")
            second = va[va.pos == 1].set_index("qid").score.reindex(first.index)
            top10 = va[va.pos < 10].groupby("qid").apply(lambda d: list(zip(d.code, d.score.round(4))))
            ck[key] = pd.DataFrame({"pred": first.code, "margin": (first.score - second).fillna(9.0),
                                    "top10": top10.reindex(first.index)})
            pickle.dump(ck, open(ck_path, "wb"))
            print(f"  {group} fold {f}: {time.time() - t0:.0f}s", flush=True)
    out = pd.concat(ck.values())
    train["pred"], train["margin"] = out.pred.reindex(train.index), out.margin.reindex(train.index)
    if "top10" in out:
        train["top10"] = out.top10.reindex(train.index)
    ok = train.pred == train.code
    r = {s: round(ok[train.coding_system == s].mean(), 4) for s in SYSTEMS}
    diag = {s: round(ok[(train.coding_system == s) & (train.subconcept == "DiagnosisCode")].mean(), 4) for s in SYSTEMS}
    print(f"RESULT {a.variant} k{a.topk}{tag}", r, "headline", min(r.values()), "| diagnosis", diag, flush=True)
    train.to_pickle(f"cache/oof_fast_{a.variant}_k{a.topk}{tag}.pkl")


if __name__ == "__main__":
    main()
