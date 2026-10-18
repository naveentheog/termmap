"""Final TermMap inference engine (used by predict.py).

For every (entity, chunk_text) we predict BOTH an ICD-10-CM and a SNOMED CT code:

 1. retrieval   lexical TF-IDF (dictionary + training terms), base SapBERT / GTE (dictionary + nearest
                training terms) and the fine-tuned SapBERT v2 retriever, merged by reciprocal-rank fusion
 2. features    string / clinical-convention features, code priors, retriever scores        -> top-41
 3. neural      fine-tuned SapBERT v2 + v3 scores, fine-tuned MedCPT cross-encoder (v4, ±25 words)
 4. helpers     predicted SNOMED semantic tag; subconcept fixed to DiagnosisCode (assessment scope)
 5. ranker A    LightGBM LambdaRank (ICD model / pooled model for SNOMED), 3 seeds averaged
 6. LLM         Qwen2.5-7B + LoRA reads term, context, 8 similar coded training terms and ranker A's top-10
 7. ranker B    same features + LLM probabilities (+ ICD<->SNOMED coupling via ranker A's partner code)
Deterministic: no sampling, ties broken by code string, LightGBM deterministic=True.

`neural` is pluggable: GPU implementations by default; the held-out test injects out-of-fold scores.
"""
import json
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import scorefeats as sf
from .data import ICD, SNO, SYSTEMS, dictionaries
from .engine import to_long
from .features import CodeInfo, FoldStats, build_features
from .features_extra import add_extra
from .hybrid import fuse, train_neighbours_query
from .retrieval import DictRetriever, Retriever
from .text import clean

DIAG = "DiagnosisCode"


class FinalMapper:
    def __init__(self, train_df, icd_xlsx, snomed_xlsx, model_dir, assets=None):
        self.cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.train = train_df.reset_index(drop=True)
        self.D = dictionaries(icd_xlsx, snomed_xlsx)
        self.info = {s: CodeInfo(self.D[s], s) for s in SYSTEMS}
        self.stats = FoldStats(self.train, self.info[SNO])
        dret = {s: DictRetriever(self.D[s]) for s in SYSTEMS}
        self.lex = {s: Retriever(dret[s], self.train[self.train.coding_system == s]) for s in SYSTEMS}
        self.rankers = {st: {g: [lgb.Booster(model_file=os.path.join(model_dir, f"ranker_{st}_{g}_s{s}.txt"))
                                 for s in range(self.cfg["seeds"])] for g in (ICD, "pooled")}
                        for st in ("A", "B")}
        aux = pickle.load(open(os.path.join(model_dir, "aux.pkl"), "rb"))
        self.tag_of, self.descs = aux["tag_of"], aux["descs"]
        from .subpred import TagModel
        sno = self.train[self.train.coding_system == SNO]
        self.tag_model = TagModel().fit(sno.entity, sno.chunk_text, sno.code.map(self.tag_of).fillna("other"))
        self.fallback = self.train.groupby("coding_system").code.agg(lambda s: s.value_counts().index[0]).to_dict()
        self.assets = assets or {}

    def _rank(self, stage, F, sys_of):
        feats = self.cfg[f"features_{stage}"]
        for col in feats:
            if col not in F:
                F[col] = np.nan
        F = F.copy()
        F["score"] = 0.0
        for s, g in ((ICD, ICD), (SNO, "pooled")):
            sel = sys_of == s
            if sel.any():
                X = F.loc[sel, feats]
                F.loc[sel, "score"] = np.mean([m.predict(X) for m in self.rankers[stage][g]], axis=0)
        F = F.sort_values(["qid", "score", "code"], ascending=[True, False, True])
        F["pos"] = F.groupby("qid").cumcount()
        return F

    @staticmethod
    def _top(F):
        first = F[F.pos == 0].set_index("qid")
        second = F[F.pos == 1].set_index("qid").score.reindex(first.index)
        top10 = F[F.pos < 10].groupby("qid").code.apply(list).to_dict()
        return first.code, (first.score - second).fillna(9.0), top10

    def predict(self, df, neural, verbose=True):
        """neural: object with dense(q), bi(q, name), ce(q, codes_by_qid), llm(q, top10) (see neural_gpu.py)."""
        log = (lambda m: print(f"[final] {m}", flush=True)) if verbose else (lambda m: None)
        q, mode = to_long(df)
        if "pair_id" in df and mode == "long":      # tests: rows of the same entity+chunk in both systems
            q["src_row"] = df["pair_id"].values
        q["subconcept"] = DIAG
        q["fold"] = 0
        lex = {}
        for s in SYSTEMS:
            idx = np.flatnonzero(q.coding_system.values == s)
            if len(idx):
                lex.update(zip(idx, self.lex[s].candidates([clean(e) for e in q.entity.values[idx]], k=50)))
        dense_rows, neigh = neural.dense(q)
        bi2 = neural.bi(q, "bi")
        bi3 = neural.bi(q, "bi3")
        log(f"retrieval done for {len(q)} queries")
        sap, gte = neigh.get("sapbert", [None] * len(q)), neigh.get("gte", [None] * len(q))
        parts = []
        for i in range(len(q)):
            c = fuse(lex.get(i), dense_rows[i], sap[i], gte[i], {"bi_dict": bi2[i]})
            if len(c) == 0:
                continue
            s = q.coding_system.iat[i]
            f = build_features(q.entity.iat[i], q.chunk_text.iat[i], DIAG, c, self.info[s], self.stats)
            f.insert(0, "qid", i)
            f.insert(1, "code", c.code.values)
            parts.append(f)
        F = pd.concat(parts, ignore_index=True)
        F = add_extra(F, q, self.info, {0: self.train})
        F = F[F["rank"] <= self.cfg["topk"]].reset_index(drop=True)
        log(f"{len(F):,} candidate rows")
        F = sf.add_ce(F, neural.ce(q, F.groupby("qid", sort=True).code.apply(list).to_dict()), "ft4")
        F = sf.add_bi(F, dict(enumerate(bi2)), "bi")
        F = sf.add_bi(F, dict(enumerate(bi3)), "bi3")
        F["is_snomed"] = (F.qid.map(q.coding_system) == SNO).astype(float)
        from .subpred import add_tagpred
        tp = self.tag_model.predict_proba(q.entity, q.chunk_text)
        tp.index = q.index
        F = add_tagpred(F, tp, self.tag_of)
        sys_of = F.qid.map(q.coding_system).values
        FA = self._rank("A", F, sys_of)
        predA, confA, top10 = self._top(FA)
        log("ranker A done")
        if self.cfg.get("coupling"):
            from .coupling import add_coupling
            other = {ICD: SNO, SNO: ICD}
            key = {(r, s): i for i, (r, s) in enumerate(zip(q.src_row.values, q.coding_system.values))}
            partner = {}
            for i, (r, s) in enumerate(zip(q.src_row.values, q.coding_system.values)):
                j = key.get((r, other[s]))
                if j is not None and j in predA.index:
                    partner[i] = predA[j]
            F = add_coupling(F, q, partner, self.train, self.descs)
        if getattr(neural, "use_llm", True):
            F = sf.add_llm(F, neural.llm(q, top10), "llmft")
            log("LLM scored")
            FB = self._rank("B", F, sys_of)
            pred, conf, _ = self._top(FB)
        else:                                       # no GPU / no adapter: ranker A is the answer
            log("LLM skipped -> ranker A")
            pred, conf = predA, confA
        q["code"] = pred.reindex(q.index)
        q["confidence"] = conf.reindex(q.index).round(4)
        q["code_stageA"] = predA.reindex(q.index)
        q["code"] = q.code.fillna(q.coding_system.map(self.fallback))
        return q, mode
