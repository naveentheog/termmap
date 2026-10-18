"""TermMapper: the inference engine used by predict.py.

Input rows can come in either evaluation format:
  A) long : entity, coding_system [, chunk_text, subconcept]   -> one code per row
  B) wide : entity / clinical_entity [, chunk_id, chunk_text]  -> one ICD + one SNOMED code per row
Missing chunk_text / subconcept are allowed (features that need them fall back to neutral values).
"""
import os

import numpy as np
import pandas as pd

from . import dense as dn
from . import ranker
from .data import ICD, SNO, SYSTEMS, dictionaries, load_train
from .features import CodeInfo, FoldStats, build_features
from .features_extra import add_extra
from .hybrid import fuse, train_neighbours_query
from .retrieval import DictRetriever, Retriever
from .text import clean


def to_long(df):
    """-> (long query frame, mode). Accepts both evaluation formats."""
    df = df.copy()
    ent = "entity" if "entity" in df else "clinical_entity"
    base = pd.DataFrame({
        "entity": df[ent].astype(str),
        "chunk_text": df["chunk_text"].astype(str) if "chunk_text" in df else "",
        "subconcept": df["subconcept"].astype(str) if "subconcept" in df else "",
        "src_row": np.arange(len(df)),
    })
    if "coding_system" in df:
        base["coding_system"] = df["coding_system"].astype(str).str.strip().values
        return base.reset_index(drop=True), "long"
    parts = [base.assign(coding_system=s) for s in SYSTEMS]
    return pd.concat(parts, ignore_index=True), "wide"


class TermMapper:
    def __init__(self, train_csv, icd_xlsx, snomed_xlsx, cache_dir="cache", model_dir="models",
                 train_df=None, train_emb=None):
        self.cache, self.model_dir = cache_dir, model_dir
        self.train = (train_df if train_df is not None else load_train(train_csv)).reset_index(drop=True)
        self.D = dictionaries(icd_xlsx, snomed_xlsx)
        self.info = {s: CodeInfo(self.D[s], s) for s in SYSTEMS}
        self.stats = FoldStats(self.train, self.info[SNO])
        dret = {s: DictRetriever(self.D[s]) for s in SYSTEMS}
        self.lex = {s: Retriever(dret[s], self.train[self.train.coding_system == s]) for s in SYSTEMS}
        self.train_emb = train_emb or {}
        self.models, self.feats = ranker.load(model_dir)
        self.fallback = self.train.groupby(["coding_system"]).code.agg(lambda s: s.value_counts().index[0]).to_dict()

    def _available_dense(self):
        return [k for k in dn.MODELS if all(os.path.exists(os.path.join(self.cache, f"emb_{k}_{s}.npy"))
                                            for s in SYSTEMS)]

    def _train_embeddings(self, key, model):
        if key in self.train_emb:
            return self.train_emb[key]
        path = os.path.join(self.cache, f"entity_emb_{key}.npy")
        emb = np.load(path) if os.path.exists(path) else None
        if emb is None or len(emb) != len(self.train):
            print(f"[engine] embedding {len(self.train)} training entities with {key} ...")
            emb = dn.embed(key, self.train.entity.tolist(), model)
            np.save(path, emb.astype(np.float16))
        self.train_emb[key] = emb
        return emb

    def _dense(self, q):
        """-> dense_rows (per query {model: {code: score}}), neighbours {model: list}."""
        rows = [dict() for _ in range(len(q))]
        neigh = {}
        for key in self._available_dense():
            model = dn.load_model(key)
            q_emb = dn.embed(key, q.entity.tolist(), model)
            for s in SYSTEMS:
                idx = np.flatnonzero(q.coding_system.values == s)
                if len(idx):
                    for i, d in zip(idx, dn.dict_search(key, s, q_emb[idx], self.cache)):
                        rows[i][key] = d
            neigh[key] = train_neighbours_query(q_emb, q.coding_system.values, self.train,
                                                self._train_embeddings(key, model))
        return rows, neigh

    def predict(self, df, dense_rows=None, neigh=None):
        q, mode = to_long(df)
        q["fold"] = 0
        lex = {}
        for s in SYSTEMS:
            idx = np.flatnonzero(q.coding_system.values == s)
            if len(idx):
                res = self.lex[s].candidates([clean(e) for e in q.entity.values[idx]], k=50)
                lex.update(zip(idx, res))
        if dense_rows is None:
            dense_rows, neigh = self._dense(q)
        neigh = neigh or {}
        sap, gte = neigh.get("sapbert", [None] * len(q)), neigh.get("gte", [None] * len(q))
        parts = []
        for i in range(len(q)):
            c = fuse(lex.get(i), dense_rows[i], sap[i], gte[i])
            if len(c) == 0:
                continue
            s = q.coding_system.iat[i]
            f = build_features(q.entity.iat[i], q.chunk_text.iat[i], q.subconcept.iat[i], c, self.info[s], self.stats)
            f.insert(0, "qid", i)
            f.insert(1, "code", c.code.values)
            parts.append(f)
        F = pd.concat(parts, ignore_index=True)
        F = add_extra(F, q, self.info, {0: self.train})
        for col in self.feats:
            if col not in F:
                F[col] = 0.0
        F = ranker.score(self.models, self.feats, F, q.coding_system)
        F = F.sort_values(["qid", "score", "code"], ascending=[True, False, True])
        F["pos"] = F.groupby("qid").cumcount()
        first = F[F.pos == 0].set_index("qid")
        second = F[F.pos == 1].set_index("qid").score
        q["code"] = first.code.reindex(q.index)
        q["confidence"] = (first.score - second.reindex(first.index)).reindex(q.index).fillna(9.0).round(4)
        q["code"] = q.code.fillna(q.coding_system.map(self.fallback))
        return q, mode
