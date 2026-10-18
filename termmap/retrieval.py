"""Candidate generation for one coding system.

Sources (each returns per-code cosine scores):
  dict_char  : char n-gram TF-IDF over dictionary descriptions   (robust to typos/OCR)
  dict_word  : word uni+bi-gram TF-IDF over dictionary descriptions (exact clinical words)
  train_char : char n-gram TF-IDF over training entities of this fold, mapped to their gold code
               ("training terms as extra synonyms" - learns HiLabs coders' local convention)
  dense      : optional precomputed SapBERT/GTE neighbours (from the Kaggle notebook)

Candidates = union of each source's top-k, scored per code by the max over its descriptions.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .text import clean

SOURCES = ("dict_char", "dict_word", "train_char")


def topk_rows(sim, k):
    sim = sim.tocsr()
    out = []
    for i in range(sim.shape[0]):
        s, e = sim.indptr[i], sim.indptr[i + 1]
        data, idx = sim.data[s:e], sim.indices[s:e]
        if len(data) > k:
            part = np.argpartition(-data, k)[:k]
            data, idx = data[part], idx[part]
        o = np.argsort(-data, kind="stable")
        out.append((idx[o], data[o]))
    return out


class TfidfIndex:
    def __init__(self, texts, keys, kind="char"):
        if kind == "char":
            self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True,
                                       min_df=1, max_df=0.05, dtype=np.float32)
        else:
            self.vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
                                       token_pattern=r"[a-z0-9+]+", min_df=1, dtype=np.float32)
        self.mat = self.vec.fit_transform(texts).T.tocsr()   # features x docs
        self.keys = np.asarray(keys)

    def search(self, queries, k, batch=64):
        res = []
        for b in range(0, len(queries), batch):
            q = self.vec.transform(queries[b:b + batch])
            for idx, sc in topk_rows(q @ self.mat, k * 4):
                best = {}
                for i, s in zip(idx, sc):
                    c = self.keys[i]
                    if s > best.get(c, 0):
                        best[c] = float(s)
                res.append(dict(sorted(best.items(), key=lambda kv: -kv[1])[:k]))
        return res


class DictRetriever:
    """Dictionary-level indexes for one system; built once, reused across folds."""

    def __init__(self, dict_df):
        texts = [clean(t) for t in dict_df.desc]
        print(f"[retrieval] indexing {len(texts):,} descriptions ...", flush=True)
        self.char = TfidfIndex(texts, dict_df.code.values, "char")
        self.word = TfidfIndex(texts, dict_df.code.values, "word")


class Retriever:
    def __init__(self, dict_ret: DictRetriever, train_rows: pd.DataFrame, dense=None):
        self.d = dict_ret
        self.train = None
        if len(train_rows):
            self.train = TfidfIndex([clean(t) for t in train_rows.entity], train_rows.code.values, "char")
        self.dense = dense   # callable(queries) -> list[dict code->score], optional

    def candidates(self, queries, k=50, per_source=50):
        """-> list (per query) of DataFrame[code, dict_char, dict_word, train_char, (dense), fused]"""
        srcs = {"dict_char": self.d.char.search(queries, per_source),
                "dict_word": self.d.word.search(queries, per_source)}
        if self.train is not None:
            srcs["train_char"] = self.train.search(queries, per_source)
        if self.dense is not None:
            srcs["dense"] = self.dense(queries)
        out = []
        for i in range(len(queries)):
            codes = set().union(*(s[i].keys() for s in srcs.values()))
            df = pd.DataFrame({"code": sorted(codes)})
            for name, s in srcs.items():
                df[name] = df.code.map(s[i]).fillna(0.0).astype(np.float32)
            score_cols = [c for c in srcs]
            df["fused"] = df[score_cols].max(axis=1) + 0.1 * df[score_cols].mean(axis=1)
            df = df.sort_values(["fused", "code"], ascending=[False, True]).head(k).reset_index(drop=True)
            df["rank"] = np.arange(len(df))
            out.append(df)
        return out
