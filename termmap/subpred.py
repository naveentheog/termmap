"""Predict subconcept (Diagnosis / Procedure / Lab / Encounter / Device) from entity + context.

The assessment file has only entity + chunk_text, so the real subconcept is missing. Without it SNOMED
drops ~1.9 points. Here a small text classifier guesses it, and the ranker gets the guessed
probabilities instead of the true label.
"""
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .text import clean

SUBS = ["DiagnosisCode", "ProcedureCode", "LabService", "EncounterType", "Device"]


def _window(entity, chunk, width=8):
    e, words = clean(entity), clean(chunk).split()
    text = " ".join(words)
    pos = text.find(e) if e else -1
    if pos < 0:
        return " ".join(words[: 2 * width])
    s = len(text[:pos].split())
    return " ".join(words[max(0, s - width): s + len(e.split()) + width])


class SubconceptModel:
    def fit(self, entities, chunks, labels):
        y = pd.Series(labels).where(lambda s: s.isin(SUBS), "DiagnosisCode").values
        self.v_ent = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
        self.v_ew = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.v_ctx = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
        X = self._X(entities, chunks, fit=True)
        self.clf = LogisticRegression(C=4.0, max_iter=2000).fit(X, y)
        return self

    def _X(self, entities, chunks, fit=False):
        ent = [clean(e) for e in entities]
        ctx = [_window(e, c) for e, c in zip(entities, chunks)]
        f = (lambda v, x: v.fit_transform(x)) if fit else (lambda v, x: v.transform(x))
        return hstack([f(self.v_ent, ent), f(self.v_ew, ent), 0.5 * f(self.v_ctx, ctx)]).tocsr()

    def predict_proba(self, entities, chunks):
        P = self.clf.predict_proba(self._X(entities, chunks))
        return pd.DataFrame(P, columns=self.clf.classes_).reindex(columns=SUBS, fill_value=0.0)


def oof_subconcept(train, n_folds=5):
    """Out-of-fold subconcept probabilities for every training row (same chunk-grouped folds)."""
    out = pd.DataFrame(0.0, index=train.index, columns=SUBS)
    for f in range(n_folds):
        tr, te = train[train.fold != f], train[train.fold == f]
        m = SubconceptModel().fit(tr.entity, tr.chunk_text, tr.subconcept)
        out.loc[te.index] = m.predict_proba(te.entity, te.chunk_text).values
    return out


def tag_given_sub(rows, tag_of):
    c = defaultdict(Counter)
    for sub, code in zip(rows.subconcept, rows.code):
        c[sub][tag_of.get(code, "unknown")] += 1
    return c


def add_subpred(F, train, probs, tag_of, folds_n=5, stats_rows=None):
    """Adds sp_<sub> probabilities, sp_top and sp_p_tag = sum_s P(s|query) * P(tag(candidate)|s).
    CV: P(tag|s) comes from the other folds' SNOMED rows only.
    Inference: pass stats_rows=<full training data>; F.qid then indexes the query table."""
    F = F.copy()
    if stats_rows is not None:
        train = stats_rows.assign(fold=-1)
        folds_n = 1
    for s in SUBS:
        F[f"sp_{s}"] = F.qid.map(probs[s]).astype(np.float32)
    F["sp_top"] = F.qid.map(pd.Series(probs.values.argmax(1), index=probs.index)).astype(np.float32)
    sno = train[train.coding_system == "SNOMEDCT_US"]
    tags = F.code.map(tag_of).fillna("unknown").values
    p = np.zeros(len(F), dtype=np.float32)
    fold_of_q = np.zeros(len(F), dtype=int) if stats_rows is not None else F.qid.map(train.fold).values
    for f in range(folds_n):
        cnt = tag_given_sub(sno[sno.fold != f], tag_of)
        m = fold_of_q == f
        qs = F.qid.values[m]
        acc = np.zeros(m.sum(), dtype=np.float32)
        for s in SUBS:
            c = cnt.get(s, Counter())
            tot = sum(c.values()) + 40
            pt = np.array([(c.get(t, 0) + 1) / tot for t in tags[m]], dtype=np.float32)
            acc += probs[s].reindex(qs).values.astype(np.float32) * pt
        p[m] = acc
    F["sp_p_tag"] = p
    return F


TAGS = ["finding", "procedure", "disorder", "observable entity", "physical object", "regime/therapy",
        "qualifier value", "situation", "substance", "record artifact", "body structure", "event"]


class TagModel(SubconceptModel):
    """Same text classifier, but predicts the SNOMED semantic tag of the gold concept."""

    def fit(self, entities, chunks, tags):
        y = pd.Series(list(tags)).where(lambda s: s.isin(TAGS), "other").values
        self.v_ent = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
        self.v_ew = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.v_ctx = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
        self.clf = LogisticRegression(C=4.0, max_iter=3000).fit(self._X(entities, chunks, fit=True), y)
        return self

    def predict_proba(self, entities, chunks):
        P = self.clf.predict_proba(self._X(entities, chunks))
        return pd.DataFrame(P, columns=self.clf.classes_).reindex(columns=TAGS + ["other"], fill_value=0.0)


def oof_tags(train, tag_of, n_folds=5):
    sno = train[train.coding_system == "SNOMEDCT_US"]
    gold = sno.code.map(tag_of).fillna("other")
    out = pd.DataFrame(np.nan, index=train.index, columns=TAGS + ["other"])
    for f in range(n_folds):
        tr, te = sno[sno.fold != f], train[train.fold == f]
        m = TagModel().fit(tr.entity, tr.chunk_text, gold[tr.index])
        out.loc[te.index] = m.predict_proba(te.entity, te.chunk_text).values
    return out


def add_tagpred(F, probs, tag_of):
    """tp_cand = predicted probability that the answer has THIS candidate's semantic tag."""
    F = F.copy()
    tags = F.code.map(tag_of).fillna("other").where(lambda s: s.isin(TAGS), "other").values
    P = probs.reindex(F.qid.values)
    cols = {c: i for i, c in enumerate(P.columns)}
    F["tp_cand"] = P.values[np.arange(len(F)), [cols[t] for t in tags]].astype(np.float32)
    F["tp_max"] = P.max(axis=1).values.astype(np.float32)
    F["tp_is_top"] = (F.tp_cand >= F.tp_max - 1e-6).astype(np.float32)
    F["tp_ratio"] = (F.tp_cand / (F.tp_max + 1e-6)).astype(np.float32)
    return F
