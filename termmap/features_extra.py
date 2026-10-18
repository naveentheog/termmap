"""Extra ranking features added on top of features.py (v4).

Targets the failure modes found in the error analysis:
  * exact / substring name match      ("d dimer" -> "D-dimer" rather than "D-dimer assay")
  * head word (last token) match       (the clinical noun usually carries the concept)
  * specificity: does the term give detail (digits, laterality, acuity, site words)? vs "unspecified"
  * ICD block voting: many candidates / train neighbours in the same 3-char block
  * code prior given subconcept (local coder convention), from the training fold only
"""
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .data import ICD, SNO
from .text import clean

DETAIL = re.compile(r"\b(left|right|bilateral|bilat|lt|rt|acute|chronic|stage|grade|type|with|w|due|"
                    r"recurrent|severe|moderate|mild|upper|lower|initial|subsequent|\d+)\b")
HISTORY = re.compile(r"\b(history|h o|hx|s p|status post|prior|previous|old)\b")
NEG = re.compile(r"\b(no|denies|denied|without|negative|absent|not|ø)\b")


def add_extra(F: pd.DataFrame, train: pd.DataFrame, info: dict, folds_n) -> pd.DataFrame:
    """train: query rows (entity, coding_system, subconcept, fold) indexed by qid.
    folds_n: int (CV, priors from other folds) or {fold: prior_rows} (inference)."""
    F = F.copy()
    q_norm = train.entity.map(clean)
    sys_of = train.coding_system
    exact, sub_q_in_d, sub_d_in_q, head = [], [], [], []
    hist_c, neg_c = [], []
    for qid, code in zip(F.qid.values, F.code.values):
        q = q_norm.at[qid]
        descs = info[sys_of.at[qid]].descs.get(code, [])[:15]
        exact.append(float(q in descs))
        sub_q_in_d.append(float(any(q and q in d for d in descs)))
        sub_d_in_q.append(float(any(d and d in q for d in descs)))
        hw = q.split()[-1] if q else ""
        head.append(float(bool(hw) and any(hw in d.split() for d in descs)))
        joined = " ".join(descs)
        hist_c.append(float("history" in joined))
        neg_c.append(float(bool(re.search(r"\b(no|absent|negative|without|not)\b", joined))))
    F["x_exact"], F["x_q_in_d"], F["x_d_in_q"], F["x_head"] = exact, sub_q_in_d, sub_d_in_q, head
    q_hist = F.qid.map(lambda i: float(bool(HISTORY.search(q_norm.at[i]))))
    q_neg = F.qid.map(lambda i: float(bool(NEG.search(q_norm.at[i]))))
    q_detail = F.qid.map(lambda i: float(len(DETAIL.findall(q_norm.at[i]))))
    F["x_hist_match"] = q_hist * np.array(hist_c)
    F["x_hist_conflict"] = (1 - q_hist) * np.array(hist_c)
    F["x_neg_match"] = q_neg * np.array(neg_c)
    F["x_neg_conflict"] = (1 - q_neg) * np.array(neg_c)
    F["x_q_detail"] = q_detail
    F["x_unspec_x_detail"] = F["unspec"] * q_detail

    is_icd = F.qid.map(sys_of).values == ICD
    F["block"] = np.where(is_icd, F.code.str[:3], F.code)
    for col in ("sap_votes", "gte_votes"):
        F[f"x_block_{col}"] = F.groupby(["qid", "block"])[col].transform("sum")
    F["x_block_best_rank"] = F.groupby(["qid", "block"])["rank"].transform("min")
    F["x_block_size"] = F.groupby(["qid", "block"])["code"].transform("size")
    F["x_rank_in_block"] = F.groupby(["qid", "block"])["rank"].rank(method="first")

    fold_of = train.fold
    F["x_code_sub_prior"] = 0.0
    F["x_block_prior"] = 0.0
    sub_of = train.subconcept
    if not isinstance(folds_n, dict):        # CV: priors for fold f come from the other folds
        folds_n = {f: train[train.fold != f] for f in range(folds_n)}
    for f, tr in folds_n.items():           # inference: {0: full training data}
        cs = Counter(zip(tr.subconcept, tr.code))
        bl = Counter(tr[tr.coding_system == ICD].code.str[:3])
        m = F.qid.map(fold_of).values == f
        sub = F.loc[m, "qid"].map(sub_of)
        F.loc[m, "x_code_sub_prior"] = np.log1p([cs.get((s, c), 0) for s, c in zip(sub, F.loc[m, "code"])])
        F.loc[m, "x_block_prior"] = np.log1p([bl.get(b, 0) for b in F.loc[m, "block"]])
    return F.drop(columns=["block"])
