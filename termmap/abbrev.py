"""Abbreviation features (idea from the DrivenData SNOMED entity-linking winners, who used abbreviation lists).

Short entities like 'chf', 'bph', 'cad' are 6% of rows and only ~44% correct (vs ~58% for the rest).
We mine abbreviations from the SNOMED dictionary itself ('CHF - Congestive heart failure',
'Acquired immune deficiency syndrome (AIDS)'), so no external resource is needed.
"""
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from .text import clean

_A1 = re.compile(r"^([A-Za-z][A-Za-z0-9]{1,6}) - (.+)$")
_A2 = re.compile(r"^(.+?) \(([A-Z][A-Z0-9]{1,6})\)$")


def build_abbrev(snomed_df):
    """abbr (lower) -> {'exp': set of cleaned expansions, 'codes': set of SNOMED codes carrying it}"""
    m = defaultdict(lambda: {"exp": set(), "codes": set()})
    for code, d in zip(snomed_df.code.values, snomed_df.desc.values):
        d = str(d)
        x = _A1.match(d)
        if x and x.group(1).isupper():
            m[x.group(1).lower()]["exp"].add(clean(x.group(2)))
            m[x.group(1).lower()]["codes"].add(code)
            continue
        x = _A2.match(d)
        if x:
            m[x.group(2).lower()]["exp"].add(clean(x.group(1)))
            m[x.group(2).lower()]["codes"].add(code)
    return {k: {"exp": v["exp"], "codes": v["codes"]} for k, v in m.items()}


def _jacc(a, b):
    a, b = set(a.split()), set(b.split())
    return len(a & b) / max(1, len(a | b))


def add_abbrev(F, rows, abbr, descs):
    """descs: {(system, code): [cleaned descriptions]}"""
    F = F.copy()
    ent = rows.entity.map(lambda e: clean(e).replace(" ", ""))
    is_abbr = rows.entity.map(lambda e: bool(re.fullmatch(r"[a-z0-9]{2,6}", clean(e).replace(" ", ""))))
    known = ent.where(is_abbr).map(lambda e: e in abbr if isinstance(e, str) else False)
    F["ab_short"] = F.qid.map(is_abbr).astype(np.float32)
    F["ab_known"] = F.qid.map(known).astype(np.float32)
    F["ab_n_exp"] = F.qid.map(lambda q: len(abbr[ent[q]]["exp"]) if known[q] else 0).astype(np.float32)
    sys_of = rows.coding_system
    hit, sim = np.zeros(len(F), np.float32), np.zeros(len(F), np.float32)
    for i, (q, c) in enumerate(zip(F.qid.values, F.code.values)):
        if not known[q]:
            continue
        a = abbr[ent[q]]
        hit[i] = float(c in a["codes"])
        ds = descs.get((sys_of[q], c), [])[:8]
        sim[i] = max((_jacc(x, d) for x in a["exp"] for d in ds), default=0.0)
    F["ab_code_hit"], F["ab_exp_sim"] = hit, sim
    g = F.groupby("qid").ab_exp_sim
    F["ab_exp_sim_gap"] = (g.transform("max") - F.ab_exp_sim).astype(np.float32)
    return F
