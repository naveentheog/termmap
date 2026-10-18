"""Features for each (term, candidate code) pair, used by the LightGBM ranker."""
import math
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .data import ICD, SNO
from .text import clean

LAT = {"left": "L", "lt": "L", "right": "R", "rt": "R", "bilateral": "B", "bilat": "B", "both": "B"}
ACUITY = {"acute": "A", "chronic": "C", "subacute": "S"}
STOP = set("of the and with without in on to for a an or by due other unspecified nos".split())
SUBCONCEPTS = ["DiagnosisCode", "ProcedureCode", "LabService", "EncounterType", "Device",
               "MedicationProfileMedication"]


def toks(t):
    return [w for w in t.split() if w not in STOP]


def flags(text, table):
    return {v for k, v in table.items() if re.search(rf"\b{k}\b", text)}


class CodeInfo:
    """Static info per code: cleaned descriptions, canonical desc, tag, leaf, specificity."""

    def __init__(self, dict_df, system):
        self.system = system
        d = dict_df.assign(text=dict_df.desc.map(clean))
        self.descs = d.groupby("code").text.agg(list).to_dict()
        if system == SNO:
            first = d.sort_values("order").drop_duplicates("code")
            self.canon = first.set_index("code").text.to_dict()
            self.tag = d.drop_duplicates("code").set_index("code").tag.to_dict()
            self.leaf = {}
        else:
            first = d.drop_duplicates("code")
            self.canon = first.set_index("code").text.to_dict()
            self.tag = {}
            self.leaf = d.drop_duplicates("code").set_index("code").is_leaf.to_dict()
        self.unspec = {c: any("unspecified" in t or " nos" in t for t in ts) for c, ts in self.descs.items()}


class FoldStats:
    """Statistics learned from the training rows of a fold (never from the row being scored)."""

    def __init__(self, train_rows, info_sno: CodeInfo):
        self.code_count = Counter(train_rows.code)
        self.tag_given_sub = defaultdict(Counter)
        if len(train_rows) and (train_rows.coding_system == SNO).any():
            t = train_rows[train_rows.coding_system == SNO]
            for sub, c in zip(t.subconcept, t.code):
                self.tag_given_sub[sub][info_sno.tag.get(c, "unknown")] += 1

    def p_tag(self, sub, tag):
        c = self.tag_given_sub.get(sub)
        if not c:
            return 0.0
        return (c.get(tag, 0) + 1) / (sum(c.values()) + 40)


def build_features(query, chunk, subconcept, cands: pd.DataFrame, info: CodeInfo, stats: FoldStats):
    q = clean(query)
    qt = set(toks(q))
    ctx = set(toks(clean(chunk))) - qt
    q_lat, q_acu = flags(q, LAT), flags(q, ACUITY)
    rows = []
    for c in cands.code:
        canon = info.canon.get(c, "")
        descs = info.descs.get(c, [canon])
        best_tsr = max(fuzz.token_set_ratio(q, t) for t in descs[:12]) / 100
        best_ratio = max(fuzz.ratio(q, t) for t in descs[:12]) / 100
        dt = set(toks(canon))
        alltoks = set().union(*(set(toks(t)) for t in descs[:12]))
        c_lat, c_acu = flags(" ".join(descs[:12]), LAT), flags(canon, ACUITY)
        code_n = len(c.replace(".", ""))
        tag = info.tag.get(c, "")
        rows.append({
            "tsr": best_tsr, "ratio": best_ratio,
            "q_cov": len(qt & alltoks) / max(len(qt), 1),
            "d_cov": len(qt & dt) / max(len(dt), 1),
            "ctx_overlap": len(ctx & alltoks) / max(len(alltoks), 1),
            "lat_match": float(bool(q_lat) and bool(q_lat & c_lat)),
            "lat_conflict": float(bool(q_lat) and bool(c_lat) and not (q_lat & c_lat)),
            "lat_extra": float(not q_lat and bool(c_lat)),
            "acu_match": float(bool(q_acu & c_acu)),
            "acu_conflict": float(bool(q_acu) and bool(c_acu) and not (q_acu & c_acu)),
            "unspec": float(info.unspec.get(c, False)),
            "leaf": float(info.leaf.get(c, True)),
            "code_len": code_n if info.system == ICD else 0,
            "prior_log": math.log1p(stats.code_count.get(c, 0)),
            "seen": float(stats.code_count.get(c, 0) > 0),
            "p_tag": stats.p_tag(subconcept, tag) if info.system == SNO else 0.0,
            "n_desc": len(descs), "canon_len": len(canon.split()), "q_len": len(q.split()),
            "sub_id": SUBCONCEPTS.index(subconcept) if subconcept in SUBCONCEPTS else -1,
        })
    f = pd.DataFrame(rows)
    num = [k for k in cands.columns if k not in ("code",)]
    out = pd.concat([cands[num].reset_index(drop=True), f], axis=1)
    for col in ("dict_char", "dict_word", "train_char", "dense",
                "sap_dict", "gte_dict", "sap_train", "gte_train"):
        if col in out:
            out[f"{col}_gap"] = out[col].max() - out[col]          # distance to the best in this list
    return out
