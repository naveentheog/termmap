"""Context features from chunk_text (v5).

The organizers said their own model fails when context changes the meaning: family history,
screening questionnaires, plans/goals (e.g. 'plan for weight loss'), negation, history of.
We look at a window of words around the entity and pair each cue with what the candidate's
description says, so the ranker can e.g. prefer 'Family history of diabetes' only when the
context mentions a relative.
"""
import re

import numpy as np
import pandas as pd

from .text import clean

CUES = {
    "family": r"\b(family|mother|father|mom|dad|brother|sister|sibling|grandmother|grandfather|aunt|uncle|fhx|fh)\b",
    "history": r"\b(history|h o|hx|s p|status post|prior|previous|past|resolved|old)\b",
    "screen": r"\b(screen|screening|questionnaire|phq|gad|score|scale|assessment|survey)\b",
    "plan": r"\b(plan|goal|goals|recommend|recommended|advised|counsel|counseling|education|discussed|encourage)\b",
    "neg": r"\b(no|not|denies|denied|without|negative|absent|ruled out|r o|free of|none)\b",
    "risk": r"\b(risk|at risk|prevent|prevention|prophylaxis)\b",
}
DESC = {
    "family": r"\bfamily history\b|\bfh\b",
    "history": r"\b(history of|personal history)\b",
    "screen": r"\b(screening|assessment|scale|score)\b",
    "plan": r"\b(counseling|education|plan|advice|management|therapy|regime)\b",
    "neg": r"\b(no|absent|negative|without|not)\b",
    "risk": r"\b(risk|at risk|prophylaxis|prevention)\b",
}


def window(entity, chunk, width=8):
    e, words = clean(entity), clean(chunk).split()
    text = " ".join(words)
    pos = text.find(e) if e else -1
    if pos < 0:
        return text
    s = len(text[:pos].split())
    return " ".join(words[max(0, s - width): s + len(e.split()) + width])


def add_context(F: pd.DataFrame, rows: pd.DataFrame, info: dict) -> pd.DataFrame:
    """rows: query table indexed by qid with entity, chunk_text, coding_system."""
    F = F.copy()
    q = {}
    for qid, r in rows.iterrows():
        win = window(r.entity, r.chunk_text)
        ent = clean(r.entity)
        q[qid] = {k: (float(bool(re.search(p, win))), float(bool(re.search(p, ent)))) for k, p in CUES.items()}
    sys_of = rows.coding_system
    desc_cache = {}

    def desc_flags(system, code):
        key = (system, code)
        if key not in desc_cache:
            t = " ".join(info[system].descs.get(code, [])[:8])
            tag = info[system].tag.get(code, "")
            desc_cache[key] = {k: float(bool(re.search(p, t))) for k, p in DESC.items()} | {
                "situation": float(tag == "situation"), "procedure": float(tag in ("procedure", "regime/therapy")),
                "finding": float(tag == "finding"), "disorder": float(tag == "disorder")}
        return desc_cache[key]

    cols = {f"c_{k}_{x}": [] for k in CUES for x in ("win", "ent", "match", "conflict")}
    extra = {"c_situation_x_family": [], "c_procedure_x_plan": [], "c_finding_x_neg": [], "c_disorder_x_hist": []}
    for qid, code in zip(F.qid.values, F.code.values):
        cq, cd = q[qid], desc_flags(sys_of.at[qid], code)
        for k in CUES:
            w, e = cq[k]
            cols[f"c_{k}_win"].append(w)
            cols[f"c_{k}_ent"].append(e)
            cols[f"c_{k}_match"].append(max(w, e) * cd[k])
            cols[f"c_{k}_conflict"].append((1 - max(w, e)) * cd[k])
        extra["c_situation_x_family"].append(cq["family"][0] * cd["situation"])
        extra["c_procedure_x_plan"].append(cq["plan"][0] * cd["procedure"])
        extra["c_finding_x_neg"].append(cq["neg"][0] * cd["finding"])
        extra["c_disorder_x_hist"].append(cq["history"][0] * cd["disorder"])
    for k, v in {**cols, **extra}.items():
        F[k] = np.asarray(v, dtype=np.float32)
    return F
