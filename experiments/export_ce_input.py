"""Export the top-K hybrid candidates (with readable descriptions) for the cross-encoder.

python export_ce_input.py [--k 20]  -> cache/ce_input.pkl  (send to the laptop with run_crossencoder.py)
Contains your own training rows + public dictionary descriptions only.
"""
import argparse
import pickle
import re

from termmap.data import dictionaries, SYSTEMS, SNO
from termmap.text import clean


def context_window(entity, chunk, width=8):
    """±width words around the entity inside its chunk (first 2*width words if not found)."""
    e, words = clean(entity), clean(chunk).split()
    text = " ".join(words)
    pos = text.find(e) if e else -1
    if pos < 0:
        return " ".join(words[: 2 * width])
    start_w = len(text[:pos].split())
    end_w = start_w + len(e.split())
    return " ".join(words[max(0, start_w - width): end_w + width])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    a = ap.parse_args()
    with open("cache/cands_hybrid.pkl", "rb") as fh:
        blob = pickle.load(fh)
    cands, train = blob["cands"], blob["train"]
    D = dictionaries("data/icd.xlsx", "data/snomed.xlsx")
    names = {}
    for s in SYSTEMS:
        d = D[s].sort_values("order") if s == SNO else D[s]
        names[s] = d.groupby("code").desc.agg(lambda x: " ; ".join(list(dict.fromkeys(x))[:3])).to_dict()
    rows, desc = [], {}
    for qid, r in train.iterrows():
        codes = list(cands[qid].code.values[: a.k])
        for c in codes:
            desc[(r.coding_system, c)] = names[r.coding_system].get(c, c)
        rows.append({"qid": qid, "entity": r.entity, "context": context_window(r.entity, r.chunk_text),
                     "system": r.coding_system, "subconcept": r.subconcept, "codes": codes})
    with open("cache/ce_input.pkl", "wb") as fh:
        pickle.dump({"rows": rows, "desc": desc, "k": a.k}, fh)
    print(f"[export] {len(rows)} queries, {len(desc)} unique (system, code) descriptions -> cache/ce_input.pkl")


if __name__ == "__main__":
    main()
