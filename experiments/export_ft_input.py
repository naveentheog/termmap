"""Export what the Kaggle fine-tuning notebook needs (kaggle/finetune_ce.py).

python export_ft_input.py [--k 30]  -> cache/ft_input.pkl

ONLY upload this to a PRIVATE Kaggle dataset after the organizers confirm it is allowed:
it contains training entities, a short context window and the gold codes.
"""
import argparse
import pickle

from export_ce_input import context_window
from termmap.data import dictionaries, SYSTEMS, SNO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--tag", action="store_true", help="v2: add SNOMED semantic tag / ICD code to the description")
    ap.add_argument("--out", default="cache/ft_input.pkl")
    ap.add_argument("--cands", default="cache/cands_hybrid.pkl")
    ap.add_argument("--width", type=int, default=8, help="context words on each side of the entity")
    a = ap.parse_args()
    with open(a.cands, "rb") as fh:
        blob = pickle.load(fh)
    cands, train = blob["cands"], blob["train"]
    D = dictionaries("data/icd.xlsx", "data/snomed.xlsx")
    names = {}
    for s in SYSTEMS:
        d = D[s].sort_values("order") if s == SNO else D[s]
        names[s] = d.groupby("code").desc.agg(lambda x: " ; ".join(list(dict.fromkeys(x))[:3])).to_dict()
        if a.tag and s == SNO:
            tag = d.drop_duplicates("code").set_index("code").tag.to_dict()
            names[s] = {c: f"{n} ({tag[c]})" if isinstance(tag.get(c), str) and tag.get(c) else n for c, n in names[s].items()}
        elif a.tag:
            names[s] = {c: f"{c} {n}" for c, n in names[s].items()}
    rows, desc = [], {}
    for qid, r in train.iterrows():
        codes = list(cands[qid].code.values[: a.k])
        for c in codes + [r.code]:
            desc[(r.coding_system, c)] = names[r.coding_system].get(c, c)
        rows.append({"qid": qid, "entity": r.entity, "context": context_window(r.entity, r.chunk_text, a.width),
                     "system": r.coding_system, "codes": codes, "gold": r.code, "fold": int(r.fold)})
    with open(a.out, "wb") as fh:
        pickle.dump({"rows": rows, "desc": desc, "k": a.k}, fh)
    hit = sum(r["gold"] in r["codes"] for r in rows) / len(rows)
    print(f"[export] {len(rows)} queries, gold in top-{a.k}: {hit:.3f}, {len(desc)} descriptions -> {a.out}")


if __name__ == "__main__":
    main()
