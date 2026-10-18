"""Export what kaggle/finetune_bienc.py needs: the two public dictionaries (code, cleaned description)
and the training rows (entity, system, gold code, fold, hard negatives from our hybrid shortlist).

python export_bienc_input.py  -> cache/bienc_input.pkl   (upload to a PRIVATE Kaggle dataset)
"""
import pickle

from termmap.data import dictionaries, SYSTEMS
from termmap.dense import clean
from export_ce_input import context_window


def main():
    with open("cache/cands_hybrid.pkl", "rb") as fh:
        blob = pickle.load(fh)
    cands, train = blob["cands"], blob["train"]
    D = dictionaries("data/icd.xlsx", "data/snomed.xlsx")
    dicts = {}
    for s in SYSTEMS:
        d = D[s][["code", "desc"] + (["tag"] if "tag" in D[s] else [])].copy()
        d["text"] = d.desc.map(clean)
        d["raw"] = d["text"]
        if "tag" in d:  # SNOMED: append the semantic tag so the retriever also learns the TYPE
            d["text"] = [f"{t} {tg}".strip() if isinstance(tg, str) and tg else t for t, tg in zip(d.text, d.tag)]
        d = d[d.text.str.len() > 0].drop_duplicates(["code", "text"])
        dicts[s] = {"codes": d.code.astype(str).tolist(), "texts": d.text.tolist(), "raw": d.raw.tolist()}
        print(f"[export] {s}: {len(d):,} descriptions, {d.code.nunique():,} codes")
    rows = []
    for qid, r in train.iterrows():
        negs = [c for c in cands[qid].code.values[:50] if c != r.code]
        rows.append({"qid": qid, "entity": clean(r.entity), "system": r.coding_system, "gold": str(r.code),
                     "fold": int(r.fold), "negs": [str(c) for c in negs], "sub": str(r.subconcept),
                     "context": context_window(r.entity, r.chunk_text, 12)})
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "cache/bienc_input.pkl"
    with open(out, "wb") as fh:
        pickle.dump({"dicts": dicts, "rows": rows}, fh)
    print(f"[export] {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
