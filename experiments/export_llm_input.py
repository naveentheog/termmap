"""Export prompts' raw material for the LLM adjudicator (llm/llm_rerank.py).

For every training row (CV, out-of-fold) we store:
  - entity, ±25-word context, coding system
  - the ranker's top-10 candidates (from an out-of-fold run with saved top-10) with readable descriptions
  - 6 most similar TRAINING examples from OTHER folds (SapBERT entity similarity, same system) with their
    gold code + description -> in-context examples of how these coders resolve ambiguity
python export_llm_input.py cache/oof_fast_<...>.pkl  -> cache/llm_input.pkl
"""
import pickle
import sys

import numpy as np
import pandas as pd

from export_ce_input import context_window
from termmap.crossenc import build_names
from termmap.data import dictionaries, SYSTEMS

K_SHOT = 8


def main(oof_path, out="cache/llm_input.pkl"):
    oof = pd.read_pickle(oof_path)
    assert "top10" in oof, "run run_fast.py again so it saves top10"
    D = dictionaries("data/icd.xlsx", "data/snomed.xlsx")
    names = build_names(D)
    names["ICD10CM"] = {c: (d.split(" ", 1)[1] if d.startswith(c + " ") else d) for c, d in names["ICD10CM"].items()}
    emb = np.load("cache/entity_emb_sapbert.npy").astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    rows = []
    for s in SYSTEMS:
        idx = oof.index[oof.coding_system == s].values
        sims = emb[idx] @ emb[idx].T
        folds = oof.fold.values[idx]
        for a, qid in enumerate(idx):
            r = oof.loc[qid]
            sim = sims[a].copy()
            sim[folds == folds[a]] = -9          # examples only from other folds (no leakage)
            nn = np.argsort(-sim)[:30]
            shots = [(oof.entity.iat[oof.index.get_loc(idx[j])], oof.code.iat[oof.index.get_loc(idx[j])])
                     for j in nn[:K_SHOT]]
            nbrs = [(oof.entity.iat[oof.index.get_loc(idx[j])], oof.code.iat[oof.index.get_loc(idx[j])],
                     int(folds[j])) for j in nn]
            rows.append({
                "qid": int(qid), "system": s, "entity": r.entity, "fold": int(r.fold),
                "context": context_window(r.entity, r.chunk_text, 25),
                "options": [c for c, _ in r.top10], "ranker_scores": [float(v) for _, v in r.top10],
                "shots": [(e, c, names[s].get(c, c)) for e, c in shots],
                "nbrs": [(e, c, names[s].get(c, c), fo) for e, c, fo in nbrs],
                "gold": r.code, "sub": r.subconcept,
            })
    desc = {(r["system"], c): names[r["system"]].get(c, c) for r in rows for c in r["options"]}
    icd_names = names["ICD10CM"]
    for (s, c), d in list(desc.items()):
        if s == "ICD10CM" and len(c) > 3 and c[:3] in icd_names:
            desc[(s, c)] = f"{d}  [category {c[:3]}: {icd_names[c[:3]].split(' ; ')[0]}]"
    pickle.dump({"rows": rows, "desc": desc}, open(out, "wb"))
    hit = np.mean([r["gold"] in r["options"] for r in rows])
    print(f"[export] {len(rows)} rows, gold in ranker top-10: {hit:.3f} -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
