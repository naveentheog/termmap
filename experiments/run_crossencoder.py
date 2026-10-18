"""Run on YOUR LAPTOP (CPU). Scores the top-K candidates of every training row with a
biomedical cross-encoder, which reads the term and the code description together.

Model: ncbi/MedCPT-Cross-Encoder (PubMedBERT, trained by NCBI for biomedical reranking).
Zero-shot: no training on your data, so no leakage and nothing to tune.

    pip install sentence-transformers
    python run_crossencoder.py                 # ~40-90 min on a 6-core laptop CPU
    python run_crossencoder.py --k 10 --no-context   # faster smoke run (~10 min)

Input : cache/ce_input.pkl   (from export_ce_input.py)
Output: cache/ce_scores.pkl  {qid: {code: [score_entity, score_with_context]}}  -> upload to the chat
Progress is checkpointed every 1000 queries, so you can stop and resume.
"""
import argparse
import os
import pickle
import time

import numpy as np

MODEL = "ncbi/MedCPT-Cross-Encoder"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="cache/ce_input.pkl")
    ap.add_argument("--out", default="cache/ce_scores.pkl")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--no-context", action="store_true", help="only score the bare entity (2x faster)")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    import torch
    from sentence_transformers import CrossEncoder
    torch.manual_seed(0)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    ce = CrossEncoder(MODEL, max_length=96, device="cpu")

    with open(a.inp, "rb") as fh:
        blob = pickle.load(fh)
    rows, desc = blob["rows"], blob["desc"]
    done = {}
    if os.path.exists(a.out):
        with open(a.out, "rb") as fh:
            done = pickle.load(fh)
        print(f"[ce] resuming: {len(done)} queries already scored")

    t0, n_new = time.time(), 0
    todo = [r for r in rows if r["qid"] not in done]
    for i in range(0, len(todo), 50):
        block = todo[i:i + 50]
        pairs_e, pairs_c, owners = [], [], []
        for r in block:
            for c in r["codes"][: a.k]:
                d = desc[(r["system"], c)]
                pairs_e.append([r["entity"], d])
                pairs_c.append([f"{r['entity']} ({r['context']})", d])
                owners.append((r["qid"], c))
        se = ce.predict(pairs_e, batch_size=a.batch, show_progress_bar=False)
        sc = se if a.no_context else ce.predict(pairs_c, batch_size=a.batch, show_progress_bar=False)
        for (qid, c), x, y in zip(owners, se, sc):
            done.setdefault(qid, {})[c] = [float(x), float(y)]
        n_new += len(block)
        if n_new % 1000 < 50 or i + 50 >= len(todo):
            with open(a.out, "wb") as fh:
                pickle.dump(done, fh)
            rate = n_new / (time.time() - t0)
            left = (len(todo) - n_new) / max(rate, 1e-9) / 60
            print(f"[ce] {len(done)}/{len(rows)} queries  ({rate:.1f}/s, ~{left:.0f} min left)", flush=True)
    print(f"[ce] done -> {a.out}  (upload this file to the chat)")


if __name__ == "__main__":
    main()
