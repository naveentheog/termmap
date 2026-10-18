"""KAGGLE NOTEBOOK (GPU T4) — fine-tune a biomedical cross-encoder on OUR candidates.

Run ONLY if the organizers allow uploading training data to a private Kaggle notebook.

Why: the zero-shot MedCPT cross-encoder gets only 29% / 22% top-1, but the gold code is in our
top-5 most of the time. Training it on our own hard negatives teaches exactly the fine
distinctions we miss (laterality, acuity, 'history of', 'family history', procedure vs finding).

How to run:
 1. Kaggle -> New Dataset (PRIVATE) -> upload cache/ft_input.pkl. Name it "termmap-ft".
 2. New Notebook -> Add Input "termmap-ft". Settings: GPU T4 x2, Internet ON, Persistence off.
 3. Paste this file into one cell and run (~45-70 min).
 4. Output tab -> download ft_scores.pkl (out-of-fold scores, for the ranker) and
    ce_full.zip (model trained on all rows, for predict.py). Bring both back to the chat.

Leakage control: 5 models, each trained WITHOUT one fold and scoring only that fold
(same GroupKFold-by-chunk folds as the ranker). A 6th model trained on everything is
used only for the assessment set.
"""
import os
import pickle
import random
import shutil
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.40"], check=False)

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import glob
IN = (glob.glob("/kaggle/input/**/ft_input.pkl", recursive=True) or ["ft_input.pkl"])[0]
print("[input]", IN, "| GPU:", torch.cuda.is_available())
OUT = "/kaggle/working"
BASE = "ncbi/MedCPT-Cross-Encoder"
N_NEG, Q_PER_BATCH, EPOCHS, LR, MAX_LEN = 7, 8, 2, 2e-5, 128
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed(s=0):
    random.seed(s), np.random.seed(s), torch.manual_seed(s), torch.cuda.manual_seed_all(s)


def query_text(r):
    return f"{r['entity']} | {r['context']}"


def make_groups(rows, desc, rng):
    """One group per query: [gold, 7 hard negatives] (hardest = highest-ranked wrong candidates,
    plus a little randomness so the model does not only see the same 7)."""
    groups = []
    for r in rows:
        negs = [c for c in r["codes"] if c != r["gold"]]
        if not negs:
            continue
        hard = negs[:5] + rng.sample(negs[5:], min(len(negs) - 5, N_NEG - 5)) if len(negs) > 5 else negs
        while len(hard) < N_NEG:
            hard.append(rng.choice(negs))
        q = query_text(r)
        groups.append([(q, desc[(r["system"], c)]) for c in [r["gold"]] + hard[:N_NEG]])
    return groups


def train_model(rows, desc, tok):
    seed(0)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).to(dev)
    rng = random.Random(0)
    steps = EPOCHS * (len(rows) // Q_PER_BATCH + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    for ep in range(EPOCHS):
        groups = make_groups(rows, desc, rng)
        rng.shuffle(groups)
        t0, tot = time.time(), 0.0
        for i in range(0, len(groups), Q_PER_BATCH):
            g = groups[i:i + Q_PER_BATCH]
            pairs = [p for grp in g for p in grp]
            enc = tok([a for a, _ in pairs], [b for _, b in pairs], truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**enc).logits.view(len(g), N_NEG + 1).float()
            loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(g), dtype=torch.long, device=dev))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            tot += loss.item()
        print(f"   epoch {ep + 1}: loss {tot / (len(groups) / Q_PER_BATCH):.4f}  ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    return model


@torch.no_grad()
def score(model, tok, rows, desc, bs=256):
    out, pairs, owners = {}, [], []
    for r in rows:
        q = query_text(r)
        for c in r["codes"]:
            pairs.append((q, desc[(r["system"], c)]))
            owners.append((r["qid"], c))
    for i in range(0, len(pairs), bs):
        p = pairs[i:i + bs]
        enc = tok([a for a, _ in p], [b for _, b in p], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            s = model(**enc).logits.view(-1).float().cpu().numpy()
        for (qid, c), v in zip(owners[i:i + bs], s):
            out.setdefault(qid, {})[c] = float(v)
    return out


def main():
    with open(IN, "rb") as fh:
        blob = pickle.load(fh)
    rows, desc = blob["rows"], blob["desc"]
    tok = AutoTokenizer.from_pretrained(BASE)
    oof = {}
    for f in sorted({r["fold"] for r in rows}):
        t0 = time.time()
        tr = [r for r in rows if r["fold"] != f]
        te = [r for r in rows if r["fold"] == f]
        print(f"[fold {f}] train {len(tr)}  score {len(te)}", flush=True)
        model = train_model(tr, desc, tok)
        s = score(model, tok, te, desc)
        oof.update(s)
        top1 = np.mean([max(s[r["qid"]], key=s[r["qid"]].get) == r["gold"] for r in te if r["qid"] in s])
        print(f"[fold {f}] fine-tuned CE alone top-1 = {top1:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        with open(f"{OUT}/ft_scores.pkl", "wb") as fh:
            pickle.dump(oof, fh)
        del model
        torch.cuda.empty_cache()
    for sys_ in ("ICD10CM", "SNOMEDCT_US"):
        rs = [r for r in rows if r["system"] == sys_ and r["qid"] in oof]
        acc = np.mean([max(oof[r["qid"]], key=oof[r["qid"]].get) == r["gold"] for r in rs])
        print(f"[oof] {sys_}: fine-tuned CE alone top-1 = {acc:.4f}")
    print("[full] training on all rows for the assessment set", flush=True)
    model = train_model(rows, desc, tok)
    model.save_pretrained(f"{OUT}/ce_full")
    tok.save_pretrained(f"{OUT}/ce_full")
    shutil.make_archive(f"{OUT}/ce_full", "zip", f"{OUT}/ce_full")
    print("[done] download ft_scores.pkl and ce_full.zip from the Output tab")


if __name__ == "__main__":
    main()
