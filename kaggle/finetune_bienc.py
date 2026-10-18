# KAGGLE NOTEBOOK (GPU T4 x2) - fine-tune SapBERT as a RETRIEVER on our training pairs (v2).
# v2: SNOMED descriptions carry the semantic tag, batch 128 (more in-batch negatives), 4 epochs,
#     and after epoch 2 the hard negatives are RE-MINED with the model itself (ANCE-style),
#     plus a base-SapBERT baseline on fold 0 so we can see the gain.
# Goal: fewer retrieval misses (gold code not in the shortlist: 11% ICD / 15.5% SNOMED today).
# Input : private dataset with bienc_input.pkl
# Output: bienc_scores.pkl (out-of-fold top-100 codes per row) + recall report
#         sapft_full/ (model trained on all rows) + emb_sapft_<system>.npy, codes_sapft_<system>.npy
import glob, os, pickle, subprocess, sys, time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.40"], check=False)
OUT = "/kaggle/working"
IN = (glob.glob("/kaggle/input/**/bienc_input.pkl", recursive=True) or ["bienc_input.pkl"])[0]
print("[input]", IN, flush=True)

WORKER = r'''
import os, pickle, random, sys, time
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer
IN, OUT, JOBS = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
BASE = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
EPOCHS, BATCH, LR, MAX_LEN, TAU, TOPK, REMINE_AFTER = 4, 128, 2e-5, 40, 0.05, 100, 2
dev = torch.device("cuda")
random.seed(0); np.random.seed(0); torch.manual_seed(0)

blob = pickle.load(open(IN, "rb")); dicts, rows = blob["dicts"], blob["rows"]
descs_of = {}
for s, d in dicts.items():
    for c, t in zip(d["codes"], d["texts"]):
        descs_of.setdefault((s, c), []).append(t)
tok = AutoTokenizer.from_pretrained(BASE)

def enc(model, texts, bs=1024):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            v = model(**b).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(v.float(), dim=-1).half())
    return torch.cat(out)

def mine(model, q_rows, k=30):
    """Hard negatives from the CURRENT model: top dictionary codes that are not the gold code."""
    model.eval()
    res = search(model, q_rows, topk=k + 5)
    model.train()
    out = {}
    for r in q_rows:
        out[r["qid"]] = [c for c in res.get(r["qid"], {}) if c != r["gold"]][:k]
    return out

def train(tr_rows):
    model = AutoModel.from_pretrained(BASE).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    rows_ok = [dict(r) for r in tr_rows if (r["system"], r["gold"]) in descs_of]
    steps = EPOCHS * (len(rows_ok) // BATCH + 1)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, 1 - s / steps))
    scaler = torch.amp.GradScaler("cuda")
    rng = random.Random(0)
    model.train()
    for ep in range(EPOCHS):
        if ep == REMINE_AFTER:
            t1 = time.time()
            mined = mine(model, rows_ok)
            for r in rows_ok:
                m = mined.get(r["qid"], [])
                r["negs"] = m[:15] + [c for c in r["negs"] if c not in m][:15]
            print(f"   re-mined hard negatives with the model ({time.time() - t1:.0f}s)", flush=True)
        rng.shuffle(rows_ok); t0, tot, n = time.time(), 0.0, 0
        for i in range(0, len(rows_ok), BATCH):
            b = rows_ok[i:i + BATCH]
            q = [r["entity"] for r in b]
            pos = [rng.choice(descs_of[(r["system"], r["gold"])]) for r in b]
            neg_codes = [rng.choice(r["negs"][:20]) if r["negs"] else r["gold"] for r in b]
            neg = [rng.choice(descs_of.get((r["system"], c), [r["entity"]])) for r, c in zip(b, neg_codes)]
            docs = pos + neg
            doc_codes = [(r["system"], r["gold"]) for r in b] + [(r["system"], c) for r, c in zip(b, neg_codes)]
            tq = tok(q, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
            td = tok(docs, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                eq = torch.nn.functional.normalize(model(**tq).last_hidden_state[:, 0].float(), dim=-1)
                ed = torch.nn.functional.normalize(model(**td).last_hidden_state[:, 0].float(), dim=-1)
            logits = eq @ ed.T / TAU
            # mask false negatives: another doc in the batch that is ALSO this query's gold code
            mask = torch.zeros_like(logits, dtype=torch.bool)
            for qi, r in enumerate(b):
                for dj, dc in enumerate(doc_codes):
                    if dj != qi and dc == (r["system"], r["gold"]):
                        mask[qi, dj] = True
            logits = logits.masked_fill(mask, -1e4)
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(b), device=dev))
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            tot += loss.item(); n += 1
        print(f"   epoch {ep + 1}: loss {tot / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    return model

def search(model, q_rows, topk=TOPK):
    out = {}
    for s, d in dicts.items():
        rs = [r for r in q_rows if r["system"] == s]
        if not rs:
            continue
        D = enc(model, d["texts"]); codes = np.asarray(d["codes"])
        Q = enc(model, [r["entity"] for r in rs])
        for i in range(0, len(rs), 256):
            sims = Q[i:i + 256].float() @ D.float().T if D.shape[0] < 200000 else torch.cat(
                [Q[i:i + 256].float() @ D[j:j + 200000].float().T for j in range(0, D.shape[0], 200000)], 1)
            val, idx = sims.topk(topk * 4, dim=1)
            val, idx = val.cpu().numpy(), idx.cpu().numpy()
            for r, vs, ix in zip(rs[i:i + 256], val, idx):
                agg = {}
                for v, j in zip(vs, ix):
                    c = codes[j]
                    if c not in agg:
                        agg[c] = float(v)
                    if len(agg) >= topk:
                        break
                out[r["qid"]] = agg
        del D; torch.cuda.empty_cache()
    return out

for job in JOBS:
    t0 = time.time()
    if job == "full":
        print("[full] training on all rows", flush=True)
        m = train(rows)
        m.save_pretrained(f"{OUT}/sapft_full"); tok.save_pretrained(f"{OUT}/sapft_full")
        for s, d in dicts.items():
            np.save(f"{OUT}/emb_sapft_{s}.npy", enc(m, d["texts"]).cpu().numpy())
            np.save(f"{OUT}/codes_sapft_{s}.npy", np.asarray(d["codes"]))
        print(f"[full] saved model + dictionary embeddings ({time.time() - t0:.0f}s)", flush=True)
        continue
    f = int(job)
    tr = [r for r in rows if r["fold"] != f]; te = [r for r in rows if r["fold"] == f]
    print(f"[fold {f}] train {len(tr)} search {len(te)}", flush=True)
    if f == 0:
        base = AutoModel.from_pretrained(BASE).to(dev).eval()
        bres = search(base, te)
        for k in (10, 50, 100):
            print(f"[fold 0] BASE SapBERT recall@{k} = {np.mean([r['gold'] in list(bres.get(r['qid'], {}))[:k] for r in te]):.4f}", flush=True)
        del base; torch.cuda.empty_cache()
    m = train(tr)
    res = search(m, te)
    pickle.dump(res, open(f"{OUT}/bienc_fold{f}.pkl", "wb"))
    for k in (1, 10, 50, 100):
        rec = np.mean([r["gold"] in list(res.get(r["qid"], {}))[:k] for r in te])
        print(f"[fold {f}] recall@{k} = {rec:.4f}", flush=True)
    print(f"[fold {f}] done ({time.time() - t0:.0f}s)", flush=True)
    del m; torch.cuda.empty_cache()
print("[worker done]", flush=True)
'''
open(f"{OUT}/bienc_worker.py", "w").write(WORKER)

plan = {"0": "0,1,2", "1": "3,4,full"}
procs = {}
for gpu, jobs in plan.items():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "TOKENIZERS_PARALLELISM": "false"}
    procs[gpu] = subprocess.Popen([sys.executable, f"{OUT}/bienc_worker.py", IN, OUT, jobs], env=env,
                                  stdout=open(f"{OUT}/bienc_gpu{gpu}.log", "w"), stderr=subprocess.STDOUT)
    print(f"[launch] GPU {gpu}: jobs {jobs}", flush=True)

seen = {g: 0 for g in plan}
while any(p.poll() is None for p in procs.values()):
    time.sleep(60)
    for g in plan:
        lines = open(f"{OUT}/bienc_gpu{g}.log").read().splitlines()
        for ln in lines[seen[g]:]:
            if any(k in ln for k in ("[fold", "epoch", "[full]", "re-mined", "Error", "error", "Traceback", "done")):
                print(f"GPU{g} | {ln}", flush=True)
        seen[g] = len(lines)
for g in plan:
    print(f"GPU{g} exit code {procs[g].returncode}")
    if procs[g].returncode != 0:
        print("\n".join(open(f"{OUT}/bienc_gpu{g}.log").read().splitlines()[-25:]))

import numpy as np
oof = {}
for f in range(5):
    p = f"{OUT}/bienc_fold{f}.pkl"
    if os.path.exists(p):
        oof.update(pickle.load(open(p, "rb")))
pickle.dump(oof, open(f"{OUT}/bienc_scores.pkl", "wb"))
rows = pickle.load(open(IN, "rb"))["rows"]
for s in ("ICD10CM", "SNOMEDCT_US"):
    rs = [r for r in rows if r["system"] == s and r["qid"] in oof]
    if rs:
        print(s, {k: round(float(np.mean([r["gold"] in list(oof[r["qid"]])[:k] for r in rs])), 4) for k in (1, 10, 50, 100)})
print("[done] download bienc_scores.pkl (small). The emb_sapft_*.npy files (~1.2 GB) are for the final run later.")
