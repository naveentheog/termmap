# KAGGLE NOTEBOOK (GPU T4 x2) - fine-tune v2: both GPUs in parallel, top-50 candidates,
# semantic tag in the description, 11 hard negatives, 3 epochs.
# Input : private dataset with ft_input_v2.pkl     Output: ft2_scores.pkl + ce2_full.zip
import glob, os, pickle, subprocess, sys, time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.40"], check=False)
OUT = "/kaggle/working"
IN = (glob.glob("/kaggle/input/**/ft_input_v2.pkl", recursive=True) or ["ft_input_v2.pkl"])[0]
print("[input]", IN, flush=True)

WORKER = r'''
import os, pickle, random, sys, time, shutil
import numpy as np, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
IN, OUT, JOBS = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
BASE = "ncbi/MedCPT-Cross-Encoder"
N_NEG, Q_PER_BATCH, EPOCHS, LR, MAX_LEN = 11, 8, 3, 2e-5, 160
dev = torch.device("cuda")

def seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def qtext(r):
    return f"{r['entity']} | {r['context']}"

def groups_for(rows, desc, rng):
    out = []
    for r in rows:
        negs = [c for c in r["codes"] if c != r["gold"]]
        if not negs:
            continue
        hard = negs[:7]
        rest = negs[7:]
        hard += rng.sample(rest, min(len(rest), N_NEG - len(hard)))
        while len(hard) < N_NEG:
            hard.append(rng.choice(negs))
        q = qtext(r)
        out.append([(q, desc[(r["system"], c)]) for c in [r["gold"]] + hard[:N_NEG]])
    return out

def train(rows, desc, tok):
    seed(0)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).to(dev)
    rng = random.Random(0)
    steps = EPOCHS * (len(rows) // Q_PER_BATCH + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    for ep in range(EPOCHS):
        gs = groups_for(rows, desc, rng); rng.shuffle(gs)
        t0, tot, n = time.time(), 0.0, 0
        for i in range(0, len(gs), Q_PER_BATCH):
            g = gs[i:i + Q_PER_BATCH]
            pairs = [p for grp in g for p in grp]
            enc = tok([a for a, _ in pairs], [b for _, b in pairs], truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**enc).logits.view(len(g), N_NEG + 1).float()
            loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(g), dtype=torch.long, device=dev))
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            tot += loss.item(); n += 1
        print(f"   epoch {ep + 1}: loss {tot / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    return model

@torch.no_grad()
def score(model, tok, rows, desc, bs=256):
    out, pairs, owners = {}, [], []
    for r in rows:
        for c in r["codes"]:
            pairs.append((qtext(r), desc[(r["system"], c)])); owners.append((r["qid"], c))
    for i in range(0, len(pairs), bs):
        p = pairs[i:i + bs]
        enc = tok([a for a, _ in p], [b for _, b in p], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            s = model(**enc).logits.view(-1).float().cpu().numpy()
        for (q, c), v in zip(owners[i:i + bs], s):
            out.setdefault(q, {})[c] = float(v)
    return out

blob = pickle.load(open(IN, "rb")); rows, desc = blob["rows"], blob["desc"]
tok = AutoTokenizer.from_pretrained(BASE)
for job in JOBS:
    t0 = time.time()
    if job == "full":
        print("[full] training on all rows", flush=True)
        m = train(rows, desc, tok)
        m.save_pretrained(f"{OUT}/ce2_full"); tok.save_pretrained(f"{OUT}/ce2_full")
        print(f"[full] saved ({time.time() - t0:.0f}s)", flush=True)
        continue
    f = int(job)
    tr = [r for r in rows if r["fold"] != f]; te = [r for r in rows if r["fold"] == f]
    print(f"[fold {f}] train {len(tr)} score {len(te)}", flush=True)
    m = train(tr, desc, tok)
    s = score(m, tok, te, desc)
    pickle.dump(s, open(f"{OUT}/ft2_fold{f}.pkl", "wb"))
    top1 = np.mean([max(s[r["qid"]], key=s[r["qid"]].get) == r["gold"] for r in te])
    print(f"[fold {f}] fine-tuned CE v2 alone top-1 = {top1:.4f} ({time.time() - t0:.0f}s)", flush=True)
    del m; torch.cuda.empty_cache()
print("[worker done]", flush=True)
'''
open(f"{OUT}/ft_worker.py", "w").write(WORKER)

plan = {"0": "0,1,2", "1": "3,4,full"}
procs, logs = {}, {}
for gpu, jobs in plan.items():
    logs[gpu] = open(f"{OUT}/worker_gpu{gpu}.log", "w")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "TOKENIZERS_PARALLELISM": "false"}
    procs[gpu] = subprocess.Popen([sys.executable, f"{OUT}/ft_worker.py", IN, OUT, jobs], env=env,
                                  stdout=logs[gpu], stderr=subprocess.STDOUT)
    print(f"[launch] GPU {gpu}: jobs {jobs}", flush=True)

seen = {g: 0 for g in plan}
while any(p.poll() is None for p in procs.values()):
    time.sleep(60)
    for g in plan:
        lines = open(f"{OUT}/worker_gpu{g}.log").read().splitlines()
        for ln in lines[seen[g]:]:
            if any(k in ln for k in ("[fold", "epoch", "[full]", "Error", "error", "Traceback", "done")):
                print(f"GPU{g} | {ln}", flush=True)
        seen[g] = len(lines)
for g in plan:
    print(f"GPU{g} exit code {procs[g].returncode}")
    if procs[g].returncode != 0:
        print("\n".join(open(f"{OUT}/worker_gpu{g}.log").read().splitlines()[-25:]))

import numpy as np, shutil
oof = {}
for f in range(5):
    p = f"{OUT}/ft2_fold{f}.pkl"
    if os.path.exists(p):
        oof.update(pickle.load(open(p, "rb")))
pickle.dump(oof, open(f"{OUT}/ft2_scores.pkl", "wb"))
rows = pickle.load(open(IN, "rb"))["rows"]
for s in ("ICD10CM", "SNOMEDCT_US"):
    rs = [r for r in rows if r["system"] == s and r["qid"] in oof]
    if rs:
        print(f"[oof] {s}: v2 alone top-1 = {np.mean([max(oof[r['qid']], key=oof[r['qid']].get) == r['gold'] for r in rs]):.4f}")
if os.path.isdir(f"{OUT}/ce2_full"):
    shutil.make_archive(f"{OUT}/ce2_full", "zip", f"{OUT}/ce2_full")
print("[done] download ft2_scores.pkl and ce2_full.zip from the Output tab")
