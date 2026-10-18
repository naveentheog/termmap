# KAGGLE NOTEBOOK (GPU T4 x2) - fine-tune SapBERT as a RETRIEVER, v3 (maximum recall, long tail).
#
# What is new vs v2 and why:
#   1. Dictionary synonym pairs: every epoch also trains on ~40k pairs of two names of the SAME code taken
#      from the dictionaries (SapBERT's original recipe). Half of our gold codes are never seen in training,
#      so the retriever must learn ALL codes, not only the 4k that appear in our 9.6k rows.
#   2. Chart-noise augmentation on those synonym queries: typos, dropped spaces, truncated words and
#      acronyms ("chronic kidney disease" -> "ckd"), because real entities look like "chronickidney dis stag 3".
#   3. Hard negatives re-mined with the model itself after epochs 2 and 4 (ANCE).
#   4. 6 epochs, batch 160, SNOMED documents carry the semantic tag.
#   5. Explanation report per fold: recall@k overall, diagnosis rows, seen vs unseen (long-tail) codes,
#      plus the base SapBERT on fold 0 for the before/after story.
# Input : private dataset with bienc_input_v3.pkl
# Output: bienc3_scores.pkl (out-of-fold top-100 codes per row), bienc3_report.csv,
#         sapft3_full/ + emb_sapft3_<system>.npy + codes_sapft3_<system>.npy (for the final run)
import glob, os, pickle, subprocess, sys, time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.40"], check=False)
OUT = os.environ.get("TM_OUT", "/kaggle/working")
IN = (glob.glob("/kaggle/input/**/bienc_input_v3.pkl", recursive=True) or [os.environ.get("TM_IN", "bienc_input_v3.pkl")])[0]
print("[input]", IN, flush=True)

WORKER = r'''
import os, pickle, random, sys, time
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer
IN, OUT, JOBS = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
BASE = os.environ.get("TM_BASE", "cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
EPOCHS = int(os.environ.get("TM_EPOCHS", 6))
BATCH = int(os.environ.get("TM_BATCH", 160))
SYN_PER_EPOCH = int(os.environ.get("TM_SYN", 40000))
LR, MAX_LEN, TAU, TOPK, REMINE_AT = 2e-5, 40, 0.05, 100, (2, 4)
TRAIN_REPEAT, NOISE_P, ACRO_P = 2, 0.5, 0.08
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = dev.type == "cuda"
random.seed(0); np.random.seed(0); torch.manual_seed(0)

blob = pickle.load(open(IN, "rb")); dicts, rows = blob["dicts"], blob["rows"]
if os.environ.get("TM_SMOKE"):
    rows = [r for r in rows if r["qid"] % 40 == 0]
    for s in dicts:
        keep = set(r["gold"] for r in rows if r["system"] == s) | set(c for r in rows if r["system"] == s for c in r["negs"])
        idx = [i for i, c in enumerate(dicts[s]["codes"]) if c in keep or i % 50 == 0]
        dicts[s] = {k: [v[i] for i in idx] for k, v in dicts[s].items()}
descs_of, raw_of = {}, {}
for s, d in dicts.items():
    for c, t, r in zip(d["codes"], d["texts"], d["raw"]):
        descs_of.setdefault((s, c), []).append(t)
        raw_of.setdefault((s, c), []).append(r)
syn_codes = {s: [c for (ss, c), v in raw_of.items() if ss == s and len(v) >= 2] for s in dicts}
all_codes = {s: sorted(set(d["codes"])) for s, d in dicts.items()}
print(f"[data] rows {len(rows)} | synonym-pair codes: " + ", ".join(f"{s} {len(v)}" for s, v in syn_codes.items()), flush=True)
tok = AutoTokenizer.from_pretrained(BASE)

def noisy(t, rng):
    """Simulate chart text: typos, dropped spaces, truncation, acronyms."""
    w = t.split()
    if len(w) >= 2 and len(w) <= 5 and rng.random() < ACRO_P:
        return "".join(x[0] for x in w if x)
    if rng.random() > NOISE_P or len(t) < 6:
        return t
    t = list(t)
    for _ in range(rng.randint(1, 2)):
        op, i = rng.random(), rng.randrange(1, len(t) - 1)
        if op < 0.25:
            del t[i]
        elif op < 0.5:
            t[i], t[i + 1 if i + 1 < len(t) else i] = t[i + 1 if i + 1 < len(t) else i], t[i]
        elif op < 0.7:
            sp = [j for j, ch in enumerate(t) if ch == " "]
            if sp:
                del t[rng.choice(sp)]
        elif op < 0.85:
            t.insert(i, t[i])
        else:
            t = t[:max(4, len(t) - rng.randint(1, 4))]
    return "".join(t).strip()

def enc(model, texts, bs=1024):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
            v = model(**b).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(v.float(), dim=-1).half())
    return torch.cat(out)

def search(model, q_rows, topk=TOPK):
    out = {}
    model.eval()
    for s, d in dicts.items():
        rs = [r for r in q_rows if r["system"] == s]
        if not rs:
            continue
        D = enc(model, d["texts"]); codes = np.asarray(d["codes"])
        Q = enc(model, [r["entity"] for r in rs])
        for i in range(0, len(rs), 256):
            q = Q[i:i + 256].float()
            sims = torch.cat([q @ D[j:j + 200000].float().T for j in range(0, D.shape[0], 200000)], 1)
            k = min(topk * 4, sims.shape[1])
            val, idx = sims.topk(k, dim=1)
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
        del D
        if AMP:
            torch.cuda.empty_cache()
    return out

def mine(model, q_rows, k=30):
    res = search(model, q_rows, topk=k + 5)
    model.train()
    return {r["qid"]: [c for c in res.get(r["qid"], {}) if c != r["gold"]][:k] for r in q_rows}

def epoch_items(rows_ok, rng):
    items = []
    for _ in range(TRAIN_REPEAT):
        for r in rows_ok:
            key = (r["system"], r["gold"])
            nc = rng.choice(r["negs"][:20]) if r["negs"] else rng.choice(all_codes[r["system"]])
            items.append((r["entity"], rng.choice(descs_of[key]), key,
                          rng.choice(descs_of.get((r["system"], nc), descs_of[key])), (r["system"], nc)))
    sys_w = [len(syn_codes[s]) for s in dicts]
    for _ in range(SYN_PER_EPOCH):
        s = rng.choices(list(dicts), weights=sys_w)[0]
        if not syn_codes[s]:
            continue
        c = rng.choice(syn_codes[s]); key = (s, c)
        i, j = rng.sample(range(len(raw_of[key])), 2)
        nc = rng.choice(all_codes[s])
        items.append((noisy(raw_of[key][i], rng), descs_of[key][j], key,
                      rng.choice(descs_of[(s, nc)]), (s, nc)))
    rng.shuffle(items)
    return items

def train(tr_rows):
    model = AutoModel.from_pretrained(BASE).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    rows_ok = [dict(r) for r in tr_rows if (r["system"], r["gold"]) in descs_of]
    n_items = TRAIN_REPEAT * len(rows_ok) + SYN_PER_EPOCH
    steps = EPOCHS * (n_items // BATCH + 1)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, 1 - s / steps))
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    rng = random.Random(0)
    model.train()
    for ep in range(EPOCHS):
        if ep in REMINE_AT:
            t1 = time.time()
            mined = mine(model, rows_ok)
            for r in rows_ok:
                m = mined.get(r["qid"], [])
                r["negs"] = m[:15] + [c for c in r["negs"] if c not in m][:15]
            print(f"   re-mined hard negatives ({time.time() - t1:.0f}s)", flush=True)
        items = epoch_items(rows_ok, rng)
        t0, tot, n = time.time(), 0.0, 0
        for i in range(0, len(items), BATCH):
            b = items[i:i + BATCH]
            q = [x[0] for x in b]
            docs = [x[1] for x in b] + [x[3] for x in b]
            dkeys = [x[2] for x in b] + [x[4] for x in b]
            tq = tok(q, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
            td = tok(docs, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                eq = torch.nn.functional.normalize(model(**tq).last_hidden_state[:, 0].float(), dim=-1)
                ed = torch.nn.functional.normalize(model(**td).last_hidden_state[:, 0].float(), dim=-1)
            logits = eq @ ed.T / TAU
            qk = [x[2] for x in b]
            mask = torch.tensor([[dj != qi and dk == qk[qi] for dj, dk in enumerate(dkeys)] for qi in range(len(b))],
                                device=dev)
            logits = logits.masked_fill(mask, -1e4)
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(b), device=dev))
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            tot += loss.item(); n += 1
        print(f"   epoch {ep + 1}: loss {tot / max(n, 1):.4f} ({time.time() - t0:.0f}s, {len(items)} pairs)", flush=True)
    model.eval()
    return model

def report(res, te, seen_codes, tag):
    lines = []
    groups = {"all": te, "diagnosis": [r for r in te if r.get("sub") == "DiagnosisCode"],
              "seen": [r for r in te if r["gold"] in seen_codes], "unseen": [r for r in te if r["gold"] not in seen_codes]}
    for s in ("ICD10CM", "SNOMEDCT_US"):
        for g, rs in groups.items():
            rs = [r for r in rs if r["system"] == s]
            if not rs:
                continue
            rec = {k: float(np.mean([r["gold"] in list(res.get(r["qid"], {}))[:k] for r in rs])) for k in (1, 10, 40, 100)}
            lines.append((tag, s, g, len(rs), rec[1], rec[10], rec[40], rec[100]))
            print(f"[{tag}] {s:<12} {g:<9} n={len(rs):<5} R@1={rec[1]:.3f} R@10={rec[10]:.3f} R@40={rec[40]:.3f} R@100={rec[100]:.3f}", flush=True)
    return lines

for job in JOBS:
    t0 = time.time()
    if job == "full":
        print("[full] training on all rows", flush=True)
        m = train(rows)
        m.save_pretrained(f"{OUT}/sapft3_full"); tok.save_pretrained(f"{OUT}/sapft3_full")
        for s, d in dicts.items():
            np.save(f"{OUT}/emb_sapft3_{s}.npy", enc(m, d["texts"]).cpu().numpy())
            np.save(f"{OUT}/codes_sapft3_{s}.npy", np.asarray(d["codes"]))
        print(f"[full] saved model + dictionary embeddings ({time.time() - t0:.0f}s)", flush=True)
        continue
    f = int(job)
    tr = [r for r in rows if r["fold"] != f]; te = [r for r in rows if r["fold"] == f]
    seen_codes = set(r["gold"] for r in tr)
    print(f"[fold {f}] train {len(tr)} search {len(te)}", flush=True)
    lines = []
    if f == 0:
        base = AutoModel.from_pretrained(BASE).to(dev).eval()
        lines += report(search(base, te), te, seen_codes, "fold 0 BASE")
        del base
    m = train(tr)
    res = search(m, te)
    pickle.dump(res, open(f"{OUT}/bienc3_fold{f}.pkl", "wb"))
    lines += report(res, te, seen_codes, f"fold {f}")
    pickle.dump(lines, open(f"{OUT}/bienc3_report_fold{f}.pkl", "wb"))
    print(f"[fold {f}] done ({time.time() - t0:.0f}s)", flush=True)
    del m
    if AMP:
        torch.cuda.empty_cache()
print("[worker done]", flush=True)
'''
open(f"{OUT}/bienc3_worker.py", "w").write(WORKER)

import torch
n_gpu = max(1, torch.cuda.device_count())
plan = {"0": "0,1,2", "1": "3,4,full"} if n_gpu >= 2 else {"0": "0,1,2,3,4,full"}
if os.environ.get("TM_SMOKE"):
    plan = {"0": os.environ.get("TM_JOBS", "0")}
procs = {}
for gpu, jobs in plan.items():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "TOKENIZERS_PARALLELISM": "false"}
    procs[gpu] = subprocess.Popen([sys.executable, f"{OUT}/bienc3_worker.py", IN, OUT, jobs], env=env,
                                  stdout=open(f"{OUT}/bienc3_gpu{gpu}.log", "w"), stderr=subprocess.STDOUT)
    print(f"[launch] GPU {gpu}: jobs {jobs}", flush=True)

seen = {g: 0 for g in plan}
keys = ("[fold", "epoch", "[full]", "[data]", "re-mined", "Error", "error", "Traceback", "done")
while any(p.poll() is None for p in procs.values()):
    time.sleep(int(os.environ.get("TM_POLL", 60)))
    for g in plan:
        lines = open(f"{OUT}/bienc3_gpu{g}.log").read().splitlines()
        for ln in lines[seen[g]:]:
            if any(k in ln for k in keys):
                print(f"GPU{g} | {ln}", flush=True)
        seen[g] = len(lines)
for g in plan:
    lines = open(f"{OUT}/bienc3_gpu{g}.log").read().splitlines()
    for ln in lines[seen[g]:]:
        if any(k in ln for k in keys):
            print(f"GPU{g} | {ln}", flush=True)
    print(f"GPU{g} exit code {procs[g].returncode}")
    if procs[g].returncode != 0:
        print("\n".join(lines[-25:]))

import numpy as np, pandas as pd
oof, rep = {}, []
for f in range(5):
    if os.path.exists(f"{OUT}/bienc3_fold{f}.pkl"):
        oof.update(pickle.load(open(f"{OUT}/bienc3_fold{f}.pkl", "rb")))
    if os.path.exists(f"{OUT}/bienc3_report_fold{f}.pkl"):
        rep += pickle.load(open(f"{OUT}/bienc3_report_fold{f}.pkl", "rb"))
pickle.dump(oof, open(f"{OUT}/bienc3_scores.pkl", "wb"))
if rep:
    df = pd.DataFrame(rep, columns=["run", "system", "group", "n", "R@1", "R@10", "R@40", "R@100"])
    df.to_csv(f"{OUT}/bienc3_report.csv", index=False)
    folds = df[~df.run.str.contains("BASE")]
    summ = folds.groupby(["system", "group"]).apply(lambda g: pd.Series({c: np.average(g[c], weights=g.n) for c in ("R@1", "R@10", "R@40", "R@100")}))
    print("\n===== OUT-OF-FOLD RECALL (all folds) =====")
    print(summ.round(3).to_string())
print("[done] download bienc3_scores.pkl and bienc3_report.csv (the emb_sapft3_*.npy files are for the final run)")
