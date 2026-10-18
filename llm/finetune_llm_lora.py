"""Fine-tuned LLM reranker (LoRA) for TermMap — run on a rented H100/H200.

The zero-shot LLM (llm_rerank.py) knows medicine but not OUR coders' conventions. Here we teach it:
a Qwen2.5 Instruct model is LoRA-fine-tuned to pick the gold letter among the ranker's top-10 options,
with the same prompt (term, context, similar coded examples, options).

Honest evaluation: 5 folds (same chunk-grouped folds). For fold f the model trains on the other folds
only, and the in-context examples of its training prompts never come from fold f (no leakage).
A 6th model trained on everything is saved for the assessment set.

Install:  pip install -q peft accelerate
Run:      python finetune_llm_lora.py --inp llm_input.pkl --model Qwen/Qwen2.5-7B-Instruct
Smoke:    python finetune_llm_lora.py --inp llm_input.pkl --smoke
Output:   llm_ft_scores.pkl {qid: {code: prob}} (out-of-fold) + lora_full/ adapter
"""
import argparse
import math
import os
import pickle
import random
import time

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"

from llm_rerank import LETTERS, SYSTEM_MSG, build_prompt


def shots_without(r, banned_fold, k=8):
    out = [(e, c, d) for e, c, d, fo in r["nbrs"] if fo != banned_fold][:k]
    return out


def make_prompt(tok, r, desc, opts, banned_fold=None):
    rr = dict(r)
    if banned_fold is not None:
        rr["shots"] = shots_without(r, banned_fold)
    msgs = [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": build_prompt(rr, desc, opts)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def letter_logits(model, tok, prompts, letter_ids, max_len, grad=False):
    """Only the LAST position's logits over the 10 letters are needed: run the transformer body and apply
    the LM head to one position (saves the full-vocabulary logits -> much less GPU memory, fits 24 GB)."""
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(model.device)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    with torch.set_grad_enabled(grad), torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
        h = base.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
        last = h[:, -1] if tok.padding_side == "left" else h[torch.arange(len(prompts), device=h.device),
                                                              enc["attention_mask"].sum(1) - 1]
        w = base.lm_head.weight[letter_ids]
        logits = last @ w.T
    return logits.float()


def train_fold(base, tok, rows, desc, letter_ids, args, banned_fold):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16 if DEV == "cuda" else torch.float32, device_map=DEV,
                                                 attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    cfg = LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    rng = random.Random(0)
    ex = [r for r in rows if r["gold"] in r["options"]]
    steps = args.epochs * math.ceil(len(ex) / args.bs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.05 * steps)) * max(0.0, 1 - s / steps))
    model.train()
    step = 0
    for ep in range(args.epochs):
        rng.shuffle(ex)
        t0, tot = time.time(), 0.0
        for i in range(0, len(ex), args.bs):
            batch = ex[i:i + args.bs]
            prompts, gold = [], []
            for r in batch:
                opts = r["options"][:]
                rng.shuffle(opts)                       # no position shortcut
                prompts.append(make_prompt(tok, r, desc, opts, banned_fold))
                gold.append(opts.index(r["gold"]))
            lg = letter_logits(model, tok, prompts, letter_ids, args.max_len, grad=True)
            mask = torch.full_like(lg, float("-inf"))
            for j, r in enumerate(batch):
                mask[j, :len(r["options"])] = 0
            loss = torch.nn.functional.cross_entropy(lg + mask, torch.tensor(gold, device=lg.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
            tot += loss.item(); step += 1
            if step % 50 == 0:
                print(f"   step {step}/{steps} loss {tot / ((i // args.bs) + 1):.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    return model


@torch.no_grad()
def score(model, tok, rows, desc, letter_ids, args, banned_fold=None):
    out = {}
    for i in range(0, len(rows), args.ebs):
        batch = rows[i:i + args.ebs]
        acc = [dict() for _ in batch]
        for order in (lambda o: o, lambda o: o[::-1]):
            opts_list = [order(r["options"]) for r in batch]
            prompts = [make_prompt(tok, r, desc, o, banned_fold) for r, o in zip(batch, opts_list)]
            lg = letter_logits(model, tok, prompts, letter_ids, args.max_len)
            for j, (r, o) in enumerate(zip(batch, opts_list)):
                p = torch.softmax(lg[j, :len(o)], 0).cpu().numpy()
                for c, v in zip(o, p):
                    acc[j][c] = acc[j].get(c, 0.0) + float(v) / 2
        for r, a in zip(batch, acc):
            out[r["qid"]] = a
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="llm_input.pkl")
    ap.add_argument("--out", default="llm_ft_scores.pkl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--ebs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--folds", default="0,1,2,3,4,full")
    ap.add_argument("--smoke", action="store_true", help="tiny run: 200 train rows, 100 scored rows, fold 0 only")
    args = ap.parse_args()
    torch.manual_seed(0); random.seed(0); np.random.seed(0)

    from transformers import AutoTokenizer
    blob = pickle.load(open(args.inp, "rb"))
    rows, desc = blob["rows"], blob["desc"]
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    letter_ids = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]
    oof = pickle.load(open(args.out, "rb")) if os.path.exists(args.out) else {}
    for job in args.folds.split(","):
        t0 = time.time()
        if job == "full":
            model = train_fold(args.model, tok, rows, desc, letter_ids, args, None)
            model.save_pretrained("lora_full"); tok.save_pretrained("lora_full")
            print(f"[full] saved lora_full ({time.time() - t0:.0f}s)", flush=True)
            continue
        f = int(job)
        tr = [r for r in rows if r["fold"] != f]
        te = [r for r in rows if r["fold"] == f]
        if args.smoke:
            tr, te = tr[::max(1, len(tr) // 200)][:200], te[::max(1, len(te) // 100)][:100]
        print(f"[fold {f}] train {len(tr)} score {len(te)}", flush=True)
        model = train_fold(args.model, tok, tr, desc, letter_ids, args, f)
        s = score(model, tok, te, desc, letter_ids, args, f)
        oof.update(s)
        pickle.dump(oof, open(args.out, "wb"))
        for sysn in ("ICD10CM", "SNOMEDCT_US"):
            rs = [r for r in te if r["system"] == sysn]
            if rs:
                llm = np.mean([max(s[r["qid"]], key=s[r["qid"]].get) == r["gold"] for r in rs])
                rk = np.mean([r["options"][0] == r["gold"] for r in rs])
                print(f"[fold {f}] {sysn:<12} n={len(rs):<5} fine-tuned LLM top-1={llm:.3f}  ranker top-1={rk:.3f}", flush=True)
        print(f"[fold {f}] done ({time.time() - t0:.0f}s)", flush=True)
        del model
        if DEV == "cuda":
            torch.cuda.empty_cache()
        if args.smoke:
            break


if __name__ == "__main__":
    main()
