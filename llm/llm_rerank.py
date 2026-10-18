"""LLM adjudicator for TermMap — run on a rented GPU (H100/H200) or any GPU with enough memory.

What it does: for every term, a large open-weight instruction model sees
  - the term and ±25 words of chart context
  - 6 similar terms from OTHER training folds with the code our coders gave them (in-context convention)
  - the ranker's top-10 candidate codes with descriptions (SNOMED with semantic tag)
and picks one letter. We read the probability of every letter (deterministic: one forward pass,
no sampling), so each candidate gets an LLM probability -> used as an extra ranker feature.

Setup (once):   pip install -q vllm
Run:            python llm_rerank.py --inp llm_input.pkl --model Qwen/Qwen2.5-72B-Instruct --quant fp8
Smoke test:     python llm_rerank.py --inp llm_input.pkl --limit 200
Output:         llm_scores.pkl  {qid: {code: probability}}  + printed accuracy report
"""
import argparse
import math
import pickle
import time

LETTERS = "ABCDEFGHIJ"
SYSTEM_MSG = (
    "You are an expert clinical coder. Map the clinical term to exactly ONE code from the options. "
    "Follow the coding conventions shown in the examples from this organisation. Read the context: "
    "family history, personal history, negation, screening, plans, laterality, acuity and encounter type "
    "change the correct code. Choose an unspecified code only when the text supports nothing more specific."
)
SYS_NAME = {"ICD10CM": "ICD-10-CM", "SNOMEDCT_US": "SNOMED CT"}


def build_prompt(r, desc, options=None):
    options = options or r["options"]
    shots = "\n".join(f'- "{e}" -> {c} {d}' for e, c, d in r["shots"])
    opts = "\n".join(f"{LETTERS[i]}. {c} {desc[(r['system'], c)]}" for i, c in enumerate(options))
    return (f"Coding system: {SYS_NAME[r['system']]}\n\n"
            f"How our coders mapped similar terms:\n{shots}\n\n"
            f'Term: "{r["entity"]}"\n'
            f'Chart context: "{r["context"]}"\n\n'
            f"Options:\n{opts}\n\n"
            f"Answer with the letter of the correct option only.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="llm_input.pkl")
    ap.add_argument("--out", default="llm_scores.pkl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct")
    ap.add_argument("--quant", default="fp8", help="fp8 | none")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel GPUs")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--orders", type=int, default=2, help="1 = ranker order; 2 = also reversed order (cancels position bias)")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    blob = pickle.load(open(a.inp, "rb"))
    rows, desc = blob["rows"], blob["desc"]
    if a.limit:
        rows = rows[:: max(1, len(rows) // a.limit)][: a.limit]   # spread over both systems
    llm = LLM(model=a.model, quantization=None if a.quant == "none" else a.quant, tensor_parallel_size=a.tp,
              max_model_len=a.max_len, gpu_memory_utilization=0.92, enable_prefix_caching=True, seed=0)
    tok = llm.get_tokenizer()
    letter_ids = {}
    for L in LETTERS:
        ids = tok.encode(L, add_special_tokens=False)
        assert len(ids) == 1, (L, ids)
        letter_ids[ids[0]] = L
    prompts, owners = [], []
    for r in rows:
        orders = [r["options"], r["options"][::-1]][: a.orders]
        for opts in orders:
            msgs = [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": build_prompt(r, desc, opts)}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            owners.append((r, opts))
    print(f"[llm] {len(prompts)} prompts, example:\n{prompts[0][-1500:]}\n", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20, seed=0, allowed_token_ids=list(letter_ids))
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    print(f"[llm] generated in {time.time() - t0:.0f}s ({(time.time() - t0) / len(prompts) * 1000:.0f} ms/term)", flush=True)
    acc = {}
    for (r, opts), o in zip(owners, outs):
        lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        p = {}
        for tid, v in lp.items():
            L = letter_ids.get(tid)
            if L is not None and LETTERS.index(L) < len(opts):
                p[opts[LETTERS.index(L)]] = math.exp(v.logprob)
        z = sum(p.values()) or 1.0
        d = acc.setdefault(r["qid"], {})
        for c, v in p.items():
            d[c] = d.get(c, 0.0) + v / z / a.orders
    scores, hits = {}, []
    for r in rows:
        p = acc.get(r["qid"], {})
        scores[r["qid"]] = p
        best = max(p, key=p.get) if p else r["options"][0]
        hits.append((r["system"], r.get("sub"), best == r["gold"], r["options"][0] == r["gold"], r["gold"] in r["options"]))
    pickle.dump(scores, open(a.out, "wb"))
    for s in ("ICD10CM", "SNOMEDCT_US"):
        for grp in ("all", "DiagnosisCode"):
            h = [x for x in hits if x[0] == s and (grp == "all" or x[1] == grp)]
            if h:
                n = len(h)
                print(f"[llm] {s:<12} {grp:<13} n={n:<5} LLM top-1={sum(x[2] for x in h) / n:.3f}  "
                      f"ranker top-1={sum(x[3] for x in h) / n:.3f}  gold in options={sum(x[4] for x in h) / n:.3f}")
    print(f"[llm] saved {a.out}")


if __name__ == "__main__":
    main()
