"""Zero-shot Qwen2.5-72B adjudicator, checkpointed in chunks (safe on Spot GPUs).

python llm72.py --inp llm_input.pkl --out llm_scores.pkl
Resumes automatically: qids already in --out are skipped. Saves after every chunk.
"""
import argparse, math, os, pickle, time

LETTERS = "ABCDEFGHIJ"
SYSTEM_MSG = (
    "You are an expert clinical coder. Map the clinical term to exactly ONE code from the options. "
    "Follow the coding conventions shown in the examples from this organisation. Read the context: "
    "family history, personal history, negation, screening, plans, laterality, acuity and encounter type "
    "change the correct code. Choose an unspecified code only when the text supports nothing more specific."
)
SYS_NAME = {"ICD10CM": "ICD-10-CM", "SNOMEDCT_US": "SNOMED CT"}


def build_prompt(r, desc, options):
    shots = "\n".join(f'- "{e}" -> {c} {d}' for e, c, d in r["shots"])
    opts = "\n".join(f"{LETTERS[i]}. {c} {desc[(r['system'], c)]}" for i, c in enumerate(options))
    return (f"Coding system: {SYS_NAME[r['system']]}\n\nHow our coders mapped similar terms:\n{shots}\n\n"
            f'Term: "{r["entity"]}"\nChart context: "{r["context"]}"\n\nOptions:\n{opts}\n\n'
            f"Answer with the letter of the correct option only.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="llm_input.pkl")
    ap.add_argument("--out", default="llm_scores.pkl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8")
    ap.add_argument("--quant", default="auto", help="auto (from checkpoint) | fp8")
    ap.add_argument("--chunk", type=int, default=800)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=4096)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    blob = pickle.load(open(a.inp, "rb"))
    rows, desc = blob["rows"], blob["desc"]
    if a.limit:
        rows = rows[:: max(1, len(rows) // a.limit)][: a.limit]
    done = pickle.load(open(a.out, "rb")) if os.path.exists(a.out) else {}
    todo = [r for r in rows if r["qid"] not in done]
    print(f"[72b] {len(done)} done, {len(todo)} to go", flush=True)
    llm = LLM(model=a.model, quantization=None if a.quant == "auto" else a.quant, max_model_len=a.max_len,
              gpu_memory_utilization=0.92, enable_prefix_caching=True, seed=0)
    tok = llm.get_tokenizer()
    lid = {}
    for L in LETTERS:
        ids = tok.encode(L, add_special_tokens=False)
        assert len(ids) == 1
        lid[ids[0]] = L
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20, seed=0, allowed_token_ids=list(lid))
    t0 = time.time()
    for s in range(0, len(todo), a.chunk):
        part = todo[s:s + a.chunk]
        prompts, owners = [], []
        for r in part:
            for opts in (r["options"], r["options"][::-1]):
                m = [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": build_prompt(r, desc, opts)}]
                prompts.append(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True))
                owners.append((r, opts))
        outs = llm.generate(prompts, sp, use_tqdm=False)
        acc = {}
        for (r, opts), o in zip(owners, outs):
            lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            p = {}
            for t, v in lp.items():
                L = lid.get(t)
                if L is not None and LETTERS.index(L) < len(opts):
                    p[opts[LETTERS.index(L)]] = math.exp(v.logprob)
            z = sum(p.values()) or 1.0
            d = acc.setdefault(r["qid"], {})
            for c, v in p.items():
                d[c] = d.get(c, 0.0) + v / z / 2
        done.update(acc)
        pickle.dump(done, open(a.out + ".tmp", "wb")); os.replace(a.out + ".tmp", a.out)
        el = time.time() - t0
        print(f"[72b] {s + len(part)}/{len(todo)} rows  {el:.0f}s  eta {el / (s + len(part)) * (len(todo) - s - len(part)):.0f}s", flush=True)
    hits = {}
    for r in rows:
        p = done.get(r["qid"])
        if not p:
            continue
        h = hits.setdefault(r["system"], [0, 0, 0])
        h[0] += 1; h[1] += max(p, key=p.get) == r["gold"]; h[2] += r["options"][0] == r["gold"]
    for s, (n, l, k) in hits.items():
        print(f"[72b] {s:<12} n={n} LLM top-1={l / n:.3f} ranker top-1={k / n:.3f}", flush=True)


if __name__ == "__main__":
    main()
