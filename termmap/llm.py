"""Fine-tuned LLM adjudicator at inference time (Qwen2.5-7B-Instruct + LoRA adapter `lora_full`).

Same prompt as training (export_llm_input.py + llm/finetune_llm_lora.py):
  - term + ±25 words of chart context
  - 8 most similar TRAINING terms (base SapBERT entity similarity, same system) with their gold code
  - the first-stage ranker's top-10 candidates (ICD descriptions carry their 3-character category name)
One forward pass per option order (ranker order and reversed); only the last position's logits over the
letters A..J are read -> deterministic probabilities, no sampling.
"""
import numpy as np

from .crossenc import build_names
from .data import ICD, SNO
from .text import clean

LETTERS = "ABCDEFGHIJ"
K_SHOT = 8
CTX_WIDTH = 25
SYSTEM_MSG = (
    "You are an expert clinical coder. Map the clinical term to exactly ONE code from the options. "
    "Follow the coding conventions shown in the examples from this organisation. Read the context: "
    "family history, personal history, negation, screening, plans, laterality, acuity and encounter type "
    "change the correct code. Choose an unspecified code only when the text supports nothing more specific."
)
SYS_NAME = {ICD: "ICD-10-CM", SNO: "SNOMED CT"}


def context_window(entity, chunk, width=CTX_WIDTH):
    e, words = clean(entity), clean(chunk).split()
    text = " ".join(words)
    pos = text.find(e) if e else -1
    if pos < 0:
        return " ".join(words[: 2 * width])
    start_w = len(text[:pos].split())
    end_w = start_w + len(e.split())
    return " ".join(words[max(0, start_w - width): end_w + width])


def llm_names(D):
    names = build_names(D)
    names[ICD] = {c: (d.split(" ", 1)[1] if d.startswith(c + " ") else d) for c, d in names[ICD].items()}
    return names


def option_desc(names, system, code):
    d = names[system].get(code, code)
    if system == ICD and len(code) > 3 and code[:3] in names[ICD]:
        d = f"{d}  [category {code[:3]}: {names[ICD][code[:3]].split(' ; ')[0]}]"
    return d


def build_prompt(r, names, options):
    shots = "\n".join(f'- "{e}" -> {c} {d}' for e, c, d in r["shots"])
    opts = "\n".join(f"{LETTERS[i]}. {c} {option_desc(names, r['system'], c)}" for i, c in enumerate(options))
    return (f"Coding system: {SYS_NAME[r['system']]}\n\n"
            f"How our coders mapped similar terms:\n{shots}\n\n"
            f'Term: "{r["entity"]}"\n'
            f'Chart context: "{r["context"]}"\n\n'
            f"Options:\n{opts}\n\n"
            f"Answer with the letter of the correct option only.")


def make_rows(q, top10, train, train_emb, q_emb, names):
    """q: query frame (qid index); top10: {qid: [codes]}; train_emb / q_emb: L2-normalised SapBERT."""
    rows = []
    for s in (ICD, SNO):
        ti = np.flatnonzero(train.coding_system.values == s)
        qi = [i for i, qid in enumerate(q.index) if q.coding_system.iat[i] == s and qid in top10]
        if not qi or not len(ti):
            continue
        sims = q_emb[qi] @ train_emb[ti].T
        for a, i in enumerate(qi):
            nn = ti[np.argsort(-sims[a], kind="stable")[:K_SHOT]]
            shots = [(train.entity.iat[j], train.code.iat[j], names[s].get(train.code.iat[j], train.code.iat[j]))
                     for j in nn]
            qid = q.index[i]
            rows.append({"qid": qid, "system": s, "entity": q.entity.iat[i],
                         "context": context_window(q.entity.iat[i], q.chunk_text.iat[i]),
                         "options": list(top10[qid]), "shots": shots})
    return rows


class LLMScorer:
    def __init__(self, base="Qwen/Qwen2.5-7B-Instruct", adapter="lora_full", batch=8, max_len=1536):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.dtype = torch.bfloat16 if bf16 else torch.float16
        self.tok = AutoTokenizer.from_pretrained(adapter)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=self.dtype, device_map="auto")
        self.model = PeftModel.from_pretrained(model, adapter).eval()
        self.letter_ids = [self.tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]
        self.batch, self.max_len = batch, max_len

    def _prompt(self, r, names, opts):
        msgs = [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": build_prompt(r, names, opts)}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _letter_logits(self, prompts):
        torch = self.torch
        enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
        base = self.model.get_base_model()
        dev = base.model.embed_tokens.weight.device
        with torch.no_grad():
            h = base.model(input_ids=enc["input_ids"].to(dev),
                           attention_mask=enc["attention_mask"].to(dev)).last_hidden_state[:, -1]
            w = base.lm_head.weight[self.letter_ids]
            return (h.to(w.device) @ w.T).float()

    def score(self, rows, names, verbose=True):
        out = {}
        for i in range(0, len(rows), self.batch):
            batch = rows[i:i + self.batch]
            acc = [dict() for _ in batch]
            for order in (lambda o: o, lambda o: o[::-1]):
                opts = [order(r["options"]) for r in batch]
                lg = self._letter_logits([self._prompt(r, names, o) for r, o in zip(batch, opts)])
                for j, o in enumerate(opts):
                    p = self.torch.softmax(lg[j, :len(o)], 0).cpu().numpy()
                    for c, v in zip(o, p):
                        acc[j][c] = acc[j].get(c, 0.0) + float(v) / 2
            for r, a in zip(batch, acc):
                out[r["qid"]] = a
            if verbose and (i // self.batch) % 50 == 0:
                print(f"[llm] {i + len(batch)}/{len(rows)}", flush=True)
        return out
