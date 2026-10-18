"""Fine-tuned cross-encoder (v2) at inference time.

Same text format as training (export_ft_input.py --tag + kaggle/finetune_ce_v2.py):
  query = "<entity> | <±8-word context window>"
  doc   = ICD:    "<code> <up to 3 descriptions joined by ' ; '>"
          SNOMED: "<up to 3 descriptions joined by ' ; '> (<semantic tag>)"
"""
import numpy as np

from .data import SNO, SYSTEMS
from .text import clean

MAX_LEN = 160


def context_window(entity, chunk, width=8):
    e, words = clean(entity), clean(chunk).split()
    text = " ".join(words)
    pos = text.find(e) if e else -1
    if pos < 0:
        return " ".join(words[: 2 * width])
    start_w = len(text[:pos].split())
    end_w = start_w + len(e.split())
    return " ".join(words[max(0, start_w - width): end_w + width])


def build_names(D):
    """{system: {code: description text exactly as seen in training}}"""
    names = {}
    for s in SYSTEMS:
        d = D[s].sort_values("order") if s == SNO else D[s]
        n = d.groupby("code").desc.agg(lambda x: " ; ".join(list(dict.fromkeys(x))[:3])).to_dict()
        if s == SNO:
            tag = d.drop_duplicates("code").set_index("code").tag.to_dict()
            n = {c: f"{v} ({tag[c]})" if isinstance(tag.get(c), str) and tag.get(c) else v for c, v in n.items()}
        else:
            n = {c: f"{c} {v}" for c, v in n.items()}
        names[s] = n
    return names


class CrossEncoder:
    def __init__(self, model_dir, names, device=None, batch=64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.dev).eval()
        self.names, self.batch = names, batch

    def score(self, queries, cand_codes):
        """queries: list of (system, entity, chunk); cand_codes: list of code lists -> list of {code: score}"""
        torch = self.torch
        pairs, owners = [], []
        for i, ((s, ent, chunk), codes) in enumerate(zip(queries, cand_codes)):
            q = f"{ent} | {context_window(ent, chunk)}"
            for c in codes:
                pairs.append((q, self.names[s].get(c, str(c))))
                owners.append((i, c))
        out = [dict() for _ in queries]
        with torch.no_grad():
            for i in range(0, len(pairs), self.batch):
                p = pairs[i:i + self.batch]
                enc = self.tok([a for a, _ in p], [b for _, b in p], truncation=True, max_length=MAX_LEN,
                               padding=True, return_tensors="pt").to(self.dev)
                if self.dev == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        s = self.model(**enc).logits.view(-1).float().cpu().numpy()
                else:
                    s = self.model(**enc).logits.view(-1).float().numpy()
                for (qi, c), v in zip(owners[i:i + self.batch], s):
                    out[qi][c] = float(v)
        return out


def add_ce_feats(F, scores, tag="ft2"):
    """Same derived columns as run_fast.py: x_<tag>, _rank, _gap, _prob (over the candidates in F)."""
    F = F.copy()
    x = f"x_{tag}"
    F[x] = np.asarray([scores.get(q, {}).get(c, np.nan) for q, c in zip(F.qid.values, F.code.values)],
                      dtype=np.float32)
    g = F.groupby("qid")[x]
    F[f"{x}_rank"] = g.rank(ascending=False, method="min")
    F[f"{x}_gap"] = g.transform("max") - F[x]
    e = np.exp(F[x] - g.transform("max"))
    F[f"{x}_prob"] = e / e.groupby(F.qid).transform("sum")
    return F
