"""GPU-side encoders used at inference: fine-tuned SapBERT retrievers (v2, v3).

Query text = cleaned entity (as in kaggle/finetune_bienc*.py); CLS pooling, L2-normalised, max 40 tokens.
Dictionary side = embeddings saved by the training notebook (emb_<name>_<system>.npy + codes_<name>_<system>.npy).
"""
import os

import numpy as np

from .text import clean

SYSTEMS = ("ICD10CM", "SNOMEDCT_US")


class BiEncoder:
    def __init__(self, model_dir, emb_dir, name, device=None, max_len=40):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).to(self.dev).eval()
        self.max_len = max_len
        self.D = {s: np.load(os.path.join(emb_dir, f"emb_{name}_{s}.npy"), mmap_mode="r") for s in SYSTEMS}
        self.C = {s: np.load(os.path.join(emb_dir, f"codes_{name}_{s}.npy"), allow_pickle=True) for s in SYSTEMS}

    def encode(self, texts, bs=512):
        torch = self.torch
        out = []
        for i in range(0, len(texts), bs):
            b = self.tok(texts[i:i + bs], truncation=True, max_length=self.max_len, padding=True,
                         return_tensors="pt").to(self.dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.dev == "cuda"):
                v = self.model(**b).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)

    def search(self, entities, systems, topk=100):
        """-> list of {code: cosine} (top-k distinct codes, descending)."""
        torch = self.torch
        q = self.encode([clean(e) for e in entities])
        res = [dict() for _ in entities]
        systems = np.asarray(systems)
        for s in SYSTEMS:
            qi = np.flatnonzero(systems == s)
            if not len(qi):
                continue
            D = torch.from_numpy(np.asarray(self.D[s], dtype=np.float16)).to(self.dev)
            codes = self.C[s]
            for a in range(0, len(qi), 256):
                ids = qi[a:a + 256]
                qq = torch.from_numpy(q[ids]).to(self.dev).to(D.dtype)
                sims = (qq @ D.T).float()
                val, idx = sims.topk(min(topk * 4, sims.shape[1]), dim=1)
                for i, vs, ix in zip(ids, val.cpu().numpy(), idx.cpu().numpy()):
                    agg = {}
                    for v, j in zip(vs, ix):
                        c = str(codes[j])
                        if c not in agg:
                            agg[c] = float(v)
                        if len(agg) >= topk:
                            break
                    res[i] = agg
            del D
        return res
