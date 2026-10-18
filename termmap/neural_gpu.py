"""Real neural components for FinalMapper.predict (run on a GPU machine, e.g. a Kaggle T4 x2 notebook).

assets (paths):
  dict_emb_dir : emb_sapbert_<sys>.npy, emb_gte_<sys>.npy, codes_<sys>.npy   (kaggle/embed_dictionaries.py)
  train_emb_dir: entity_emb_sapbert.npy, entity_emb_gte.npy                   (training entities, same models)
  bi_model, bi_emb_dir   : sapft_full/  + emb_sapft_<sys>.npy,  codes_sapft_<sys>.npy   (SapBERT v2)
  bi3_model, bi3_emb_dir : sapft3_full/ + emb_sapft3_<sys>.npy, codes_sapft3_<sys>.npy  (SapBERT v3)
  ce_model     : ce4_full/  (MedCPT cross-encoder v4)
  llm_base, llm_adapter : Qwen/Qwen2.5-7B-Instruct + lora_full/
"""
import gc
import os

import numpy as np

from . import dense as dn
from .crossenc import build_names
from .data import SYSTEMS
from .hybrid import train_neighbours_query
from .llm import CTX_WIDTH, LLMScorer, context_window, llm_names, make_rows

CE_MAX_LEN = 256


def _free():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


class GPUNeural:
    def __init__(self, mapper, assets):
        self.m, self.a = mapper, assets
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_llm = bool(assets.get("llm_adapter")) and not assets.get("no_llm") and self.device == "cuda"
        self.q_emb = {}

    def dense(self, q):
        rows = [dict() for _ in range(len(q))]
        neigh = {}
        for key in ("sapbert", "gte"):
            model = dn.load_model(key, device=self.device)
            e = dn.embed(key, q.entity.tolist(), model)
            self.q_emb[key] = e
            for s in SYSTEMS:
                idx = np.flatnonzero(q.coding_system.values == s)
                if len(idx):
                    for i, d in zip(idx, dn.dict_search(key, s, e[idx], self.a["dict_emb_dir"])):
                        rows[i][key] = d
            temb = self._train_emb(key, model)
            neigh[key] = train_neighbours_query(e, q.coding_system.values, self.m.train, temb)
            del model
            _free()
        return rows, neigh

    def _train_emb(self, key, model=None):
        """Embeddings of the training entities (computed once, then cached next to the other weights)."""
        path = os.path.join(self.a["train_emb_dir"], f"entity_emb_{key}.npy")
        if os.path.exists(path):
            e = np.load(path)
            if len(e) == len(self.m.train):
                return e.astype(np.float32)
        e = dn.embed(key, self.m.train.entity.tolist(), model or dn.load_model(key, device=self.device))
        os.makedirs(self.a["train_emb_dir"], exist_ok=True)
        np.save(path, e.astype(np.float16))
        return e.astype(np.float32)

    def bi(self, q, name):
        from .neural import BiEncoder
        tag = {"bi": "sapft", "bi3": "sapft3"}[name]
        enc = BiEncoder(self.a[f"{name}_model"], self.a[f"{name}_emb_dir"], tag, device=self.device)
        out = enc.search(q.entity.tolist(), q.coding_system.values, topk=100)
        del enc
        _free()
        return out

    def ce(self, q, codes_by_qid):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        names = build_names(self.m.D)
        tok = AutoTokenizer.from_pretrained(self.a["ce_model"])
        model = AutoModelForSequenceClassification.from_pretrained(self.a["ce_model"]).to(self.device).eval()
        pairs, owners = [], []
        for qid, codes in codes_by_qid.items():
            ent = q.entity.iat[qid]
            qt = f"{ent} | {context_window(ent, q.chunk_text.iat[qid], 25)}"
            for c in codes:
                pairs.append((qt, names[q.coding_system.iat[qid]].get(c, str(c))))
                owners.append((qid, c))
        out = {}
        with torch.no_grad():
            bs = 256 if self.device == "cuda" else 32
            for i in range(0, len(pairs), bs):
                p = pairs[i:i + bs]
                enc = tok([a for a, _ in p], [b for _, b in p], truncation=True, max_length=CE_MAX_LEN,
                          padding=True, return_tensors="pt").to(self.device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                    s = model(**enc).logits.view(-1).float().cpu().numpy()
                for (qid, c), v in zip(owners[i:i + bs], s):
                    out.setdefault(qid, {})[c] = float(v)
        del model
        _free()
        return out

    def llm(self, q, top10):
        names = llm_names(self.m.D)
        temb = self._train_emb("sapbert")
        temb /= np.linalg.norm(temb, axis=1, keepdims=True) + 1e-9
        qe = self.q_emb["sapbert"].astype(np.float32)
        qe /= np.linalg.norm(qe, axis=1, keepdims=True) + 1e-9
        rows = make_rows(q, top10, self.m.train, temb, qe, names)
        scorer = LLMScorer(self.a.get("llm_base", "Qwen/Qwen2.5-7B-Instruct"), self.a["llm_adapter"])
        out = scorer.score(rows, names)
        del scorer
        _free()
        return out
