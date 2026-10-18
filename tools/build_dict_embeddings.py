"""Setup step (GPU, ~10-20 min): embed the two PUBLIC code dictionaries with the 4 encoders.

python tools/build_dict_embeddings.py --weights weights
Writes (exactly as the training notebooks did):
  weights/dict_emb/   codes_<sys>.npy, emb_sapbert_<sys>.npy, emb_gte_<sys>.npy   (kaggle/embed_dictionaries.py)
  weights/sapft_emb/  codes_sapft_<sys>.npy,  emb_sapft_<sys>.npy                  (fine-tuned SapBERT v2)
  weights/sapft3_emb/ codes_sapft3_<sys>.npy, emb_sapft3_<sys>.npy                 (fine-tuned SapBERT v3)
Skips any file that already exists.
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from termmap.data import SYSTEMS, dictionaries  # noqa: E402
from termmap.dense import clean, load_model  # noqa: E402

_TAG = re.compile(r"\s*\(([^()]+)\)\s*$")


def base_texts(icd_xlsx, sno_xlsx):
    """Same text as kaggle/embed_dictionaries.py."""
    icd = pd.read_excel(icd_xlsx, dtype=str).rename(columns={"Code": "code", "Description": "desc"}).dropna()
    icd = icd[~icd.code.str.contains("-")]
    sno = pd.read_excel(sno_xlsx, sheet_name="SNOMED codes", dtype=str)
    sno = sno.rename(columns={"snomed_code": "code", "description": "desc"}).dropna(subset=["code", "desc"])
    tags = sno.desc.str.extract(_TAG)[0]
    freq = tags.value_counts()
    ok = tags.isin(set(freq[freq >= 200].index))
    sno["desc"] = sno.desc.where(~ok, sno.desc.str.replace(_TAG, "", regex=True))
    out = {}
    for name, d in (("ICD10CM", icd), ("SNOMEDCT_US", sno)):
        d = d.assign(code=d.code.str.strip(), text=d.desc.map(clean))
        out[name] = d[d.text != ""].drop_duplicates(["code", "text"]).reset_index(drop=True)
    return out


def tuned_texts(D):
    """Same text as export_bienc_input.py (SNOMED descriptions carry the semantic tag)."""
    out = {}
    for s in SYSTEMS:
        d = D[s][["code", "desc"] + (["tag"] if "tag" in D[s] else [])].copy()
        d["text"] = d.desc.map(clean)
        if "tag" in d:
            d["text"] = [f"{t} {tg}".strip() if isinstance(tg, str) and tg else t for t, tg in zip(d.text, d.tag)]
        d = d[d.text.str.len() > 0].drop_duplicates(["code", "text"])
        out[s] = (d.code.astype(str).tolist(), d.text.tolist())
    return out


def encode_tuned(model_dir, texts, bs=1024):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    m = AutoModel.from_pretrained(model_dir).cuda().eval()
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], truncation=True, max_length=40, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            v = m(**b).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(v.float(), dim=-1).half().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--icd", default="data/icd.xlsx")
    ap.add_argument("--snomed", default="data/snomed.xlsx")
    a = ap.parse_args()
    W = a.weights
    os.makedirs(f"{W}/dict_emb", exist_ok=True)
    bt = base_texts(a.icd, a.snomed)
    for s, d in bt.items():
        np.save(f"{W}/dict_emb/codes_{s}.npy", d.code.values.astype(str))
    for key in ("sapbert", "gte"):
        if all(os.path.exists(f"{W}/dict_emb/emb_{key}_{s}.npy") for s in SYSTEMS):
            continue
        m = load_model(key, device="cuda")
        m.half()
        for s, d in bt.items():
            e = m.encode(d.text.tolist(), batch_size=1024, normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=True).astype(np.float16)
            np.save(f"{W}/dict_emb/emb_{key}_{s}.npy", e)
            print(key, s, e.shape, flush=True)
    tt = tuned_texts(dictionaries(a.icd, a.snomed))
    for name in ("sapft", "sapft3"):
        od = f"{W}/{name}_emb"
        os.makedirs(od, exist_ok=True)
        for s, (codes, texts) in tt.items():
            if os.path.exists(f"{od}/emb_{name}_{s}.npy"):
                continue
            np.save(f"{od}/codes_{name}_{s}.npy", np.asarray(codes))
            np.save(f"{od}/emb_{name}_{s}.npy", encode_tuned(f"{W}/{name}_full", texts))
            print(name, s, len(codes), flush=True)
    print("[setup] dictionary embeddings ready")


if __name__ == "__main__":
    main()
