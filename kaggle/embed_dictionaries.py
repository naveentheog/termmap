"""KAGGLE NOTEBOOK (GPU T4 x2) — embed the two code dictionaries with SapBERT and GTE.

Privacy: ONLY the public code dictionaries go to Kaggle. The proprietary training/eval
terms are embedded locally on CPU by termmap/dense.py (a few minutes for ~10k short strings).

How to run:
 1. Kaggle -> Create -> New Dataset (PRIVATE) -> upload icd10cm_codes.xlsx and
    SNOMEDCT_US_codes_and_descriptions.xlsx. Name it "termmap-dicts".
 2. New Notebook -> Add Input -> your "termmap-dicts" dataset.
    Settings: Accelerator = GPU T4 x2, Internet = ON.
 3. Paste this whole file into one cell and run (~30-60 min for both models).
 4. Download from the Output tab: emb_<model>_<system>.npy and codes_<system>.npy
    -> put them in the project's cache/ folder.
"""
import os
import re
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers", "openpyxl"], check=True)

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, models

IN = "/kaggle/input/termmap-dicts"
OUT = "/kaggle/working"
MODELS = {
    "sapbert": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",   # CLS pooling (as in the paper)
    "gte": "thenlper/gte-base",                                     # HiLabs' own index uses gte-base
}
torch.manual_seed(0)
_TAG = re.compile(r"\s*\(([^()]+)\)\s*$")
_non = re.compile(r"[^a-z0-9+]+")
clean = lambda t: re.sub(r"\s+", " ", _non.sub(" ", str(t).lower().replace("<br>", " "))).strip()


URLS = {"icd10cm": "https://hiringdrive2026.blob.core.windows.net/hackathon/icd10cm_codes.xlsx",
        "snomed": "https://hiringdrive2026.blob.core.windows.net/hackathon/SNOMEDCT_US_codes_and_descriptions.xlsx"}


def find(name_part):
    """Use an attached dataset if present, else download the public dictionary from HiLabs' link."""
    if os.path.isdir(IN):
        for f in os.listdir(IN):
            if name_part.lower() in f.lower():
                return os.path.join(IN, f)
    import urllib.request
    dst = f"/tmp/{name_part}.xlsx"
    if not os.path.exists(dst):
        print("downloading", URLS[name_part.lower()]); urllib.request.urlretrieve(URLS[name_part.lower()], dst)
    return dst


def load_dicts():
    icd = pd.read_excel(find("icd10cm"), dtype=str).rename(columns={"Code": "code", "Description": "desc"}).dropna()
    icd = icd[~icd.code.str.contains("-")]
    sno = pd.read_excel(find("snomed"), sheet_name="SNOMED codes", dtype=str)
    sno = sno.rename(columns={"snomed_code": "code", "description": "desc"}).dropna(subset=["code", "desc"])
    tags = sno.desc.str.extract(_TAG)[0]
    freq = tags.value_counts(); ok = tags.isin(set(freq[freq >= 200].index))
    sno["desc"] = sno.desc.where(~ok, sno.desc.str.replace(_TAG, "", regex=True))
    out = {}
    for name, d in (("ICD10CM", icd), ("SNOMEDCT_US", sno)):
        d = d.assign(code=d.code.str.strip(), text=d.desc.map(clean))
        d = d[d.text != ""].drop_duplicates(["code", "text"]).reset_index(drop=True)
        out[name] = d
        print(name, len(d), "descriptions")
    return out


def load_model(key):
    name = MODELS[key]
    if key == "sapbert":
        word = models.Transformer(name, max_seq_length=32)
        pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="cls")
        return SentenceTransformer(modules=[word, pool], device="cuda")
    m = SentenceTransformer(name, device="cuda"); m.max_seq_length = 64
    return m


dicts = load_dicts()
for system, d in dicts.items():
    np.save(f"{OUT}/codes_{system}.npy", d.code.values.astype(str))
    with open(f"{OUT}/texts_{system}.txt", "w") as fh:
        fh.write("\n".join(d.text))
for key in MODELS:
    model = load_model(key)
    model.half()
    for system, d in dicts.items():
        emb = model.encode(d.text.tolist(), batch_size=1024, show_progress_bar=True,
                           normalize_embeddings=True, convert_to_numpy=True).astype(np.float16)
        np.save(f"{OUT}/emb_{key}_{system}.npy", emb)
        print(key, system, emb.shape)
    del model; torch.cuda.empty_cache()
print("done ->", os.listdir(OUT))
