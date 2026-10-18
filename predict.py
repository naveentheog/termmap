"""Single command that produces the submission file (run on a GPU machine, e.g. Kaggle T4 x2).

python predict.py --input assessment.csv --out predictions.csv --assets assets.json

Input  : CSV with clinical_entity (or entity) and chunk_text [, chunk_id]   (wide format: one row per entity)
         or entity + coding_system [+ chunk_text]                            (long format: one code per row)
Output : wide -> chunk_id, clinical_entity, icd10cm_code, snomed_code ; long -> input columns + code
         plus <out>_debug.csv with ranker confidence (margin between best and runner-up) per prediction.
assets.json: paths of the model files, see termmap/neural_gpu.py.
Deterministic: no sampling, ties broken by code string, LightGBM deterministic=True.
"""
import argparse
import json
import os
import time

import pandas as pd


def load_dotenv(path=".env"):
    """Minimal .env reader (KEY=VALUE lines); real environment variables win."""
    if os.path.exists(path):
        for line in open(path):
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                if k.strip() and v.strip():
                    os.environ.setdefault(k.strip(), v.strip())


load_dotenv()

from termmap.data import load_train
from termmap.engine_final import FinalMapper
from termmap.neural_gpu import GPUNeural

PUBLIC_DICTS = {
    "icd": ("icd10cm_codes.xlsx", "https://hiringdrive2026.blob.core.windows.net/hackathon/icd10cm_codes.xlsx"),
    "snomed": ("SNOMEDCT_US_codes_and_descriptions.xlsx",
               "https://hiringdrive2026.blob.core.windows.net/hackathon/SNOMEDCT_US_codes_and_descriptions.xlsx"),
}


def resolve(path, kind=None):
    """Use the given path; else the original HiLabs file name in data/; else (dictionaries only) download it."""
    if os.path.exists(path):
        return path
    if kind:
        name, url = PUBLIC_DICTS[kind]
        alt = os.path.join(os.path.dirname(path) or "data", name)
        if os.path.exists(alt):
            return alt
        import urllib.request
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        print(f"[predict] {path} not found, downloading the public dictionary {name}")
        urllib.request.urlretrieve(url, path)
        return path
    raise SystemExit(f"[predict] missing {path}: put the HiLabs training file there (see data/README.txt) or pass --train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="outputs/predictions.csv")
    ap.add_argument("--train", default="data/hackathon_train.csv")
    ap.add_argument("--icd", default="data/icd.xlsx")
    ap.add_argument("--snomed", default="data/snomed.xlsx")
    ap.add_argument("--models", default="models_final")
    ap.add_argument("--assets", default="assets.json")
    ap.add_argument("--no-llm", action="store_true", help="skip the 7B LLM (CPU-only machines): ranker A output")
    a = ap.parse_args()

    t0 = time.time()
    inp = pd.read_csv(a.input, dtype=str, keep_default_na=False)
    inp.columns = [c.strip() for c in inp.columns]
    assets = json.load(open(a.assets))
    assets["no_llm"] = a.no_llm
    tm = FinalMapper(load_train(resolve(a.train)), resolve(a.icd, "icd"), resolve(a.snomed, "snomed"), a.models)
    q, mode = tm.predict(inp, GPUNeural(tm, assets))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if mode == "long":
        out = inp.copy()
        out["code"] = q.sort_values("src_row").code.values
    else:
        ent = "clinical_entity" if "clinical_entity" in inp else "entity"
        out = pd.DataFrame({"chunk_id": inp["chunk_id"] if "chunk_id" in inp else range(len(inp)),
                            "clinical_entity": inp[ent]})
        for s, col in (("ICD10CM", "icd10cm_code"), ("SNOMEDCT_US", "snomed_code")):
            out[col] = q[q.coding_system == s].sort_values("src_row").code.values
    out.to_csv(a.out, index=False)
    q.drop(columns=["chunk_text"]).to_csv(a.out.replace(".csv", "_debug.csv"), index=False)
    print(f"[predict] {len(out):,} rows ({mode} format) -> {a.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
