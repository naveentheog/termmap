# KAGGLE NOTEBOOK (GPU T4 x2, Internet ON) - run the final TermMap pipeline on the assessment file.
#
# Add these as inputs (all PRIVATE):
#   termmap-code      : this repo as a zip/dataset (code + models_final/ + data/ + cache/entity_emb_*.npy)
#   dictionary embeddings output (emb_sapbert_*, emb_gte_*, codes_*.npy)   <- kaggle/embed_dictionaries.py
#   SapBERT v2 output (sapft_full/, emb_sapft_*.npy, codes_sapft_*.npy)     <- kaggle/finetune_bienc.py
#   SapBERT v3 output (sapft3_full/, emb_sapft3_*.npy, codes_sapft3_*.npy)  <- kaggle/finetune_bienc_v3.py
#   cross-encoder v4 output (ce4_full/ or ce4_full.zip)                    <- kaggle/finetune_ce_v4.py
#   lora_full/ (Qwen2.5-7B LoRA adapter)                                    <- llm/finetune_llm_lora.py
#   the assessment CSV
# Paste this file into one cell and run. Output: /kaggle/working/predictions.csv
import glob, json, os, shutil, subprocess, sys, zipfile

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "sentence-transformers", "openpyxl",
                "lightgbm", "rapidfuzz"], check=False)
INP, WORK = "/kaggle/input", "/kaggle/working"


def find_file(name):
    hits = sorted(glob.glob(f"{INP}/**/{name}", recursive=True))
    return hits[0] if hits else None


def find_dir(name, must_have="config.json"):
    for p in sorted(glob.glob(f"{INP}/**/{name}", recursive=True)):
        if os.path.isdir(p) and (not must_have or os.path.exists(os.path.join(p, must_have))):
            return p
    z = find_file(f"{name}.zip")
    if z:
        dst = f"{WORK}/{name}"
        zipfile.ZipFile(z).extractall(dst)
        for root, _, files in os.walk(dst):
            if must_have in files:
                return root
    return None


# code: copy the repo to a writable place (dictionary parsing writes cache pickles next to the xlsx)
repo_src = os.path.dirname(find_file("predict.py"))
REPO = f"{WORK}/termmap"
if not os.path.exists(REPO):
    shutil.copytree(repo_src, REPO)
os.chdir(REPO)

assets = {
    "dict_emb_dir": os.path.dirname(find_file("emb_sapbert_SNOMEDCT_US.npy")),
    "train_emb_dir": os.path.dirname(find_file("entity_emb_sapbert.npy")),
    "bi_model": find_dir("sapft_full"),
    "bi_emb_dir": os.path.dirname(find_file("emb_sapft_SNOMEDCT_US.npy")),
    "bi3_model": find_dir("sapft3_full"),
    "bi3_emb_dir": os.path.dirname(find_file("emb_sapft3_SNOMEDCT_US.npy")),
    "ce_model": find_dir("ce4_full"),
    "llm_base": "Qwen/Qwen2.5-7B-Instruct",
    "llm_adapter": find_dir("lora_full", "adapter_config.json"),
}
# dense.dict_search expects codes_<system>.npy next to the dictionary embeddings
print(json.dumps(assets, indent=1))
missing = [k for k, v in assets.items() if not v]
assert not missing, f"missing inputs: {missing}"
json.dump(assets, open("assets.json", "w"))

ASSESS = os.environ.get("TM_INPUT") or [p for p in glob.glob(f"{INP}/**/*.csv", recursive=True)
                                        if "train" not in os.path.basename(p).lower()][0]
print("[input]", ASSESS)
subprocess.run([sys.executable, "predict.py", "--input", ASSESS, "--out", f"{WORK}/predictions.csv",
                "--assets", "assets.json"], check=True)
print(open(f"{WORK}/predictions.csv").read()[:300])
