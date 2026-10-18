"""Download all TermMap weights from Hugging Face into weights/ (one-time setup).

python tools/download_weights.py                      # uses the repo id below
python tools/download_weights.py --repo <user>/<repo> # or another copy

Contents of the weights repo:
  sapft_full/  sapft3_full/  ce4_full/  lora_full/      fine-tuned models (ours)
  dict_emb/  sapft_emb/  sapft3_emb/                    dictionary embeddings (public dictionaries only)
Qwen2.5-7B-Instruct, SapBERT and GTE are pulled from their official Hugging Face repos on first use.
"""
import argparse
import os

HF_REPO = os.environ.get("TERMMAP_HF_REPO", "Naveenmeena/termmap-weights")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--out", default="weights")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=a.repo, repo_type="model", local_dir=a.out, token=os.environ.get("HF_TOKEN") or None)
    need = ["sapft_full", "sapft3_full", "ce4_full", "lora_full", "dict_emb", "sapft_emb", "sapft3_emb"]
    missing = [d for d in need if not os.path.isdir(os.path.join(a.out, d))]
    print("[weights] ready" if not missing else f"[weights] missing {missing}: run tools/build_dict_embeddings.py for *_emb")


if __name__ == "__main__":
    main()
