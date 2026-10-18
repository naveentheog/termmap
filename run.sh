#!/usr/bin/env bash
# Single command: evaluation CSV in -> predictions CSV out.
#   bash run.sh <evaluation.csv> [predictions.csv]
# Downloads our weights from Hugging Face on the first run, then predicts (deterministic).
set -e
IN="${1:?usage: bash run.sh <evaluation.csv> [predictions.csv]}"
OUT="${2:-predictions.csv}"
cd "$(dirname "$0")"
if [ ! -f weights/lora_full/adapter_model.safetensors ]; then
  python tools/download_weights.py
fi
python predict.py --input "$IN" --out "$OUT" "${@:3}"
