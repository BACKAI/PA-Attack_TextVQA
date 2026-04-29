#!/usr/bin/env bash
set -euo pipefail

PA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PA_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
COCO_VAL_IMAGE_DIR="${COCO_VAL_IMAGE_DIR:-/var/tmp/jnuadmin_vlm/VLM/dataset/MSCOCO/val2014}"
MODEL_PATH="${MODEL_PATH:-/var/tmp/jnuadmin_vlm/VLM/Attack/PA-Attack_TextVQA/models/llava-v1.5-7b}"
PROTOTYPE_PATH="${PROTOTYPE_PATH:-prototypes_llava_tokens_bestpca/prototypes_tokens_3000_20_1024.pt}"
NUM_SAMPLES="${NUM_SAMPLES:-3000}"
NUM_PROTOTYPES="${NUM_PROTOTYPES:-20}"
PCA_DIM="${PCA_DIM:-1024}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PRECISION="${PRECISION:-float16}"

export PYTHONPATH="$PA_ROOT:${PYTHONPATH:-}"

"$PYTHON_BIN" pa_code/generate_llava_prototype_server.py \
  --coco-val-image-dir "$COCO_VAL_IMAGE_DIR" \
  --model-path "$MODEL_PATH" \
  --output "$PROTOTYPE_PATH" \
  --num-samples "$NUM_SAMPLES" \
  --num-prototypes "$NUM_PROTOTYPES" \
  --pca-dim "$PCA_DIM" \
  --batch-size "$BATCH_SIZE" \
  --precision "$PRECISION"
