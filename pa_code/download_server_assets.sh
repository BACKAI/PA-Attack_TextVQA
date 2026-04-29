#!/usr/bin/env bash
set -euo pipefail

PA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PA_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
HF_MODEL_ID="${HF_MODEL_ID:-liuhaotian/llava-v1.5-7b}"
MODEL_DIR="${MODEL_DIR:-models/llava-v1.5-7b}"
PROTOTYPE_PATH="${PROTOTYPE_PATH:-prototypes_llava_tokens_bestpca/prototypes_tokens_3000_20_1024.pt}"
PROTOTYPE_URL="${PROTOTYPE_URL:-}"

mkdir -p "$(dirname "$MODEL_DIR")" "$(dirname "$PROTOTYPE_PATH")"

echo "Installing Hugging Face CLI if needed..."
"$PYTHON_BIN" -m pip install -U "huggingface_hub[cli]"

if [[ "${DOWNLOAD_LLAVA:-1}" == "1" ]]; then
  echo "Downloading LLaVA model from Hugging Face: $HF_MODEL_ID"
  huggingface-cli download "$HF_MODEL_ID" \
    --local-dir "$MODEL_DIR" \
    --local-dir-use-symlinks False
  echo "LLaVA model path: $MODEL_DIR"
fi

if [[ -n "$PROTOTYPE_URL" ]]; then
  echo "Downloading PA-Attack prototype from PROTOTYPE_URL"
  if command -v curl >/dev/null 2>&1; then
    curl -L "$PROTOTYPE_URL" -o "$PROTOTYPE_PATH"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$PROTOTYPE_PATH" "$PROTOTYPE_URL"
  else
    echo "Neither curl nor wget is available." >&2
    exit 1
  fi
  echo "Prototype path: $PROTOTYPE_PATH"
else
  cat <<EOF
No PROTOTYPE_URL was provided.

The public PA-Attack repository currently has no release asset for:
  $PROTOTYPE_PATH

Options:
  1. Put your private URL in PROTOTYPE_URL and rerun this script.
  2. Generate it on the server with the original PA-Attack workflow:
       CUDA_VISIBLE_DEVICES=0 bash bash/llava_prototype_generation.sh

Note: prototype/prototype_pca.py has a hard-coded COCO path:
  /home/datasets/coco2014/val2014
Adjust that path or provide a matching symlink before generation.
EOF
fi
