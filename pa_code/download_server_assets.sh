#!/usr/bin/env bash
set -euo pipefail

PA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PA_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
HF_MODEL_ID="${HF_MODEL_ID:-liuhaotian/llava-v1.5-7b}"
MODEL_DIR="${MODEL_DIR:-models/llava-v1.5-7b}"
PROTOTYPE_PATH="${PROTOTYPE_PATH:-prototypes_llava_tokens_bestpca/prototypes_tokens_3000_20_1024.pt}"
PROTOTYPE_URL="${PROTOTYPE_URL:-}"
INSTALL_HF_CLI="${INSTALL_HF_CLI:-1}"

mkdir -p "$(dirname "$MODEL_DIR")" "$(dirname "$PROTOTYPE_PATH")"

ensure_hf_hub_compatible() {
  if [[ "$INSTALL_HF_CLI" != "1" ]]; then
    return 0
  fi

  if "$PYTHON_BIN" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

try:
    value = version("huggingface-hub")
except PackageNotFoundError:
    raise SystemExit(1)

major = int(value.split(".", 1)[0])
raise SystemExit(0 if major < 1 else 1)
PY
  then
    return 0
  fi

  echo "Installing a transformers-compatible Hugging Face Hub package..."
  "$PYTHON_BIN" -m pip install "huggingface_hub>=0.36.2,<1.0"
}

ensure_hf_cli() {
  if command -v hf >/dev/null 2>&1 || command -v huggingface-cli >/dev/null 2>&1; then
    return 0
  fi

  if [[ "$INSTALL_HF_CLI" != "1" ]]; then
    echo "Neither hf nor huggingface-cli is available, and INSTALL_HF_CLI=$INSTALL_HF_CLI." >&2
    exit 1
  fi

  echo "Installing a transformers-compatible Hugging Face CLI..."
  "$PYTHON_BIN" -m pip install "huggingface_hub>=0.36.2,<1.0"
}

download_hf_repo() {
  local repo_id="$1"
  local local_dir="$2"

  ensure_hf_cli

  if command -v hf >/dev/null 2>&1; then
    hf download "$repo_id" --local-dir "$local_dir"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$repo_id" \
      --local-dir "$local_dir" \
      --local-dir-use-symlinks False
  else
    echo "Hugging Face CLI is still unavailable after installation." >&2
    exit 1
  fi
}

ensure_hf_hub_compatible

if [[ "${DOWNLOAD_LLAVA:-1}" == "1" ]]; then
  echo "Downloading LLaVA model from Hugging Face: $HF_MODEL_ID"
  download_hf_repo "$HF_MODEL_ID" "$MODEL_DIR"
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
