#!/usr/bin/env bash
set -euo pipefail

PA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PA_ROOT"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-pa}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA_WHEEL="${CUDA_WHEEL:-cu118}"
TORCH_VERSION="${TORCH_VERSION:-2.0.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.15.2}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers==4.31.0}"
INSTALL_FULL_REQUIREMENTS="${INSTALL_FULL_REQUIREMENTS:-0}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Load conda first, then rerun this script." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV_NAME"; then
  echo "Using existing conda environment: $CONDA_ENV_NAME"
else
  echo "Creating conda environment: $CONDA_ENV_NAME"
  conda create -y -n "$CONDA_ENV_NAME" "python=$PYTHON_VERSION" pip
fi

run_in_env() {
  conda run -n "$CONDA_ENV_NAME" "$@"
}

echo "Upgrading base packaging tools in $CONDA_ENV_NAME"
run_in_env python -m pip install --upgrade pip setuptools wheel

echo "Installing PyTorch $TORCH_VERSION / torchvision $TORCHVISION_VERSION ($CUDA_WHEEL)"
run_in_env python -m pip install \
  --index-url "https://download.pytorch.org/whl/${CUDA_WHEEL}" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}"

echo "Installing TextVQA/LLaVA PA-Attack runtime dependencies"
run_in_env python -m pip install \
  "accelerate==0.24.0" \
  "einops==0.6.1" \
  "einops-exts==0.0.4" \
  "ftfy==6.1.1" \
  "huggingface_hub>=0.36.2,<1.0" \
  "numpy==1.24.2" \
  "open-clip-torch==2.19.0" \
  "Pillow==9.5.0" \
  "protobuf==3.20.3" \
  "regex==2023.5.5" \
  "requests==2.25.1" \
  "safetensors" \
  "scikit-learn==1.3.2" \
  "scipy==1.10.1" \
  "sentencepiece==0.1.98" \
  "shortuuid==1.0.11" \
  "timm==0.6.13" \
  "tokenizers==0.13.3" \
  "tqdm==4.65.0" \
  "$TRANSFORMERS_SPEC"

if [[ "$INSTALL_FULL_REQUIREMENTS" == "1" ]]; then
  echo "Installing original PA-Attack requirements.txt as requested"
  run_in_env python -m pip install -r requirements.txt
fi

echo "Checking key imports"
PYTHONPATH="$PA_ROOT:${PYTHONPATH:-}" conda run -n "$CONDA_ENV_NAME" python - <<'PY'
import open_clip
import torch
import transformers
from llava.model.builder import load_pretrained_model
from vlm_eval.attacks.veattack import pgd_veattack

print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("open_clip import ok")
print("llava/veattack import ok")
PY

cat <<EOF

Conda environment is ready.

Use it on the server with:
  conda activate $CONDA_ENV_NAME
  export PYTHON_BIN=python

Then download assets and run PA-Attack from this repository root.
EOF
