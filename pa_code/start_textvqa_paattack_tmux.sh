#!/usr/bin/env bash
set -euo pipefail

PA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PA_ROOT"

SESSION_NAME="${SESSION_NAME:-pa_textvqa_attack}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pa}"
PYTHON_BIN="${PYTHON_BIN:-/home/jnuadmin/miniconda3/envs/${CONDA_ENV_NAME}/bin/python}"
TEXTVQA_ROOT="${TEXTVQA_ROOT:-/var/tmp/jnuadmin_vlm/VLM/dataset/textvqa}"
MODEL_PATH="${MODEL_PATH:-/var/tmp/jnuadmin_vlm/VLM/Attack/PA-Attack_TextVQA/models/llava-v1.5-7b}"
GPUS="${GPUS:-0 1 2 3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/var/tmp/jnuadmin_vlm/VLM/outputs/textvqa_paattack_llava_4gpu}"
RUN_ID="${RUN_ID:-server_run_001}"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  echo "Attach with: tmux attach -t $SESSION_NAME"
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" \
  "cd '$PA_ROOT' && \
   source \"\$(conda info --base)/etc/profile.d/conda.sh\" && \
   conda activate '$CONDA_ENV_NAME' && \
   export PYTHONPATH='$PA_ROOT':\"\${PYTHONPATH:-}\" && \
   export PYTHON_BIN='$PYTHON_BIN' && \
   export TEXTVQA_ROOT='$TEXTVQA_ROOT' && \
   export MODEL_PATH='$MODEL_PATH' && \
   export GPUS='$GPUS' && \
   export OUTPUT_ROOT='$OUTPUT_ROOT' && \
   export RUN_ID='$RUN_ID' && \
   bash pa_code/run_textvqa_paattack_server.sh"

cat <<EOF
Started tmux session: $SESSION_NAME

Attach:
  tmux attach -t $SESSION_NAME

Detach inside tmux:
  Ctrl-b then d

Logs:
  tail -f $OUTPUT_ROOT/logs/train_shard_00.log
EOF
