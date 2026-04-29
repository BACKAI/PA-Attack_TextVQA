#!/usr/bin/env bash
set -euo pipefail

PA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PA_ROOT"

export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export PYTHONPATH="$PA_ROOT:${PYTHONPATH:-}"

TEXTVQA_ROOT="${TEXTVQA_ROOT:?Set TEXTVQA_ROOT to the server TextVQA root directory}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS="${GPUS:-0 1 2 3}"
SPLITS="${SPLITS:-train validation}"
TRAIN_IMAGE_LIMIT="${TRAIN_IMAGE_LIMIT:-15000}"
EPS="${EPS:-2}"
STEPS="${STEPS:-100}"
STAGE1_STEPS="${STAGE1_STEPS:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10}"
SAVE_PNG="${SAVE_PNG:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
GENERATED_DIR="${GENERATED_DIR:-pa_code/generated/textvqa_server}"
OUTPUT_ROOT="${OUTPUT_ROOT:-pa_code/outputs/textvqa_paattack_${RUN_ID}}"
PROTOTYPE="${PROTOTYPE:-prototypes_llava_tokens_bestpca/prototypes_tokens_3000_20_1024.pt}"

read -r -a GPU_ARRAY <<< "$GPUS"
NUM_SHARDS="${NUM_SHARDS:-${#GPU_ARRAY[@]}}"

if [[ "$NUM_SHARDS" -lt 1 ]]; then
  echo "NUM_SHARDS must be >= 1" >&2
  exit 1
fi

"$PYTHON_BIN" pa_code/prepare_textvqa_server.py \
  --textvqa-root "$TEXTVQA_ROOT" \
  --output-dir "$GENERATED_DIR" \
  --train-image-limit "$TRAIN_IMAGE_LIMIT" \
  --num-shards "$NUM_SHARDS"

"$PYTHON_BIN" pa_code/preflight_textvqa_server.py \
  --generated-dir "$GENERATED_DIR" \
  --prototype "$PROTOTYPE" \
  --python "$PYTHON_BIN"

mkdir -p "$OUTPUT_ROOT/logs"

run_shard() {
  local split="$1"
  local shard_index="$2"
  local gpu="$3"
  local shard
  shard="$(printf 'shard_%02d' "$shard_index")"
  local shard_dir="$GENERATED_DIR/$split/shards/$shard"
  local output_dir="$OUTPUT_ROOT/$split/$shard"
  local image_folder
  if [[ "$split" == "train" ]]; then
    image_folder="$TEXTVQA_ROOT/original_format/train_images"
  elif [[ "$split" == "validation" ]]; then
    image_folder="$TEXTVQA_ROOT/original_format/validation_images"
  else
    echo "Unknown split: $split" >&2
    exit 1
  fi

  local save_png_args=()
  if [[ "$SAVE_PNG" == "1" ]]; then
    save_png_args=(--save-png)
  fi

  echo "Launching split=$split shard=$shard gpu=$gpu output=$output_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" pa_code/attack_textvqa_llava.py \
    --model-path "$MODEL_PATH" \
    --question-file "$shard_dir/textvqa_${split}_${shard}_questions.json" \
    --image-folder "$image_folder" \
    --output-dir "$output_dir" \
    --prototype "$PROTOTYPE" \
    --eps "$EPS" \
    --steps "$STEPS" \
    --stage1-steps "$STAGE1_STEPS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --skip-existing \
    "${save_png_args[@]}" \
    > "$OUTPUT_ROOT/logs/${split}_${shard}.log" 2>&1 &
}

for split in $SPLITS; do
  echo "Starting split=$split with $NUM_SHARDS shards on GPUs: $GPUS"
  pids=()
  for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu="${GPU_ARRAY[$((shard_index % ${#GPU_ARRAY[@]}))]}"
    run_shard "$split" "$shard_index" "$gpu"
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    echo "One or more $split shards failed. Recent shard logs:" >&2
    for log_file in "$OUTPUT_ROOT"/logs/"${split}"_shard_*.log; do
      echo "===== $log_file =====" >&2
      tail -n 80 "$log_file" >&2 || true
    done
    exit 1
  fi
  echo "Finished split=$split"
done

echo "All requested splits finished. Outputs: $OUTPUT_ROOT"
