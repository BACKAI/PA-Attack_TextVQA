#!/usr/bin/env bash
set -euo pipefail

PA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PA_ROOT"
export PYTHONDONTWRITEBYTECODE=1

TEXTVQA_ROOT="${TEXTVQA_ROOT:-/path/to/textvqa}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" pa_code/prepare_textvqa10.py \
  --textvqa-root "$TEXTVQA_ROOT"

"$PYTHON_BIN" pa_code/preflight_textvqa10.py \
  --strict \
  --python "$PYTHON_BIN" \
  --model-path "$MODEL_PATH"

GENERATED_DIR="pa_code/generated/textvqa_train10"
TEXTVQA_IMAGE_DIR="$TEXTVQA_ROOT/original_format/train_images"
TEXTVQA_QUESTIONS="$GENERATED_DIR/textvqa_train10_questions_vqa_format.json"
TEXTVQA_ANNOTATIONS="$GENERATED_DIR/textvqa_train10_annotations_vqa_format.json"

"$PYTHON_BIN" -m vlm_eval.run_evaluation_paattack \
  --eval_textvqa \
  --attack veattack \
  --eps 2 \
  --steps 100 \
  --mask_out none \
  --precision float16 \
  --num_samples 10 \
  --query_set_size 1 \
  --shots 0 \
  --batch_size 1 \
  --results_file llava_textvqa_train10 \
  --model llava \
  --temperature 0.0 \
  --num_beams 1 \
  --out_base_path pa_code/outputs/textvqa_train10_paattack \
  --model_path "$MODEL_PATH" \
  --vision_encoder_pretrained openai \
  --textvqa_image_dir_path "$TEXTVQA_IMAGE_DIR" \
  --textvqa_train_questions_json_path "$TEXTVQA_QUESTIONS" \
  --textvqa_train_annotations_json_path "$TEXTVQA_ANNOTATIONS" \
  --textvqa_test_questions_json_path "$TEXTVQA_QUESTIONS" \
  --textvqa_test_annotations_json_path "$TEXTVQA_ANNOTATIONS"
