# BACKAI 서버 경로 기준 실행 메모

## GitHub
```bash
https://github.com/BACKAI/PA-Attack_TextVQA
```

## 서버의 기존 경로
```bash
VLM_ROOT=/var/tmp/jnuadmin_vlm/VLM
TEXTVQA_ROOT=/var/tmp/jnuadmin_vlm/VLM/dataset/textvqa
OUTPUT_ROOT=/var/tmp/jnuadmin_vlm/VLM/outputs/textvqa_paattack_llava_4gpu
PA_REPO=/var/tmp/jnuadmin_vlm/VLM/Attack/PA-Attack_TextVQA
EXPORT_ROOT=/var/tmp/jnuadmin_vlm/VLM/exports
```

## clone
```bash
cd /var/tmp/jnuadmin_vlm/VLM/Attack
git clone https://github.com/BACKAI/PA-Attack_TextVQA.git
cd PA-Attack_TextVQA
```

## PA 전용 conda 환경
기존 `vqattack-textvqa` 환경은 VQAttack용으로 유지하고, PA-Attack은 새 환경에서 실행하는 것을 권장한다.

```bash
cd /var/tmp/jnuadmin_vlm/VLM/Attack/PA-Attack_TextVQA

export CONDA_ENV_NAME=pa
export PYTHON_VERSION=3.10
bash pa_code/setup_pa_conda_env.sh

conda activate pa
export PYTHON_BIN=python
```

이 스크립트는 `torch==2.0.1`, `torchvision==0.15.2`, `open-clip-torch`, `einops-exts`, `transformers==4.31.0`, `huggingface_hub<1.0` 등 TextVQA/LLaVA PA-Attack에 필요한 최소 환경을 만든다.

## 모델과 prototype
LLaVA-1.5-7B는 Hugging Face에서 받을 수 있다.

이미 `huggingface_hub 1.x`로 올라가서 `transformers 4.26.1 requires huggingface-hub<1.0` 경고가 나온 경우에는 먼저 아래처럼 되돌린다.

```bash
python -m pip install "huggingface_hub>=0.36.2,<1.0"
```

수정된 `download_server_assets.sh`도 실행 시 이 호환 범위를 확인하고 필요하면 자동으로 되돌린다.

```bash
export PYTHON_BIN=python
export HF_MODEL_ID=liuhaotian/llava-v1.5-7b
export MODEL_DIR=/var/tmp/jnuadmin_vlm/VLM/Attack/PA-Attack_TextVQA/models/llava-v1.5-7b
bash pa_code/download_server_assets.sh
```

PA-Attack 원본 GitHub에는 prototype `.pt` 공개 release asset이 없다. 이미 별도 저장소나 private URL에 올려둔 경우:

```bash
export PROTOTYPE_URL="https://your-private-url/prototypes_tokens_3000_20_1024.pt"
export PROTOTYPE_PATH=/var/tmp/jnuadmin_vlm/VLM/Attack/PA-Attack_TextVQA/prototypes_llava_tokens_bestpca/prototypes_tokens_3000_20_1024.pt
bash pa_code/download_server_assets.sh
```

위 예시의 `https://your-private-url/...`은 placeholder이므로 그대로 실행하면 안 된다. 실제 다운로드 가능한 URL이 없으면 아래 prototype 생성 절차를 사용한다.

URL이 없으면 서버에서 생성해야 한다.

```bash
CUDA_VISIBLE_DEVICES=0 bash bash/llava_prototype_generation.sh
```

주의: 원본 `prototype/prototype_pca.py`는 COCO val 경로를 `/home/datasets/coco2014/val2014`로 하드코딩한다. 현재 서버에는 MSCOCO가 `/var/tmp/jnuadmin_vlm/VLM/dataset/MSCOCO` 아래에 있으므로, 다음 중 하나가 필요하다.

```bash
# 방법 1: symlink
sudo mkdir -p /home/datasets/coco2014
sudo ln -s /var/tmp/jnuadmin_vlm/VLM/dataset/MSCOCO/val2014 /home/datasets/coco2014/val2014

# 방법 2: prototype/prototype_pca.py의 coco_path를 직접 수정
# coco_path = "/var/tmp/jnuadmin_vlm/VLM/dataset/MSCOCO/val2014"
```

권한 문제나 원본 파일 수정을 피하려면 `pa_code`의 서버용 generator를 사용한다.

```bash
cd /var/tmp/jnuadmin_vlm/VLM/Attack/PA-Attack_TextVQA
conda activate pa

export PYTHON_BIN=python
export COCO_VAL_IMAGE_DIR=/var/tmp/jnuadmin_vlm/VLM/dataset/MSCOCO/val2014
export MODEL_PATH=/var/tmp/jnuadmin_vlm/VLM/Attack/PA-Attack_TextVQA/models/llava-v1.5-7b
export PROTOTYPE_PATH=prototypes_llava_tokens_bestpca/prototypes_tokens_3000_20_1024.pt

CUDA_VISIBLE_DEVICES=0 bash pa_code/generate_llava_prototype_server.sh
```

## PA-Attack 실행
GPU 0~3이 비어 있으므로 기본값으로 사용한다. 기존 VQAttack 프로세스는 GPU 4~7을 쓰고 있으므로 충돌하지 않는다.

tmux에서 실행하려면 아래 wrapper를 권장한다. 환경변수와 conda activation을 tmux 세션 안에서 자동으로 설정한다.

```bash
cd /var/tmp/jnuadmin_vlm/VLM/Attack/PA-Attack_TextVQA

export SESSION_NAME=pa_textvqa_attack
export CONDA_ENV_NAME=pa
export GPUS="0 1 2 3"
export OUTPUT_ROOT=/var/tmp/jnuadmin_vlm/VLM/outputs/textvqa_paattack_llava_4gpu
export RUN_ID=server_run_001

bash pa_code/start_textvqa_paattack_tmux.sh
tmux attach -t pa_textvqa_attack
```

기존 세션이 남아 있으면 먼저 접속하거나 종료한다.

```bash
tmux attach -t pa_textvqa_attack
# 또는
tmux kill-session -t pa_textvqa_attack
```

수동 실행은 아래와 같다.

```bash
cd /var/tmp/jnuadmin_vlm/VLM/Attack/PA-Attack_TextVQA

export TEXTVQA_ROOT=/var/tmp/jnuadmin_vlm/VLM/dataset/textvqa
export MODEL_PATH=/var/tmp/jnuadmin_vlm/VLM/Attack/PA-Attack_TextVQA/models/llava-v1.5-7b
export PYTHON_BIN=python
export GPUS="0 1 2 3"
export OUTPUT_ROOT=/var/tmp/jnuadmin_vlm/VLM/outputs/textvqa_paattack_llava_4gpu
export RUN_ID=server_run_001

bash pa_code/run_textvqa_paattack_server.sh
```

## 생성되는 입력 범위
서버의 현재 TextVQA 구조 기준:

- train JSON: 34,602 questions
- train images: 21,953 files
- validation JSON: 5,000 questions
- validation images: 3,166 files

실행 스크립트는 다음을 생성한다.

- train: 원본 train JSON에서 처음 등장하는 unique image 15,000장과 그 이미지들의 모든 question
- validation: unique image 3,166장 전체와 모든 question
- shard: GPU 0~3에 맞춰 image 기준 4개 shard

## 출력 위치
```bash
/var/tmp/jnuadmin_vlm/VLM/outputs/textvqa_paattack_llava_4gpu
```

출력 구조:

```bash
textvqa_paattack_llava_4gpu/
  train/
    shard_00/
      adv_tensors_by_image/*.pt
      answers.jsonl
      manifest.json
    shard_01/
    shard_02/
    shard_03/
  validation/
    shard_00/
    shard_01/
    shard_02/
    shard_03/
  logs/
```

## 중단 후 재시작
같은 `OUTPUT_ROOT`와 `RUN_ID`로 다시 실행하면 `--skip-existing` 때문에 이미 저장된 image/question은 건너뛴다.

```bash
export OUTPUT_ROOT=/var/tmp/jnuadmin_vlm/VLM/outputs/textvqa_paattack_llava_4gpu
export RUN_ID=server_run_001
bash pa_code/run_textvqa_paattack_server.sh
```

## preflight만 확인
```bash
python pa_code/prepare_textvqa_server.py \
  --textvqa-root /var/tmp/jnuadmin_vlm/VLM/dataset/textvqa \
  --output-dir pa_code/generated/textvqa_server \
  --train-image-limit 15000 \
  --num-shards 4

python pa_code/preflight_textvqa_server.py \
  --generated-dir pa_code/generated/textvqa_server \
  --prototype prototypes_llava_tokens_bestpca/prototypes_tokens_3000_20_1024.pt \
  --python python
```
