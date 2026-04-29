# TextVQA PA-Attack 서버 실행 안내

## 목적
서버에서 LLaVA-1.5-7B pivot 모델로 TextVQA를 PA-Attack 방식으로 교란한다.

## 처리 범위
- train: 원본 train JSON에서 처음 등장하는 unique image 기준 앞에서부터 15,000장 선택, 선택된 이미지의 모든 question 포함.
- validation: validation unique image 3,166장 전체, 해당 이미지의 모든 question 포함.
- 현재 로컬 데이터 기준 생성 결과:
  - train: 15,000 images, 24,007 questions
  - validation: 3,166 images, 5,000 questions
- shard는 이미지 기준으로 4개로 나누며, 각 shard는 한 GPU 프로세스가 처리한다.

## 서버 준비물
- PA-Attack repository
- TextVQA 폴더
  - `$TEXTVQA_ROOT/original_format/TextVQA_0.5.1_train.json`
  - `$TEXTVQA_ROOT/original_format/TextVQA_0.5.1_val.json`
  - `$TEXTVQA_ROOT/original_format/train_images`
  - `$TEXTVQA_ROOT/original_format/validation_images`
- LLaVA-1.5-7B 접근 가능 모델 경로 또는 Hugging Face ID
- PA-Attack LLaVA prototype:
  - `prototypes_llava_tokens_bestpca/prototypes_tokens_3000_20_1024.pt`
- Python 환경 핵심 패키지:
  - `torch`, `torchvision`, `open-clip-torch`, `einops`, `einops-exts`, `transformers` LLaVA 호환 버전

## PA 전용 conda 환경
기존 `vqattack-textvqa` 환경은 다른 공격 코드에 맞춰져 있으므로 PA-Attack은 별도 환경을 권장한다.

```bash
cd /var/tmp/jnuadmin_vlm/VLM/Attack/PA-Attack_TextVQA

export CONDA_ENV_NAME=pa
export PYTHON_VERSION=3.10
bash pa_code/setup_pa_conda_env.sh

conda activate pa
export PYTHON_BIN=python
```

기본 스크립트는 TextVQA/LLaVA PA-Attack 실행에 필요한 최소 패키지를 설치한다. 원본 `requirements.txt` 전체 설치가 꼭 필요하면 `INSTALL_FULL_REQUIREMENTS=1`을 지정할 수 있지만, 현재 TextVQA 실행에는 권장하지 않는다.

## 모델/prototype 다운로드
LLaVA 모델은 Hugging Face에서 받을 수 있다.

이미 `huggingface_hub 1.x`가 설치되어 `transformers 4.26.1`과 충돌한 경우에는 먼저 호환 버전으로 되돌린다.

```bash
python -m pip install "huggingface_hub>=0.36.2,<1.0"
```

수정된 `download_server_assets.sh`도 실행 시 이 호환 범위를 확인하고 필요하면 자동으로 되돌린다.

```bash
export HF_MODEL_ID=liuhaotian/llava-v1.5-7b
export MODEL_DIR=models/llava-v1.5-7b
bash pa_code/download_server_assets.sh
```

PA-Attack 원본 GitHub에는 prototype `.pt` release asset이 없다. 이미 생성해 둔 prototype을 private URL로 올린 경우에는 다음처럼 받는다.

```bash
export PROTOTYPE_URL="https://your-private-url/prototypes_tokens_3000_20_1024.pt"
export PROTOTYPE_PATH=prototypes_llava_tokens_bestpca/prototypes_tokens_3000_20_1024.pt
bash pa_code/download_server_assets.sh
```

`https://your-private-url/...`은 실제 URL이 아니라 예시다. 실제 prototype URL이 없으면 아래처럼 서버에서 생성해야 한다.

prototype URL이 없으면 서버에서 원본 방식으로 생성한다.

```bash
CUDA_VISIBLE_DEVICES=0 bash bash/llava_prototype_generation.sh
```

주의: `prototype/prototype_pca.py`에는 COCO 경로 `/home/datasets/coco2014/val2014`가 하드코딩되어 있다. 서버 경로가 다르면 파일을 수정하거나 symlink를 만들어야 한다.

권한 문제나 원본 파일 수정을 피하려면 `pa_code`의 서버용 generator를 사용한다.

```bash
export COCO_VAL_IMAGE_DIR=/var/tmp/jnuadmin_vlm/VLM/dataset/MSCOCO/val2014
export MODEL_PATH=/var/tmp/jnuadmin_vlm/VLM/Attack/PA-Attack_TextVQA/models/llava-v1.5-7b
export PROTOTYPE_PATH=prototypes_llava_tokens_bestpca/prototypes_tokens_3000_20_1024.pt

CUDA_VISIBLE_DEVICES=0 bash pa_code/generate_llava_prototype_server.sh
```

## 실행
BACKAI 서버의 현재 경로 기준 명령은 `pa_code/SERVER_PATHS_BACKAI.md`에 따로 정리했다.

장시간 실행은 tmux wrapper를 권장한다.

```bash
export SESSION_NAME=pa_textvqa_attack
export CONDA_ENV_NAME=pa
export GPUS="0 1 2 3"
export OUTPUT_ROOT=/var/tmp/jnuadmin_vlm/VLM/outputs/textvqa_paattack_llava_4gpu
export RUN_ID=server_run_001

bash pa_code/start_textvqa_paattack_tmux.sh
tmux attach -t pa_textvqa_attack
```

```bash
cd PA-Attack

export TEXTVQA_ROOT=/path/to/textvqa
export MODEL_PATH=models/llava-v1.5-7b
export PYTHON_BIN=python
export GPUS="0 1 2 3"

bash pa_code/run_textvqa_paattack_server.sh
```

## 선택 옵션
```bash
export SPLITS="train validation"   # 기본값
export TRAIN_IMAGE_LIMIT=15000     # 기본값
export EPS=2                       # 기본값
export STEPS=100                   # 기본값
export STAGE1_STEPS=50             # 기본값
export MAX_NEW_TOKENS=10           # 기본값
export SAVE_PNG=0                  # 1이면 adv png도 저장
export RUN_ID=my_run_name          # 출력 폴더 이름 고정
```

## 출력
기본 출력 위치:
```bash
pa_code/outputs/textvqa_paattack_${RUN_ID}
```

각 split/shard 아래에 다음이 생성된다.
- `adv_tensors_by_image/{image_id}.pt`: 이미지별 adversarial tensor
- `answers.jsonl`: question별 LLaVA 답변과 연결된 adv tensor 경로
- `manifest.json`: 실행 설정과 처리 수량
- `logs/{split}_shard_XX.log`: shard별 로그

## 재시작
스크립트는 `--skip-existing`을 사용한다. 중간에 끊기면 같은 `RUN_ID`로 다시 실행하면 이미 저장된 image/question은 건너뛴다.

## 주의
기존 `vlm_eval.run_evaluation_paattack`는 non-ensemble `veattack`에서 adversarial tensor 저장이 비어 있는 문제가 있어, 이 서버 실행은 `pa_code/attack_textvqa_llava.py`를 사용한다. 이 파일은 PA-Attack의 CLIP prototype loss와 `vlm_eval.attacks.veattack.pgd_veattack`을 그대로 사용하되, TextVQA/LLaVA 전용으로 저장 로직을 명시했다.
