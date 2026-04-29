# PA-Attack TextVQA 서버 실행 핸드오프

## 변경 사항
- `pa_code/prepare_textvqa_server.py`
  - train은 unique image 기준 앞에서부터 15,000장, validation은 전체 image를 선택한다.
  - 선택된 이미지에 연결된 모든 question을 포함한다.
  - split별 full JSON과 4개 image-based shard JSON을 생성한다.
- `pa_code/attack_textvqa_llava.py`
  - LLaVA-1.5-7B와 PA-Attack prototype loss로 TextVQA shard를 공격한다.
  - 이미지별 adversarial tensor를 저장하고, question별 답변 로그를 `answers.jsonl`에 쓴다.
- `pa_code/run_textvqa_paattack_server.sh`
  - GPU 0~3을 기본 사용해 train shard 4개 실행 후 validation shard 4개를 실행한다.
  - `--skip-existing`으로 재시작 가능하다.
- `pa_code/preflight_textvqa_server.py`
  - shard 수량, prototype 존재, Python import 상태를 점검한다.
- `pa_code/setup_pa_conda_env.sh`
  - 기존 VQAttack 환경과 분리된 `pa` conda 환경을 만들고 TextVQA/LLaVA PA-Attack 최소 의존성을 설치한다.
- `pa_code/README_server.md`
  - 서버 실행 방법과 출력 구조를 정리했다.
- `pa_code/prepare_textvqa10.py`
  - `D:\VLM\dataset\textvqa\original_format\TextVQA_0.5.1_train.json`에서 10개 unique image/question 샘플을 선택한다.
  - PA-Attack `VQADataset`이 기대하는 VQA-style `questions`/`annotations` JSON을 생성한다.
- `pa_code/preflight_textvqa10.py`
  - 생성 JSON, image file, record count, `question_id` 정렬, prototype 존재 여부, Python import 상태를 확인한다.
  - 서버에서 실행할 PA-Attack 명령을 출력한다.
- `pa_code/run_paattack_textvqa10.sh`
  - 서버에서 `TEXTVQA_ROOT`, `MODEL_PATH`, `PYTHON_BIN`만 지정해 10장 공격을 실행하는 래퍼다.
- 생성 산출물:
  - `pa_code/generated/textvqa_train10/textvqa_train10_questions_vqa_format.json`
  - `pa_code/generated/textvqa_train10/textvqa_train10_annotations_vqa_format.json`
  - `pa_code/generated/textvqa_train10/manifest.json`

## 로컬에서 통과한 것
- TextVQA train 원본에서 10개 unique image/question 샘플 생성 성공.
- 10개 이미지 파일 존재 확인 성공.
- `questions`/`annotations` 각각 10개 record 확인 성공.
- `question_id` 순서 일치 확인 성공.
- preflight 데이터 계약 확인 성공.
- 서버용 shard 생성 성공.
  - train: 15,000 images, 24,007 questions
  - validation: 3,166 images, 5,000 questions
  - 4개 shard 합산이 full split 수량과 일치.
- 서버용 Python 스크립트 문법 검사 성공.
- 서버용 bash 스크립트 문법 검사 성공.
- PA 전용 conda 환경 생성 스크립트 bash 문법 검사 성공.

## 로컬에서 막힌 것
- 실제 `python -m vlm_eval.run_evaluation_paattack ...` 진입점은 로컬 conda 환경 의존성 부족으로 중단됨.
- 확인된 첫 실패:
  - `ModuleNotFoundError: No module named 'einops_exts'`
- 추가로 로컬 환경 전반에서 확인된 문제:
  - `open_clip` 미설치.
  - `llavaa`, `my` 환경은 `transformers`가 너무 새로워 로컬 LLaVA 등록과 충돌:
    - `ValueError: 'llava' is already used by a Transformers config, pick another name.`
- PA-Attack이 요구하는 prototype 파일 없음:
  - `prototypes_llava_tokens_bestpca/prototypes_tokens_3000_20_1024.pt`
- 서버용 preflight도 로컬에서는 다음 항목 때문에 실패한다.
  - prototype 없음.
  - `open_clip` 없음.

## 서버 실행 전제
- 권장: `pa_code/setup_pa_conda_env.sh`로 PA-Attack 전용 `pa` conda 환경 생성.
- `open-clip-torch`, `einops`, `einops-exts`, `torch`, `torchvision`, LLaVA 호환 `transformers` 버전 필요.
- LLaVA-1.5-7B는 PA-Attack 코드 기준 `liuhaotian/llava-v1.5-7b` 사용.
- 다음 prototype 파일 필요:
  - `prototypes_llava_tokens_bestpca/prototypes_tokens_3000_20_1024.pt`
- 서버의 TextVQA 경로를 `TEXTVQA_ROOT`로 지정해야 한다.

## 서버 다운로드 이슈 대응
- `download_server_assets.sh`가 `huggingface_hub`를 1.x로 올리면 `transformers 4.26.1`과 충돌한다.
- 스크립트는 실행 시 `huggingface_hub>=0.36.2,<1.0` 범위를 확인하고, 사용 가능한 경우 `hf download`를 먼저 쓰도록 수정했다.
- 서버에서 이미 1.x로 올라간 경우:
```bash
python -m pip install "huggingface_hub>=0.36.2,<1.0"
```
- `PROTOTYPE_URL="https://your-private-url/..."`은 placeholder이므로 실제 URL이 없으면 다운로드가 아니라 prototype 생성 절차를 사용해야 한다.

## 서버 실행 예시
```bash
cd PA-Attack

export TEXTVQA_ROOT=/path/to/textvqa
export MODEL_PATH=models/llava-v1.5-7b
export PYTHON_BIN=python
export GPUS="0 1 2 3"

bash pa_code/run_textvqa_paattack_server.sh
```

## 결론
- TextVQA 대규모 대상 선택과 4-GPU shard 생성 로직은 로컬에서 확인됐다.
- 기존 PA-Attack runner는 non-ensemble `veattack` 저장 로직이 비어 있어, 서버용으로 `pa_code/attack_textvqa_llava.py`를 사용한다.
- 서버에서 의존성과 prototype을 맞추면 `pa_code/run_textvqa_paattack_server.sh`로 train 15,000 image/all questions와 validation 전체를 실행할 수 있다.
