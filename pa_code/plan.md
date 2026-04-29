# PA-Attack TextVQA 서버 실행 계획

## 목표
`D:\VLM\dataset\textvqa` 기준 TextVQA train 15,000 image/all questions와 validation 전체 image/all questions를 서버에서 LLaVA-1.5-7B pivot PA-Attack으로 교란할 수 있게 만든다.

## 범위
- 원본 PA-Attack 코드는 수정하지 않는다.
- 새 코드와 실행 보조 파일은 `D:\VLM\Attack\PA-Attack\pa_code` 아래에만 둔다.
- 로컬에서는 데이터 선택, shard 생성, 문법, 실행 전제 조건을 확인한다.
- 실제 LLaVA-1.5-7B PA-Attack 실행은 서버 GPU 0~3에서 수행할 수 있도록 명령을 고정한다.

## 가정
- 서버에는 PA-Attack 의존성, CUDA, LLaVA-1.5-7B 모델 접근 권한이 준비될 수 있다.
- 서버에도 TextVQA 원본 train/validation JSON과 image 폴더가 있거나, 같은 구조로 경로를 지정할 수 있다.
- PA-Attack의 LLaVA 공격에는 `prototypes_llava_tokens_bestpca/prototypes_tokens_3000_20_1024.pt`가 필요하다.
- 기존 VQAttack 환경과 충돌하면 PA-Attack 전용 conda 환경 `pa`를 새로 만든다.

## 마일스톤
1. TextVQA 원본 train/validation JSON을 읽어 이미지 기준 대상 범위를 생성한다.
2. 선택된 이미지에 연결된 모든 question을 포함한 VQA-style JSON을 쓴다.
3. GPU 0~3용 image-based shard를 만든다.
4. PA-Attack prototype loss를 사용하는 LLaVA/TextVQA 전용 공격 실행기를 작성한다.
5. 서버용 shell 실행 스크립트와 preflight를 제공한다.
6. PA-Attack 전용 conda 환경 생성 스크립트를 제공한다.

## 완료 기준
- train 15,000 image와 validation 전체 image 기준 shard manifest가 생성된다.
- shard question 합산이 full split question 수와 일치한다.
- 서버에서 실행할 단일 shell 스크립트가 준비된다.
- 서버에서 `conda create -n pa` 기반 전용 환경을 만들 수 있다.
- 로컬에서 막히는 항목은 prototype/env 문제로 분리해 보고된다.

## 검증
- 로컬: `prepare_textvqa_server.py` 실행, Python/bash 문법 검사, `preflight_textvqa_server.py` 실행.
- 서버: `bash pa_code/run_textvqa_paattack_server.sh` 실행.
