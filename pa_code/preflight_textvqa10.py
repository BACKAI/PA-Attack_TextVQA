import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENERATED_DIR = Path(__file__).resolve().parent / "generated" / "textvqa_train10"
DEFAULT_PROTO = PA_ROOT / "prototypes_llava_tokens_bestpca_cls" / "prototypes_tokens_3000_20_1024.pt"


def ok(message):
    print(f"[OK] {message}")


def warn(message):
    print(f"[WARN] {message}")


def fail(message, failures):
    print(f"[FAIL] {message}")
    failures.append(message)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_dataset(generated_dir, failures):
    manifest_path = generated_dir / "manifest.json"
    if not manifest_path.exists():
        fail(f"manifest not found: {manifest_path}", failures)
        return

    manifest = load_json(manifest_path)
    questions_path = Path(manifest["questions_json"])
    annotations_path = Path(manifest["annotations_json"])
    image_dir = Path(manifest["image_dir"])

    for path, label in [
        (questions_path, "questions json"),
        (annotations_path, "annotations json"),
        (image_dir, "image dir"),
    ]:
        if path.exists():
            ok(f"{label}: {path}")
        else:
            fail(f"{label} missing: {path}", failures)

    if failures:
        return

    questions_payload = load_json(questions_path)
    annotations_payload = load_json(annotations_path)
    questions = questions_payload.get("questions", [])
    annotations = annotations_payload.get("annotations", [])

    if len(questions) == 10 and len(annotations) == 10:
        ok("questions/annotations each contain 10 records")
    else:
        fail(f"unexpected record counts: questions={len(questions)}, annotations={len(annotations)}", failures)

    qids = [item.get("question_id") for item in questions]
    ann_qids = [item.get("question_id") for item in annotations]
    if qids == ann_qids:
        ok("question_id order matches annotations")
    else:
        fail("question_id order mismatch between questions and annotations", failures)

    missing_images = []
    for item in questions:
        image_path = image_dir / f"{item['image_id']}.jpg"
        if not image_path.exists():
            missing_images.append(str(image_path))
    if missing_images:
        fail(f"missing image files: {missing_images}", failures)
    else:
        ok("all 10 image files are present")


def resolve_python(python_exe):
    if not python_exe:
        return None
    python_text = str(python_exe)
    python_path = Path(python_text)
    if python_path.exists():
        return str(python_path)
    return shutil.which(python_text)


def check_python_module(python_exe, module_name, failures, required=False):
    resolved_python = resolve_python(python_exe)
    if not resolved_python:
        warn(f"skip import check for {module_name}: no --python provided")
        return
    cmd = [resolved_python, "-c", f"import {module_name}; print('ok')"]
    result = subprocess.run(cmd, cwd=str(PA_ROOT), text=True, capture_output=True)
    if result.returncode == 0:
        ok(f"python import works: {module_name}")
    else:
        message = f"python import failed: {module_name}: {result.stderr.strip() or result.stdout.strip()}"
        if required:
            fail(message, failures)
        else:
            warn(message)


def check_pa_dataset(python_exe, generated_dir, failures, required=False):
    resolved_python = resolve_python(python_exe)
    if not resolved_python:
        warn("skip PA VQADataset check: no usable --python provided")
        return
    manifest = load_json(generated_dir / "manifest.json")
    code = (
        "from open_flamingo.eval.eval_datasets import VQADataset; "
        "import sys; "
        "ds=VQADataset(sys.argv[1], sys.argv[2], sys.argv[3], False, 'textvqa'); "
        "item=ds[0]; "
        "print(len(ds), item['question_id'], item['question'])"
    )
    cmd = [
        resolved_python,
        "-c",
        code,
        manifest["image_dir"],
        manifest["questions_json"],
        manifest["annotations_json"],
    ]
    result = subprocess.run(cmd, cwd=str(PA_ROOT), text=True, capture_output=True)
    if result.returncode == 0:
        ok(f"PA VQADataset can load generated subset: {result.stdout.strip()}")
    else:
        message = f"PA VQADataset check failed: {result.stderr.strip() or result.stdout.strip()}"
        if required:
            fail(message, failures)
        else:
            warn(message)


def build_attack_command(generated_dir, proto_path, model_path):
    manifest = load_json(generated_dir / "manifest.json")
    questions = manifest["questions_json"]
    annotations = manifest["annotations_json"]
    image_dir = manifest["image_dir"]
    out_dir = str(Path("pa_code") / "outputs" / "textvqa_train10_paattack")
    return [
        "python",
        "-m",
        "vlm_eval.run_evaluation_paattack",
        "--eval_textvqa",
        "--attack",
        "veattack",
        "--eps",
        "2",
        "--steps",
        "100",
        "--mask_out",
        "none",
        "--precision",
        "float16",
        "--num_samples",
        "10",
        "--query_set_size",
        "1",
        "--shots",
        "0",
        "--batch_size",
        "1",
        "--results_file",
        "llava_textvqa_train10",
        "--model",
        "llava",
        "--temperature",
        "0.0",
        "--num_beams",
        "1",
        "--out_base_path",
        out_dir,
        "--model_path",
        model_path,
        "--vision_encoder_pretrained",
        "openai",
        "--textvqa_image_dir_path",
        image_dir,
        "--textvqa_train_questions_json_path",
        questions,
        "--textvqa_train_annotations_json_path",
        annotations,
        "--textvqa_test_questions_json_path",
        questions,
        "--textvqa_test_annotations_json_path",
        annotations,
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Preflight PA-Attack TextVQA train10 inputs.")
    parser.add_argument("--generated-dir", type=Path, default=DEFAULT_GENERATED_DIR)
    parser.add_argument("--prototype", type=Path, default=DEFAULT_PROTO)
    parser.add_argument("--python", default=None)
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    failures = []
    validate_dataset(args.generated_dir, failures)

    if args.prototype.exists():
        ok(f"prototype found: {args.prototype}")
    else:
        message = f"prototype missing: {args.prototype}"
        if args.strict:
            fail(message, failures)
        else:
            warn(message)

    if args.python:
        resolved_python = resolve_python(args.python)
        if resolved_python:
            ok(f"python executable found: {resolved_python}")
        else:
            fail(f"python executable missing: {args.python}", failures)
        check_python_module(args.python, "torch", failures, required=args.strict)
        check_python_module(args.python, "open_clip", failures, required=args.strict)
        check_python_module(args.python, "llava", failures, required=args.strict)
        check_pa_dataset(args.python, args.generated_dir, failures, required=args.strict)

    command = build_attack_command(args.generated_dir, args.prototype, args.model_path)
    print("\n[SERVER COMMAND]")
    print(" ".join(f'"{part}"' if any(ch in part for ch in " \\") else part for part in command))

    if failures:
        print("\nPreflight failed.")
        for item in failures:
            print(f"- {item}")
        return 1

    print("\nPreflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
