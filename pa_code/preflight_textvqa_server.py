import argparse
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path


PA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENERATED_DIR = Path(__file__).resolve().parent / "generated" / "textvqa_server"
DEFAULT_PROTO = PA_ROOT / "prototypes_llava_tokens_bestpca" / "prototypes_tokens_3000_20_1024.pt"


def status(label, ok, detail=""):
    prefix = "OK" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"[{prefix}] {label}{suffix}")


def resolve_python(value):
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return str(path)
    return shutil.which(value)


def check_imports(python_bin):
    resolved = resolve_python(python_bin)
    if not resolved:
        status("python", False, python_bin)
        return False
    code = (
        "mods=['torch','open_clip','llava','vlm_eval.attacks.veattack']; "
        "import importlib.util, sys; "
        "missing=[m for m in mods if importlib.util.find_spec(m) is None]; "
        "print('missing=' + ','.join(missing)); "
        "raise SystemExit(1 if missing else 0)"
    )
    result = subprocess.run([resolved, "-c", code], cwd=str(PA_ROOT), text=True, capture_output=True)
    ok = result.returncode == 0
    status("python imports", ok, (result.stdout + result.stderr).strip())
    return ok


def check_generated(generated_dir):
    manifest_path = generated_dir / "manifest.json"
    if not manifest_path.exists():
        status("generated manifest", False, str(manifest_path))
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ok = True
    for split, expected_images in [("train", 15000), ("validation", None)]:
        split_manifest = manifest["splits"].get(split)
        if not split_manifest:
            status(f"{split} manifest", False, "missing")
            ok = False
            continue
        image_count = split_manifest["image_count"]
        question_count = split_manifest["question_count"]
        expected_ok = image_count == expected_images if expected_images is not None else image_count > 0
        status(f"{split} image/question count", expected_ok, f"images={image_count}, questions={question_count}")
        ok = ok and expected_ok
        shard_question_sum = sum(item["question_count"] for item in split_manifest["shards"])
        shard_image_sum = sum(item["image_count"] for item in split_manifest["shards"])
        shard_ok = shard_question_sum == question_count and shard_image_sum == image_count
        status(f"{split} shard sums", shard_ok, f"images={shard_image_sum}, questions={shard_question_sum}")
        ok = ok and shard_ok
    return ok


def parse_args():
    parser = argparse.ArgumentParser(description="Preflight TextVQA PA-Attack server setup.")
    parser.add_argument("--generated-dir", type=Path, default=DEFAULT_GENERATED_DIR)
    parser.add_argument("--prototype", type=Path, default=DEFAULT_PROTO)
    parser.add_argument("--python", default="python")
    return parser.parse_args()


def main():
    args = parse_args()
    ok = True
    ok = check_generated(args.generated_dir) and ok
    proto_ok = args.prototype.exists()
    status("prototype", proto_ok, str(args.prototype))
    ok = proto_ok and ok
    ok = check_imports(args.python) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
