import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_TEXTVQA_ROOT = Path(r"D:\VLM\dataset\textvqa")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "generated" / "textvqa_train10"


def canonical_answer(answers):
    cleaned = [str(answer).strip().lower() for answer in answers if str(answer).strip()]
    if not cleaned:
        return None
    return Counter(cleaned).most_common(1)[0][0]


def answer_objects(answers):
    cleaned = [str(answer).strip().lower() for answer in answers if str(answer).strip()]
    if not cleaned:
        cleaned = [""]
    return [{"answer_id": index + 1, "answer": answer} for index, answer in enumerate(cleaned)]


def select_unique_image_records(records, limit):
    selected = []
    seen_images = set()
    for record in records:
        image_id = str(record.get("image_id", "")).strip()
        question = str(record.get("question", "")).strip()
        answers = record.get("answers", [])
        if not image_id or not question or canonical_answer(answers) is None:
            continue
        if image_id in seen_images:
            continue
        selected.append(record)
        seen_images.add(image_id)
        if len(selected) == limit:
            break
    if len(selected) < limit:
        raise RuntimeError(f"Only found {len(selected)} usable unique-image records, requested {limit}.")
    return selected


def build_vqa_payloads(records):
    questions = []
    annotations = []
    samples = []
    for record in records:
        image_id = str(record["image_id"])
        question_id = int(record["question_id"])
        question = str(record["question"])
        image_path = str(record.get("image_path") or f"train_images/{image_id}.jpg")
        answers = record.get("answers", [])
        questions.append(
            {
                "question_id": question_id,
                "image_id": image_id,
                "question": question,
            }
        )
        annotations.append(
            {
                "question_id": question_id,
                "image_id": image_id,
                "question_type": "textvqa",
                "answer_type": "other",
                "answers": answer_objects(answers),
            }
        )
        samples.append(
            {
                "question_id": question_id,
                "image_id": image_id,
                "image_file": f"{image_id}.jpg",
                "source_image_path": image_path,
                "question": question,
                "canonical_answer": canonical_answer(answers),
            }
        )

    info = {
        "description": "TextVQA train subset converted for PA-Attack smoke execution",
        "version": "0.5.1",
        "split": "train",
    }
    question_payload = {
        "info": info,
        "task_type": "Open-Ended",
        "data_type": "textvqa",
        "data_subtype": "train10",
        "questions": questions,
    }
    annotation_payload = {
        "info": info,
        "annotations": annotations,
    }
    return question_payload, annotation_payload, samples


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a 10-image TextVQA train subset for PA-Attack.")
    parser.add_argument("--textvqa-root", type=Path, default=DEFAULT_TEXTVQA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    original_dir = args.textvqa_root / "original_format"
    train_json = original_dir / "TextVQA_0.5.1_train.json"
    train_image_dir = original_dir / "train_images"

    payload = json.loads(train_json.read_text(encoding="utf-8"))
    records = select_unique_image_records(payload["data"], args.limit)
    questions, annotations, samples = build_vqa_payloads(records)

    missing_images = [
        sample["image_file"]
        for sample in samples
        if not (train_image_dir / sample["image_file"]).exists()
    ]
    if missing_images:
        raise FileNotFoundError(f"Missing image files in {train_image_dir}: {missing_images}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    questions_path = args.output_dir / "textvqa_train10_questions_vqa_format.json"
    annotations_path = args.output_dir / "textvqa_train10_annotations_vqa_format.json"
    manifest_path = args.output_dir / "manifest.json"

    write_json(questions_path, questions)
    write_json(annotations_path, annotations)
    write_json(
        manifest_path,
        {
            "textvqa_root": str(args.textvqa_root),
            "image_dir": str(train_image_dir),
            "questions_json": str(questions_path),
            "annotations_json": str(annotations_path),
            "record_count": len(samples),
            "samples": samples,
            "notes": [
                "This subset contains one question per unique train image.",
                "Pass image_dir as --textvqa_image_dir_path when running PA-Attack.",
                "The same subset is used as train and test input for zero-shot LLaVA smoke execution.",
            ],
        },
    )
    print(json.dumps(json.loads(manifest_path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
