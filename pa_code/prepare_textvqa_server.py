import argparse
import json
from collections import Counter, OrderedDict
from math import ceil
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "generated" / "textvqa_server"


SPLIT_CONFIG = {
    "train": {
        "json_name": "TextVQA_0.5.1_train.json",
        "image_dir_name": "train_images",
        "image_limit": 15000,
    },
    "validation": {
        "json_name": "TextVQA_0.5.1_val.json",
        "image_dir_name": "validation_images",
        "image_limit": None,
    },
}


def canonical_answer(answers):
    cleaned = [str(answer).strip().lower() for answer in answers if str(answer).strip()]
    if not cleaned:
        return ""
    return Counter(cleaned).most_common(1)[0][0]


def answer_objects(answers):
    cleaned = [str(answer).strip().lower() for answer in answers if str(answer).strip()]
    if not cleaned:
        cleaned = [""]
    return [{"answer_id": index + 1, "answer": answer} for index, answer in enumerate(cleaned)]


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def group_by_image(records):
    grouped = OrderedDict()
    for record in records:
        image_id = str(record.get("image_id", "")).strip()
        question = str(record.get("question", "")).strip()
        if not image_id or not question:
            continue
        grouped.setdefault(image_id, []).append(record)
    return grouped


def select_records_by_image(records, image_limit):
    grouped = group_by_image(records)
    selected_image_ids = list(grouped.keys())
    if image_limit is not None:
        selected_image_ids = selected_image_ids[:image_limit]
    selected_records = []
    for image_id in selected_image_ids:
        selected_records.extend(grouped[image_id])
    return selected_image_ids, selected_records


def build_vqa_payloads(records, split_name):
    questions = []
    annotations = []
    samples = []
    for record in records:
        image_id = str(record["image_id"])
        question_id = int(record["question_id"])
        question = str(record["question"])
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
                "question": question,
                "canonical_answer": canonical_answer(answers),
            }
        )

    info = {
        "description": f"TextVQA {split_name} subset for PA-Attack server execution",
        "version": "0.5.1",
        "split": split_name,
    }
    return (
        {
            "info": info,
            "task_type": "Open-Ended",
            "data_type": "textvqa",
            "data_subtype": split_name,
            "questions": questions,
        },
        {
            "info": info,
            "annotations": annotations,
        },
        samples,
    )


def validate_images(image_dir, image_ids):
    missing = [image_id for image_id in image_ids if not (image_dir / f"{image_id}.jpg").exists()]
    if missing:
        preview = ", ".join(missing[:20])
        raise FileNotFoundError(f"Missing {len(missing)} images in {image_dir}: {preview}")


def shard_image_ids(image_ids, num_shards):
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    shard_size = ceil(len(image_ids) / num_shards)
    return [image_ids[index : index + shard_size] for index in range(0, len(image_ids), shard_size)]


def write_dataset_files(base_dir, split_name, image_dir, image_ids, records, num_shards):
    split_dir = base_dir / split_name
    questions, annotations, samples = build_vqa_payloads(records, split_name)
    questions_path = split_dir / f"textvqa_{split_name}_questions_vqa_format.json"
    annotations_path = split_dir / f"textvqa_{split_name}_annotations_vqa_format.json"
    write_json(questions_path, questions)
    write_json(annotations_path, annotations)

    grouped = group_by_image(records)
    shard_infos = []
    for shard_index, shard_image_ids_list in enumerate(shard_image_ids(image_ids, num_shards)):
        shard_records = []
        for image_id in shard_image_ids_list:
            shard_records.extend(grouped[image_id])
        shard_questions, shard_annotations, _ = build_vqa_payloads(shard_records, split_name)
        shard_dir = split_dir / "shards" / f"shard_{shard_index:02d}"
        shard_questions_path = shard_dir / f"textvqa_{split_name}_shard_{shard_index:02d}_questions.json"
        shard_annotations_path = shard_dir / f"textvqa_{split_name}_shard_{shard_index:02d}_annotations.json"
        write_json(shard_questions_path, shard_questions)
        write_json(shard_annotations_path, shard_annotations)
        shard_manifest = {
            "split": split_name,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "image_count": len(shard_image_ids_list),
            "question_count": len(shard_records),
            "image_dir": str(image_dir),
            "questions_json": str(shard_questions_path),
            "annotations_json": str(shard_annotations_path),
            "image_ids_first": shard_image_ids_list[:5],
            "image_ids_last": shard_image_ids_list[-5:],
        }
        write_json(shard_dir / "manifest.json", shard_manifest)
        shard_infos.append(shard_manifest)

    manifest = {
        "split": split_name,
        "image_dir": str(image_dir),
        "image_count": len(image_ids),
        "question_count": len(records),
        "questions_json": str(questions_path),
        "annotations_json": str(annotations_path),
        "samples_preview": samples[:10],
        "shards": shard_infos,
    }
    write_json(split_dir / "manifest.json", manifest)
    return manifest


def parse_splits(value):
    if value == "all":
        return ["train", "validation"]
    splits = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(splits) - set(SPLIT_CONFIG))
    if unknown:
        raise ValueError(f"Unknown split(s): {', '.join(unknown)}")
    return splits


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare TextVQA train/validation PA-Attack server shards.")
    parser.add_argument("--textvqa-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", default="all", help="all, train, validation, or comma-separated list")
    parser.add_argument("--train-image-limit", type=int, default=15000)
    parser.add_argument("--num-shards", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    original_dir = args.textvqa_root / "original_format"
    split_manifests = {}
    for split_name in parse_splits(args.splits):
        config = dict(SPLIT_CONFIG[split_name])
        if split_name == "train":
            config["image_limit"] = args.train_image_limit
        json_path = original_dir / config["json_name"]
        image_dir = original_dir / config["image_dir_name"]
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        image_ids, records = select_records_by_image(payload["data"], config["image_limit"])
        validate_images(image_dir, image_ids)
        split_manifests[split_name] = write_dataset_files(
            args.output_dir,
            split_name,
            image_dir,
            image_ids,
            records,
            args.num_shards,
        )

    top_manifest = {
        "textvqa_root": str(args.textvqa_root),
        "output_dir": str(args.output_dir),
        "num_shards": args.num_shards,
        "splits": split_manifests,
        "selection_rule": {
            "train": "first N unique images in original train JSON order, then all questions for those images",
            "validation": "all unique validation images, then all questions for those images",
        },
    }
    write_json(args.output_dir / "manifest.json", top_manifest)
    print(json.dumps(top_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
