import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm

from prototype.prototype_pca import (
    CocoImages,
    build_prototypes_with_pca,
    extract_features,
    force_cudnn_initialization,
    get_eval_model,
    eval_model as clip_eval_model,
)


class ArgsLike:
    model = "llava"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coco-val-image-dir",
        default="/var/tmp/jnuadmin_vlm/VLM/dataset/MSCOCO/val2014",
    )
    parser.add_argument(
        "--model-path",
        default="/var/tmp/jnuadmin_vlm/VLM/Attack/PA-Attack_TextVQA/models/llava-v1.5-7b",
    )
    parser.add_argument(
        "--output",
        default="prototypes_llava_tokens_bestpca/prototypes_tokens_3000_20_1024.pt",
    )
    parser.add_argument("--num-samples", type=int, default=3000)
    parser.add_argument("--num-prototypes", type=int, default=20)
    parser.add_argument("--pca-dim", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--precision", default="float16", choices=["float16", "float32"])
    return parser.parse_args()


def main():
    args = parse_args()
    image_dir = Path(args.coco_val_image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"COCO val image directory does not exist: {image_dir}")

    image_count = sum(1 for path in image_dir.iterdir() if path.suffix.lower() == ".jpg")
    if image_count < args.num_samples:
        raise ValueError(
            f"Requested {args.num_samples} images, but only found {image_count}: {image_dir}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LLaVA prototype generation.")

    model_args = {
        "model_path": args.model_path,
        "vision_encoder_pretrained": "openai",
        "precision": args.precision,
        "temperature": "0.0",
        "num_beams": "1",
    }

    print("Loading LLaVA model for prototype generation")
    lvlm_model = get_eval_model(ArgsLike(), model_args, adversarial=True)
    force_cudnn_initialization()
    lvlm_model.set_device(0)

    print(f"Loading {args.num_samples} COCO val images from {image_dir}")
    dataset = CocoImages(str(image_dir), lvlm_model=lvlm_model, n_samples=args.num_samples)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = "cuda"
    clip_eval_model.vision_encoder.output_tokens = True
    _, tokens_feats = extract_features(dataloader, clip_eval_model.vision_encoder, device)

    print(
        f"Building {args.num_prototypes} token prototypes with PCA dim {args.pca_dim}"
    )
    proto_tokens, _ = build_prototypes_with_pca(
        tokens_feats,
        n_proto=args.num_prototypes,
        pca_dim=args.pca_dim,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(proto_tokens, output)
    print(f"Saved prototype: {output}")


if __name__ == "__main__":
    main()
