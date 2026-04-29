import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

PA_ROOT = Path(__file__).resolve().parents[1]
if str(PA_ROOT) not in sys.path:
    sys.path.insert(0, str(PA_ROOT))

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from vlm_eval.attacks.veattack import pgd_veattack


class ClipVisionModel(torch.nn.Module):
    def __init__(self, model, normalize):
        super().__init__()
        self.model = model
        self.normalize = normalize

    def forward(self, vision, output_normalize, tokens=False, attention=False):
        if not tokens:
            feature = self.model(self.normalize(vision))
            if output_normalize:
                feature = F.normalize(feature, dim=-1)
            return feature
        self.model.output_tokens = True
        if attention:
            try:
                feature, token_features, attentions = self.model(self.normalize(vision), output_attentions=True)
            except TypeError as exc:
                if "output_attentions" not in str(exc):
                    raise
                feature, token_features = self.model(self.normalize(vision))
                attentions = None
            if output_normalize:
                feature = F.normalize(feature, dim=-1)
                token_features = F.normalize(token_features, dim=-1)
            return feature, token_features, attentions
        feature, token_features = self.model(self.normalize(vision))
        if output_normalize:
            feature = F.normalize(feature, dim=-1)
            token_features = F.normalize(token_features, dim=-1)
        return feature, token_features


class ComputeLossWrapper:
    def __init__(self, embedding_orig, tokens_orig, target_proto_tokens, tokens_mask, reduction="none"):
        self.embedding_orig = embedding_orig
        self.tokens_orig = tokens_orig
        self.target_proto_tokens = target_proto_tokens
        self.tokens_mask = tokens_mask
        self.reduction = reduction

    def __call__(self, embedding, tokens):
        loss_orig = cosine_similarity_loss(tokens, self.tokens_orig, self.reduction)
        loss_target = cosine_similarity_loss(tokens, self.target_proto_tokens, self.reduction)
        return ((loss_orig - loss_target) * self.tokens_mask).sum()


def cosine_similarity_loss(out, targets, reduction="none"):
    out_norm = F.normalize(out, p=2, dim=-1)
    targets_norm = F.normalize(targets, p=2, dim=-1)
    loss = 1.0 - (out_norm * targets_norm).sum(dim=-1)
    if reduction == "mean":
        return loss.mean()
    return loss


def inverse_normalize(image_normalized, image_processor):
    mean = torch.tensor(image_processor.image_mean).view(1, 3, 1, 1).to(image_normalized.device)
    std = torch.tensor(image_processor.image_std).view(1, 3, 1, 1).to(image_normalized.device)
    return torch.clamp(image_normalized * std + mean, 0, 1)


def load_questions(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["questions"]


def group_questions_by_image(questions):
    grouped = OrderedDict()
    for question in questions:
        image_id = str(question["image_id"])
        grouped.setdefault(image_id, []).append(question)
    return grouped


def make_prompt(question, model_config, conv_mode):
    qs = question.strip() + "\nAnswer the question using a single word or phrase."
    if getattr(model_config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def generate_answer(model, tokenizer, image_processor, model_config, adv_image, question, args):
    prompt = make_prompt(question, model_config, args.conv_mode)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).to(device=args.device, non_blocking=True)
    normalizer = transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    image_tensor = normalizer(adv_image).to(dtype=args.model_dtype, device=args.device, non_blocking=True)
    stop_str = conv_templates[args.conv_mode].sep
    if conv_templates[args.conv_mode].sep_style == SeparatorStyle.TWO:
        stop_str = conv_templates[args.conv_mode].sep2
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    input_token_len = input_ids.shape[1]
    output = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
    if output.endswith(stop_str):
        output = output[: -len(stop_str)]
    return output.strip()


def build_tokens_mask(attentions, attention_layer, mask_scale):
    if attentions is None:
        raise ValueError("attentions is required for attention mask")
    cls2patch = attentions[attention_layer][:, :, 0, 1:].mean(dim=1)
    return F.softmax(mask_scale * cls2patch, dim=-1)


def build_fallback_tokens_mask(tokens, mask_scale):
    token_scores = tokens.detach().norm(dim=-1)
    token_scores = (token_scores - token_scores.mean(dim=-1, keepdim=True)) / (
        token_scores.std(dim=-1, keepdim=True) + 1e-6
    )
    return F.softmax((mask_scale / 20.0) * token_scores, dim=-1)


def build_auto_tokens_mask(tokens, attentions, args):
    if attentions is not None:
        return build_tokens_mask(attentions, args.attention_layer, args.mask_scale)
    return build_fallback_tokens_mask(tokens, args.mask_scale)


def attack_image(clean_image, clip_model_vision, proto_tokens, args):
    output_normalize = False
    clean_image = clean_image.to(args.device, non_blocking=True)
    with torch.no_grad():
        embedding_orig, tokens_orig, attentions = clip_model_vision(
            vision=clean_image,
            output_normalize=output_normalize,
            tokens=True,
            attention=True,
        )
        tokens_mask = build_auto_tokens_mask(tokens_orig, attentions, args)

    tokens_norm = F.normalize(tokens_orig, dim=-1)
    proto_tokens_norm = F.normalize(proto_tokens, dim=-1)
    token_sims = torch.einsum("bld,nld->bnl", tokens_norm, proto_tokens_norm)
    emb_similarity = token_sims.mean(dim=-1)
    _, min_idx = torch.min(emb_similarity, dim=1)
    target_proto_tokens = proto_tokens[min_idx].unsqueeze(0)

    loss_wrapper = ComputeLossWrapper(embedding_orig, tokens_orig, target_proto_tokens, tokens_mask, "none")
    perturbation = torch.zeros_like(clean_image).uniform_(-args.eps_float, args.eps_float).requires_grad_(True)
    adv_image = pgd_veattack(
        forward=clip_model_vision,
        loss_fn=loss_wrapper,
        data_clean=clean_image,
        norm="linf",
        eps=args.eps_float,
        iterations=args.stage1_steps,
        stepsize=args.stepsize,
        output_normalize=output_normalize,
        perturbation=perturbation,
        mode="max",
        verbose=False,
    )

    with torch.no_grad():
        _, tokens_adv, attentions = clip_model_vision(
            vision=adv_image,
            output_normalize=output_normalize,
            tokens=True,
            attention=True,
        )
        tokens_mask = build_auto_tokens_mask(tokens_adv, attentions, args)

    loss_wrapper = ComputeLossWrapper(embedding_orig, tokens_orig, target_proto_tokens, tokens_mask, "none")
    perturbation = torch.zeros_like(adv_image).uniform_(-args.eps_float, args.eps_float).requires_grad_(True)
    return pgd_veattack(
        forward=clip_model_vision,
        loss_fn=loss_wrapper,
        data_clean=adv_image,
        norm="linf",
        eps=args.eps_float,
        iterations=args.steps,
        stepsize=args.stepsize,
        output_normalize=output_normalize,
        perturbation=perturbation,
        mode="max",
        verbose=False,
    ).detach()


def append_jsonl(path, item):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_done_question_ids(path):
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            done.add(json.loads(line)["question_id"])
    return done


def parse_args():
    parser = argparse.ArgumentParser(description="Run PA-Attack on TextVQA shards with LLaVA and save adversarial tensors.")
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prototype", default="prototypes_llava_tokens_bestpca/prototypes_tokens_3000_20_1024.pt")
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--eps", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--stage1-steps", type=int, default=50)
    parser.add_argument("--stepsize", type=float, default=1.0 / 255.0)
    parser.add_argument("--attention-layer", type=int, default=12)
    parser.add_argument("--mask-scale", type=float, default=20.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-png", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.device = "cuda"
    args.model_dtype = torch.float16
    args.eps_float = args.eps / 255.0

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    output_dir = Path(args.output_dir)
    adv_tensor_dir = output_dir / "adv_tensors_by_image"
    adv_png_dir = output_dir / "adv_png_by_image"
    output_dir.mkdir(parents=True, exist_ok=True)
    adv_tensor_dir.mkdir(parents=True, exist_ok=True)
    if args.save_png:
        adv_png_dir.mkdir(parents=True, exist_ok=True)
    answers_path = output_dir / "answers.jsonl"
    manifest_path = output_dir / "manifest.json"

    questions = load_questions(args.question_file)
    grouped = group_questions_by_image(questions)
    done_question_ids = load_done_question_ids(answers_path) if args.skip_existing else set()

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    model, image_processor, tokenizer, _ = load_pretrained_model(
        os.path.expanduser(args.model_path),
        args.model_base,
        model_name,
        pretrained_rob_path="openai",
        dtype="float16",
    )
    model.eval()

    clip_model_orig, _, image_processor_clip = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    normalize = image_processor_clip.transforms[-1]
    clip_model_vision = ClipVisionModel(model=clip_model_orig.visual, normalize=normalize).to(args.device)
    clip_model_orig.to(args.device)
    clip_model_vision.eval()

    proto_tokens = torch.load(args.prototype, map_location="cpu").to(args.device, non_blocking=True)
    started_at = time.strftime("%Y-%m-%d_%H-%M-%S")
    write_manifest = {
        "started_at": started_at,
        "model_path": args.model_path,
        "question_file": args.question_file,
        "image_folder": args.image_folder,
        "output_dir": str(output_dir),
        "prototype": args.prototype,
        "image_count": len(grouped),
        "question_count": len(questions),
        "eps": args.eps,
        "steps": args.steps,
        "stage1_steps": args.stage1_steps,
        "save_png": args.save_png,
    }
    manifest_path.write_text(json.dumps(write_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    to_pil = transforms.ToPILImage()
    for image_id, image_questions in tqdm(grouped.items(), total=len(grouped), desc="PA-Attack TextVQA"):
        if args.skip_existing and all(item["question_id"] in done_question_ids for item in image_questions):
            continue
        image_path = Path(args.image_folder) / f"{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")
        image_normalized = process_images([image], image_processor, model.config)
        clean_image = inverse_normalize(image_normalized, image_processor)

        adv_tensor_path = adv_tensor_dir / f"{image_id}.pt"
        if args.skip_existing and adv_tensor_path.exists():
            adv_image = torch.load(adv_tensor_path, map_location=args.device)
            if adv_image.ndim == 3:
                adv_image = adv_image.unsqueeze(0)
        else:
            adv_image = attack_image(clean_image, clip_model_vision, proto_tokens, args)
            torch.save(adv_image.squeeze(0).detach().cpu(), adv_tensor_path)
            if args.save_png:
                to_pil(adv_image.squeeze(0).detach().cpu()).save(adv_png_dir / f"{image_id}.png")

        for item in image_questions:
            question_id = int(item["question_id"])
            if args.skip_existing and question_id in done_question_ids:
                continue
            answer = generate_answer(
                model,
                tokenizer,
                image_processor,
                model.config,
                adv_image,
                item["question"],
                args,
            )
            append_jsonl(
                answers_path,
                {
                    "question_id": question_id,
                    "image_id": image_id,
                    "question": item["question"],
                    "answer": answer,
                    "adv_tensor": str(adv_tensor_path),
                },
            )
            done_question_ids.add(question_id)

    write_manifest["finished_at"] = time.strftime("%Y-%m-%d_%H-%M-%S")
    write_manifest["answers_jsonl"] = str(answers_path)
    manifest_path.write_text(json.dumps(write_manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
