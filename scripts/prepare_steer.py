import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
import re
import sys
import urllib.request
import zipfile
from pathlib import Path
import threading
import torch
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shiftdc.models import OpenAI


COCO_TRAIN2014_ZIP_URL = "http://images.cocodataset.org/zips/train2014.zip"
LLAVA_DATASET_REPO = "liuhaotian/LLaVA-Instruct-150K"
LLAVA_FILENAME = "llava_instruct_80k.json"

EXCLUDED_POLICIES = {
    "political_lobbying",
    "legal_opinion",
    "financial_advice",
    "health_consultation",
    "gov_decision"
}

NUM_SAMPLES = 160
RANDOM_SEED = 42

DEFAULT_CAPTION_MODEL = "gpt-4o-2024-11-20"
MMSB_FONT_URL = "https://github.com/isXinLiu/MM-SafetyBench/raw/main/assets/font/ARIAL.TTF"
TYPO_FONT_SIZE = 90
TYPO_MAX_WIDTH = 1024

KEYWORD_PROMPT = (
    "Based on the given Request, provide a **phrase** that describes the image. "
    "Output only a short phrase, not a full sentence. "
)
# LLAVA_CAPTION_PROMPT = (
#     "Based on the given Request, describe the image. "
#     "Keep your description concise. "
#     "Request: {request}"
# )
CAPTION_PROMPT = (
    "Describe the image. "
    "Keep your description concise. "
)


def download_llava_images(data_root: Path) -> Path:
    images_dir = data_root / "llava" / "images"
    target_dir = images_dir / "original"
    train2014_dir = images_dir / "train2014"
    images_dir.mkdir(parents=True, exist_ok=True)

    if target_dir.exists() and any(target_dir.iterdir()):
        return target_dir

    zip_path = images_dir / "train2014.zip"
    if not zip_path.exists():
        torch.hub.download_url_to_file(COCO_TRAIN2014_ZIP_URL, str(zip_path), progress=True)

    if not train2014_dir.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(images_dir)

    if train2014_dir.exists() and not target_dir.exists():
        train2014_dir.rename(target_dir)

    return target_dir


def build_llava_data(data_root: Path, num_samples: int = NUM_SAMPLES) -> Path:
    output_json = data_root / "llava" / "data.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    if output_json.exists():
        return output_json

    instruct_json_path = hf_hub_download(
        repo_id=LLAVA_DATASET_REPO,
        repo_type="dataset",
        filename=LLAVA_FILENAME
    )

    with open(instruct_json_path, "r") as f:
        all_rows = json.load(f)

    rng = random.Random(RANDOM_SEED)

    def _llava_image_rel_path(row: dict) -> Path:
        return Path("llava/images/original") / f"COCO_train2014_{row['id']}.jpg"

    indices = rng.sample(range(len(all_rows)), num_samples)
    sampled = [all_rows[i] for i in indices]

    records = []
    for i, row in enumerate(sampled):
        conversations = row["conversations"]
        if isinstance(conversations, str):
            conversations = json.loads(conversations)

        question = conversations[0]["value"]
        question = question.replace("<image>", "").strip()

        records.append(
            {
                "id": i,
                "image_path": _llava_image_rel_path(row).as_posix(),
                "question": question
            }
        )

    with output_json.open("w") as f:
        json.dump(records, f, indent=2)

    return output_json


def _sample_uniform_by_policy(
    grouped: dict[str, list[dict]],
    num_samples: int,
    rng: random.Random
) -> list[dict]:
    policies = sorted(grouped.keys())

    available = sum(len(grouped[p]) for p in policies)
    if available < num_samples:
        raise ValueError(
            f"Requested {num_samples} MM-SafetyBench samples but only {available} eligible rows are available."
        )

    base = num_samples // len(policies)
    target = {p: min(base, len(grouped[p])) for p in policies}
    selected = sum(target.values())
    remain = num_samples - selected

    while remain > 0:
        progressed = False
        for p in policies:
            if remain == 0:
                break
            if target[p] < len(grouped[p]):
                target[p] += 1
                remain -= 1
                progressed = True
        if not progressed:
            break

    sampled: list[dict] = []
    for p in policies:
        sampled.extend(rng.sample(grouped[p], target[p]))

    rng.shuffle(sampled)
    return sampled


def build_mmsb_data(
    data_root: Path,
    mmsb_data_dir: Path,
    num_samples: int = NUM_SAMPLES
) -> Path:
    output_json = data_root / "mmsb" / "data.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    if output_json.exists():
        return output_json

    source_json = mmsb_data_dir / "data.json"
    with source_json.open("r") as f:
        all_rows = json.load(f)

    grouped_sd: dict[str, list[dict]] = {}
    row_index: dict[tuple[str, str, str], dict] = {}

    for row in all_rows:
        policy = str(row["policy"])
        if policy.lower() in EXCLUDED_POLICIES:
            continue

        image_type = str(row["image_type"])
        image_name = Path(row["image_path"]).name
        row_index[(policy, image_type, image_name)] = row

        if image_type == "SD":
            grouped_sd.setdefault(policy, []).append(row)

    rng = random.Random(RANDOM_SEED)
    sampled_sd = _sample_uniform_by_policy(grouped_sd, num_samples, rng)

    records = []
    for sd_row in sampled_sd:
        policy = str(sd_row["policy"])
        image_name = Path(sd_row["image_path"]).name

        for image_type in ("SD", "SD_TYPO", "TYPO"):
            row = row_index.get((policy, image_type, image_name))
            if row is None:
                continue

            rel = Path(row["image_path"])
            src_image = (mmsb_data_dir / rel).resolve()
            rel_to_current = os.path.relpath(src_image, data_root.resolve())

            records.append({
                **row,
                "image_path": Path(rel_to_current).as_posix()
            })

    with output_json.open("w") as f:
        json.dump(records, f, indent=2)

    return output_json


# https://github.com/isXinLiu/MM-SafetyBench/blob/main/creation/2_img_process.py
def _ensure_mmsb_font(data_root: Path) -> Path:
    font_path = data_root / "llava" / "assets" / "font" / "ARIAL.TTF"
    font_path.parent.mkdir(parents=True, exist_ok=True)
    if not font_path.exists():
        urllib.request.urlretrieve(MMSB_FONT_URL, str(font_path))
    return font_path


def _typo_format_text(text: str, font: ImageFont.FreeTypeFont) -> tuple[str, int]:
    img = Image.new("RGB", (TYPO_MAX_WIDTH, 100), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    words = text.split()
    if not words:
        return "", 1

    formatted = words[0]
    cur_len = draw.textlength(formatted, font=font)
    line_num = 1
    for word in words[1:]:
        add_len = draw.textlength(f" {word}", font=font)
        if cur_len + add_len < TYPO_MAX_WIDTH:
            formatted += f" {word}"
            cur_len += add_len
        else:
            formatted += f"\n {word}"
            cur_len = draw.textlength(f" {word}", font=font)
            line_num += 1
    return formatted, line_num


def _draw_typo_image(text: str, output_path: Path, font: ImageFont.FreeTypeFont) -> None:
    formatted_text, line_num = _typo_format_text(text, font)
    max_height = TYPO_FONT_SIZE * (line_num + 1)
    img = Image.new("RGB", (TYPO_MAX_WIDTH, max_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, TYPO_FONT_SIZE / 2.0), formatted_text, (0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def _vertical_concat(image1: Path, image2: Path, output_path: Path) -> None:
    img1 = Image.open(image1)
    img2 = Image.open(image2)
    width1, height1 = img1.size
    width2, height2 = img2.size
    result = Image.new("RGB", (max(width1, width2), height1 + height2))
    result.paste(img1, (0, 0))
    result.paste(img2, (0, height1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)


def build_llava_typo_and_sd_typo(data_root: Path, llava_data_json: Path) -> Path:
    with llava_data_json.open("r") as f:
        rows = json.load(f)

    if not rows:
        return llava_data_json

    base_rows = [row for row in rows if row.get("image_type", "SD") == "SD"]
    existing_types = {
        (Path(row["image_path"]).name, row.get("image_type", "SD"))
        for row in rows
    }

    pending_base_rows = []
    for row in base_rows:
        image_name = Path(row["image_path"]).name
        has_typo = (image_name, "TYPO") in existing_types
        has_sd_typo = (image_name, "SD_TYPO") in existing_types
        if not (has_typo and has_sd_typo):
            pending_base_rows.append(row)

    if not pending_base_rows:
        return llava_data_json

    font_path = _ensure_mmsb_font(data_root)
    font = ImageFont.truetype(str(font_path), TYPO_FONT_SIZE)

    new_rows = []
    for row in pending_base_rows:
        base = dict(row)
        base.setdefault("image_type", "SD")
        base_id = int(base["id"])

        keyword = str(base.get("keyword", "")).strip() or str(base.get("question", "")).strip()
        keyword = re.sub(r"[^\w\s]", "", keyword).lower()
        image_name = Path(base["image_path"]).name

        original_abs = data_root / base["image_path"]
        typo_rel = Path("llava/images/TYPO") / image_name
        sd_typo_rel = Path("llava/images/SD_TYPO") / image_name
        typo_abs = data_root / typo_rel
        sd_typo_abs = data_root / sd_typo_rel

        if (image_name, "TYPO") not in existing_types:
            _draw_typo_image(keyword, typo_abs, font)
            typo_row = {**base, "image_path": typo_rel.as_posix(), "image_type": "TYPO"}
            typo_row["id"] = base_id + NUM_SAMPLES
            typo_row.pop("caption", None)
            new_rows.append(typo_row)
            existing_types.add((image_name, "TYPO"))

        if (image_name, "SD_TYPO") not in existing_types:
            if not typo_abs.exists():
                _draw_typo_image(keyword, typo_abs, font)
            _vertical_concat(original_abs, typo_abs, sd_typo_abs)
            sd_typo_row = {**base, "image_path": sd_typo_rel.as_posix(), "image_type": "SD_TYPO"}
            sd_typo_row["id"] = base_id + NUM_SAMPLES * 2
            sd_typo_row.pop("caption", None)
            new_rows.append(sd_typo_row)
            existing_types.add((image_name, "SD_TYPO"))

    if new_rows:
        rows.extend(new_rows)
        with llava_data_json.open("w") as f:
            json.dump(rows, f, indent=2)

    return llava_data_json


def add_captions(
    data_root: Path,
    data_json: Path,
    api_key: str,
    model_name: str,
    workers: int
) -> None:
    _add_generated_field(
        data_root=data_root,
        data_json=data_json,
        field_name="caption",
        prompt_template=CAPTION_PROMPT,
        api_key=api_key,
        model_name=model_name,
        workers=workers,
        max_output_tokens=128,
        progress_desc=f"Captioning {data_json.parent.name}"
    )


def add_keywords(
    data_root: Path,
    data_json: Path,
    api_key: str,
    model_name: str,
    workers: int
) -> None:
    _add_generated_field(
        data_root=data_root,
        data_json=data_json,
        field_name="keyword",
        prompt_template=KEYWORD_PROMPT,
        api_key=api_key,
        model_name=model_name,
        workers=workers,
        max_output_tokens=16,
        progress_desc=f"Keywording {data_json.parent.name}"
    )


def _add_generated_field(
    data_root: Path,
    data_json: Path,
    field_name: str,
    prompt_template: str,
    api_key: str,
    model_name: str,
    workers: int,
    max_output_tokens: int,
    progress_desc: str,
) -> None:
    with data_json.open("r") as f:
        rows = json.load(f)

    if not rows:
        return
    # pending_payloads = [(idx, row) for idx, row in enumerate(rows)]
    pending_payloads = [(idx, row) for idx, row in enumerate(rows) if field_name not in row]
    if not pending_payloads:
        print(f"Skip {field_name} generation {data_json}: all items already have `{field_name}`.")
        return

    thread_local = threading.local()

    def _get_model() -> OpenAI:
        model = getattr(thread_local, "model", None)
        if model is None:
            model = OpenAI(checkpoint=model_name, key=api_key)
            thread_local.model = model
        return model

    def _generation(payload: tuple[int, dict]):
        idx, row = payload
        model = _get_model()

        request_text = row.get("question") or row.get("jailbreak_query") or ""
        prompt = (
            prompt_template.format(request=request_text.strip())
            if "{request}" in prompt_template
            else prompt_template
        )
        image_path = str((data_root / row["image_path"]).resolve())

        value = model.generate(
            question=prompt,
            image_path=image_path,
            max_output_tokens=max_output_tokens
        )[0]["response"]

        return idx, value

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_generation, payload) for payload in pending_payloads]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=progress_desc):
            idx, value = fut.result()
            rows[idx][field_name] = value
            with data_json.open("w") as f:
                json.dump(rows, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        required=True,
        help="Root data directory."
    )
    parser.add_argument(
        "--mmsb_data_dir",
        type=str,
        required=True,
        help="Directory containing MM-SafetyBench data.json for sampling."
    )
    parser.add_argument(
        "--openai_key",
        type=str,
        default=None,
        help="OpenAI API key. If omitted, loaded from .env / OPENAI_API_KEY."
    )
    parser.add_argument(
        "--openai_model",
        type=str,
        default=DEFAULT_CAPTION_MODEL,
        help="OpenAI model used for caption generation."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of workers for parallel caption generation."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(args.data_dir).resolve()
    mmsb_root = Path(args.mmsb_data_dir).resolve()

    image_dir = download_llava_images(root)
    llava_data_json = build_llava_data(root)
    mmsb_data_json = build_mmsb_data(root, mmsb_root)

    load_dotenv()

    api_key = args.openai_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OpenAI API key. Pass --openai_key or set OPENAI_API_KEY in .env.")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    add_keywords(
        data_root=root,
        data_json=llava_data_json,
        api_key=api_key,
        model_name=args.openai_model,
        workers=args.workers
    )
    llava_data_json = build_llava_typo_and_sd_typo(root, llava_data_json)

    add_captions(
        data_root=root,
        data_json=llava_data_json,
        api_key=api_key,
        model_name=args.openai_model,
        workers=args.workers
    )
    add_captions(
        data_root=root,
        data_json=mmsb_data_json,
        api_key=api_key,
        model_name=args.openai_model,
        workers=args.workers
    )

    print(f"Prepared images at {image_dir}")
    print(f"Wrote {llava_data_json}")
    print(f"Wrote {mmsb_data_json}")
