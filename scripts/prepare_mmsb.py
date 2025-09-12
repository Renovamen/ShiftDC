import argparse
import json
from pathlib import Path
from datasets import get_dataset_config_names, load_dataset
from PIL import Image
from tqdm import tqdm


DATASET_ID = "PKU-Alignment/MM-SafetyBench"
IMAGE_TYPES = ("SD", "SD_TYPO", "TYPO")


def _save_image(image_data: Image.Image, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image_data.convert("RGB").save(save_path, format="JPEG")


def convert_mmsb(data_dir: str) -> Path:
    data_root = Path(data_dir).resolve()
    image_root = data_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    output = []
    query_id = 0

    for subset in get_dataset_config_names(DATASET_ID):
        text_only_ds = load_dataset(DATASET_ID, subset, split="Text_only")
        text_only_questions = {row["id"]: row["question"] for row in text_only_ds}

        for image_type in IMAGE_TYPES:
            ds = load_dataset(DATASET_ID, subset, split=image_type)

            for row in tqdm(ds, desc=f"{subset}/{image_type}", unit="item"):
                sample_id = row["id"]
                question = row["question"]
                redteam_query = text_only_questions[sample_id]
                policy = subset

                image_rel_path = Path(policy) / image_type / f"{sample_id}.jpg"
                image_abs_path = image_root / image_rel_path
                if not image_abs_path.exists():
                    _save_image(row["image"], image_abs_path)

                output.append(
                    {
                        "id": query_id,
                        "jailbreak_query": question,
                        "redteam_query": redteam_query,
                        "policy": policy,
                        "image_path": f"images/{image_rel_path.as_posix()}",
                        "image_type": image_type
                    }
                )
                query_id += 1

    output_json = data_root / "data.json"
    with output_json.open("w") as f:
        json.dump(output, f, indent=2)

    return output_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        required=True,
        help=(
            "Target MM-SafetyBench directory. Output will be written under "
            "<data_dir>/images and <data_dir>/data.json."
        )
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = convert_mmsb(args.data_dir)
    print(f"Wrote {out}")
