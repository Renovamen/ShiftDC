import argparse
import json
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from shiftdc.models import build_vlm
from shiftdc.utils import make_dir, prompt


def run_caption(model_name: str, data_dir: str, output_dir: str, start_index: int = 0) -> Path:
    data_root = Path(data_dir).resolve()
    input_json = data_root / "data.json"

    with input_json.open("r") as f:
        data = json.load(f)

    model = build_vlm(model_name)

    benchmark = data_root.name
    checkpoint = model.checkpoint.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_jsonl = Path(output_dir) / benchmark / checkpoint / timestamp / "caption.jsonl"
    make_dir(str(output_jsonl))

    with output_jsonl.open("w") as f:
        for i, item in tqdm(enumerate(data), total=len(data), desc=f"Captioning {benchmark}"):
            if i < start_index:
                continue

            image_path = str(data_root / item["image_path"])
            completion = model.generate(
                question=prompt.CAPTION_BASED_ON_Q.format(question=item["jailbreak_query"]),
                image_path=image_path,
                max_tokens=128
            )[0]

            record = {
                **completion,
                "id": item["id"],
                "policy": item.get("policy"),
                "redteam_query": item.get("redteam_query"),
                "jailbreak_query": item.get("jailbreak_query"),
                "image_path": item.get("image_path")
            }
            if "image_type" in item:
                record["image_type"] = item["image_type"]

            f.write(json.dumps(record) + "\n")
            f.flush()

    return output_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        required=True
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        required=True,
        help="Input data directory containing `data.json` and `images` folder.",
    )
    parser.add_argument(
        "-s",
        "--start_index",
        type=int,
        default=0,
        help="Start from this sample index.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="outputs",
        help="Base output directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output = run_caption(
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        start_index=args.start_index,
    )

    print(f"Wrote {output}")
