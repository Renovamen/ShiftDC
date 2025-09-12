import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from shiftdc.models import HuggingFace
from shiftdc.utils import get_parent, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, required=True)
    parser.add_argument("--caption_jsonl", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_model_len", type=int, default=1024)
    parser.add_argument("--start_index", type=int, default=0)
    return parser.parse_args()


def run_vl_activation(
    model_name: str,
    caption_jsonl: str,
    data_dir: str,
    batch_size: int,
    max_model_len: int,
    start_index: int,
) -> Path:
    if batch_size <= 0:
        raise ValueError("--batch_size must be > 0.")
    if max_model_len <= 0:
        raise ValueError("--max_model_len must be > 0.")
    if start_index < 0:
        raise ValueError("--start_index must be >= 0.")

    caption_path = Path(caption_jsonl).resolve()
    data_root = Path(data_dir).resolve()
    rows = load_json(str(caption_path))
    indexed_rows = list(enumerate(rows))[start_index:]
    if len(indexed_rows) == 0:
        raise ValueError("No samples selected after applying --start_index.")

    prompts: list[str] = []
    image_paths: list[str] = []
    index_rows: list[dict[str, object]] = []
    for act_row_idx, (source_row_idx, row) in enumerate(indexed_rows):
        prompt_vl = row["jailbreak_query"].strip()
        image_path = str((data_root / str(row["image_path"])).resolve())
        prompts.append(prompt_vl)
        image_paths.append(image_path)
        index_rows.append(
            {
                "row_idx": act_row_idx,
                "source_row_idx": source_row_idx,
                "id": row.get("id"),
                "image_path": image_path
            }
        )

    hf = HuggingFace(model_name)
    try:
        _, count, all_activations = hf.extract_last_token_activations(
            prompts=prompts,
            image_paths=image_paths,
            batch_size=batch_size,
            max_model_len=max_model_len,
            desc="Extracting VL activations",
        )
    finally:
        hf.del_model()

    if count != len(prompts):
        raise ValueError(f"Activation count mismatch: expected {len(prompts)}, got {count}.")
    if all_activations.shape[0] != len(prompts):
        raise ValueError(
            f"Activation tensor first dim mismatch: expected {len(prompts)}, got {all_activations.shape[0]}."
        )

    out_dir = Path(get_parent(str(caption_path)))
    vl_npy = out_dir / "vl_activations.npy"
    vl_index = out_dir / "vl_index.jsonl"
    vl_meta = out_dir / "vl_meta.json"

    np.save(vl_npy, all_activations.astype(np.float32))
    with vl_index.open("w") as f:
        for row in index_rows:
            f.write(json.dumps(row) + "\n")

    metadata = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model_name": model_name,
        "caption_jsonl": str(caption_path),
        "data_dir": str(data_root),
        "output_dir": str(out_dir),
        "activation_file": str(vl_npy),
        "index_file": str(vl_index),
        "shape": list(all_activations.shape),
        "count": count,
        "args": {
            "batch_size": batch_size,
            "max_model_len": max_model_len,
            "start_index": start_index,
        },
        "mode": "vl",
    }
    with vl_meta.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {vl_npy}")
    print(f"Wrote {vl_index}")
    print(f"Wrote {vl_meta}")
    return vl_npy


if __name__ == "__main__":
    args = parse_args()
    run_vl_activation(
        model_name=args.model_name,
        caption_jsonl=args.caption_jsonl,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        start_index=args.start_index,
    )
