import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from shiftdc.models import HuggingFace
from shiftdc.utils import get_parent, load_json, prompt


MODE_CONFIG: dict[str, dict[str, Any]] = {
    "tt": {
        "default_batch_size": 32,
        "file_prefix": "tt",
    },
    "vl": {
        "default_batch_size": 16,
        "file_prefix": "vl",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=sorted(MODE_CONFIG), required=True)
    parser.add_argument("--caption_jsonl", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_model_len", type=int, default=1024)
    parser.add_argument("--start_index", type=int, default=0)
    return parser.parse_args()


def resolve_batch_size(mode: str, batch_size: int | None) -> int:
    if batch_size is None:
        return int(MODE_CONFIG[mode]["default_batch_size"])
    return batch_size


def build_activation_inputs(
    rows: list[dict[str, Any]],
    data_root: Path,
    mode: str,
    start_index: int,
) -> tuple[list[str], list[str] | None, list[dict[str, object]]]:
    indexed_rows = list(enumerate(rows))[start_index:]
    if len(indexed_rows) == 0:
        raise ValueError("No samples selected after applying --start_index.")

    prompts: list[str] = []
    image_paths: list[str] = []
    index_rows: list[dict[str, object]] = []

    for act_row_idx, (source_row_idx, row) in enumerate(indexed_rows):
        image_path = str((data_root / str(row["image_path"])).resolve())
        if mode == "tt":
            current_prompt = prompt.ANSWER_Q_BASED_ON_CAPTION.format(
                query=row["jailbreak_query"].strip(),
                caption=row["caption"].strip(),
            )
        else:
            current_prompt = row["jailbreak_query"].strip()
            image_paths.append(image_path)

        prompts.append(current_prompt)
        index_rows.append(
            {
                "row_idx": act_row_idx,
                "source_row_idx": source_row_idx,
                "id": row.get("id"),
                "image_path": image_path,
            }
        )

    return prompts, image_paths or None, index_rows


def run_activation(
    model_name: str,
    mode: str,
    caption_jsonl: str,
    data_dir: str,
    batch_size: int | None,
    max_model_len: int,
    start_index: int,
) -> Path:
    if batch_size is not None and batch_size <= 0:
        raise ValueError("--batch_size must be > 0.")
    if max_model_len <= 0:
        raise ValueError("--max_model_len must be > 0.")
    if start_index < 0:
        raise ValueError("--start_index must be >= 0.")

    batch_size = resolve_batch_size(mode, batch_size)

    caption_path = Path(caption_jsonl).resolve()
    data_root = Path(data_dir).resolve()
    rows = load_json(str(caption_path))
    prompts, image_paths, index_rows = build_activation_inputs(
        rows=rows,
        data_root=data_root,
        mode=mode,
        start_index=start_index,
    )

    hf = HuggingFace(model_name)
    try:
        _, count, all_activations = hf.extract_last_token_activations(
            prompts=prompts,
            image_paths=image_paths,
            batch_size=batch_size,
            max_model_len=max_model_len,
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
    file_prefix = str(MODE_CONFIG[mode]["file_prefix"])
    activation_path = out_dir / f"{file_prefix}_activations.npy"
    index_path = out_dir / f"{file_prefix}_index.jsonl"
    meta_path = out_dir / f"{file_prefix}_meta.json"

    np.save(activation_path, all_activations.astype(np.float32))
    with index_path.open("w") as f:
        for row in index_rows:
            f.write(json.dumps(row) + "\n")

    metadata = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model_name": model_name,
        "caption_jsonl": str(caption_path),
        "data_dir": str(data_root),
        "output_dir": str(out_dir),
        "activation_file": str(activation_path),
        "index_file": str(index_path),
        "shape": list(all_activations.shape),
        "count": count,
        "args": {
            "batch_size": batch_size,
            "max_model_len": max_model_len,
            "start_index": start_index,
        },
        "mode": mode,
    }
    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {activation_path}")
    print(f"Wrote {index_path}")
    print(f"Wrote {meta_path}")
    return activation_path


if __name__ == "__main__":
    args = parse_args()
    run_activation(
        model_name=args.model_name,
        mode=args.mode,
        caption_jsonl=args.caption_jsonl,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        start_index=args.start_index,
    )
