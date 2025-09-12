import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from shiftdc.models import HuggingFace
from shiftdc.utils import get_parent, load_json, prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, required=True)
    parser.add_argument("--caption_jsonl", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_model_len", type=int, default=1024)
    parser.add_argument("--start_index", type=int, default=0)
    return parser.parse_args()


def run_tt_activation(
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
    index_rows: list[dict[str, object]] = []
    for act_row_idx, (source_row_idx, row) in enumerate(indexed_rows):
        prompt_tt = prompt.ANSWER_Q_BASED_ON_CAPTION.format(
            query=row["jailbreak_query"].strip(),
            caption=row["caption"].strip()
        )
        image_path = str((data_root / str(row["image_path"])).resolve())
        prompts.append(prompt_tt)
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
            image_paths=None,
            batch_size=batch_size,
            max_model_len=max_model_len,
            desc="Extracting TT activations"
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
    tt_npy = out_dir / "tt_activations.npy"
    tt_index = out_dir / "tt_index.jsonl"
    tt_meta = out_dir / "tt_meta.json"

    np.save(tt_npy, all_activations.astype(np.float32))
    with tt_index.open("w") as f:
        for row in index_rows:
            f.write(json.dumps(row) + "\n")

    metadata = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model_name": model_name,
        "caption_jsonl": str(caption_path),
        "data_dir": str(data_root),
        "output_dir": str(out_dir),
        "activation_file": str(tt_npy),
        "index_file": str(tt_index),
        "shape": list(all_activations.shape),
        "count": count,
        "args": {
            "batch_size": batch_size,
            "max_model_len": max_model_len,
            "start_index": start_index,
        },
        "mode": "tt",
    }
    with tt_meta.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {tt_npy}")
    print(f"Wrote {tt_index}")
    print(f"Wrote {tt_meta}")
    return tt_npy


if __name__ == "__main__":
    args = parse_args()
    run_tt_activation(
        model_name=args.model_name,
        caption_jsonl=args.caption_jsonl,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        start_index=args.start_index,
    )
