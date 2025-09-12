import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np

from shiftdc.models import HuggingFace
from shiftdc.utils import load_json, prompt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, required=True)
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default="data/steer",
        help="Input root directory containing `llava/data.json` and `mmsb/data.json`.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="outputs/safety_shift_hf",
        help="Directory to save activations.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_model_len", type=int, default=1024)
    return parser.parse_args()


def _sample_rows(rows: list[dict], max_samples: int | None, seed: int) -> list[dict]:
    if max_samples is None or max_samples >= len(rows):
        return rows
    rng = random.Random(seed)
    indices = rng.sample(range(len(rows)), max_samples)
    return [rows[i] for i in indices]


def _build_prompts(rows: list[dict], query_field: str) -> list[str]:
    prompts: list[str] = []
    for row in rows:
        query = row.get(query_field)
        caption = row.get("caption")
        prompts.append(prompt.ANSWER_Q_BASED_ON_CAPTION.format(
            query=query.strip(),
            caption=caption.strip()
        ))
    return prompts


def run_safety_shift(
    model_name: str,
    input_dir: str,
    output_dir: str,
    batch_size: int,
    max_samples: int | None,
    seed: int,
    max_model_len: int
) -> Path:
    if batch_size <= 0:
        raise ValueError("--batch_size must be > 0.")
    if max_model_len <= 0:
        raise ValueError("--max_model_len must be > 0.")

    input_root = Path(input_dir).resolve()
    llava_path = input_root / "llava" / "data.json"
    mmsb_path = input_root / "mmsb" / "data.json"
    if not llava_path.exists():
        raise FileNotFoundError(f"Missing llava json: {llava_path}")
    if not mmsb_path.exists():
        raise FileNotFoundError(f"Missing mmsb json: {mmsb_path}")

    llava_rows = _sample_rows(load_json(str(llava_path)), max_samples=max_samples, seed=seed)
    mmsb_rows = _sample_rows(load_json(str(mmsb_path)), max_samples=max_samples, seed=seed)

    llava_prompts = _build_prompts(llava_rows, query_field="question")
    mmsb_prompts = _build_prompts(mmsb_rows, query_field="jailbreak_query")

    output_root = Path(output_dir).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / Path(model_name).name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    hf = HuggingFace(model_name)
    num_hidden_layers = hf.num_hidden_layers

    try:
        llava_sum, llava_count, llava_all = hf.extract_last_token_activations(
            prompts=llava_prompts,
            batch_size=batch_size,
            max_model_len=max_model_len,
            desc="Extracting llava"
        )
        mmsb_sum, mmsb_count, mmsb_all = hf.extract_last_token_activations(
            prompts=mmsb_prompts,
            batch_size=batch_size,
            max_model_len=max_model_len,
            desc="Extracting mmsb"
        )
    finally:
        hf.del_model()

    mean_llava = (llava_sum / llava_count).astype(np.float32)
    mean_mmsb = (mmsb_sum / mmsb_count).astype(np.float32)
    delta_mmsb_minus_llava = (mean_mmsb - mean_llava).astype(np.float32)
    delta_llava_minus_mmsb = (mean_llava - mean_mmsb).astype(np.float32)

    np.save(run_dir / "mean_llava.npy", mean_llava)
    np.save(run_dir / "mean_mmsb.npy", mean_mmsb)
    np.save(run_dir / "delta_mmsb_minus_llava.npy", delta_mmsb_minus_llava)
    np.save(run_dir / "delta_llava_minus_mmsb.npy", delta_llava_minus_mmsb)
    np.save(run_dir / "all_llava_activations.npy", llava_all)
    np.save(run_dir / "all_mmsb_activations.npy", mmsb_all)

    layer_norms_m2l = np.linalg.norm(delta_mmsb_minus_llava, axis=1)
    layer_norms_l2m = np.linalg.norm(delta_llava_minus_mmsb, axis=1)
    metadata = {
        "timestamp": timestamp,
        "model_name": model_name,
        "extract_backend": "hf",
        "input_dir": str(input_root),
        "llava_json": str(llava_path),
        "mmsb_json": str(mmsb_path),
        "prompt_template": prompt.ANSWER_Q_BASED_ON_CAPTION,
        "counts": {
            "llava": llava_count,
            "mmsb": mmsb_count,
        },
        "shape": {
            "num_layers": int(delta_mmsb_minus_llava.shape[0]),
            "hidden_size": int(delta_mmsb_minus_llava.shape[1]),
            "tensor_shape": [int(delta_mmsb_minus_llava.shape[0]), int(delta_mmsb_minus_llava.shape[1])],
            "all_llava_shape": list(llava_all.shape),
            "all_mmsb_shape": list(mmsb_all.shape),
        },
        "layer_ids": list(range(num_hidden_layers)),
        "delta_layer_norm_stats": {
            "mmsb_minus_llava": {
                "min": float(layer_norms_m2l.min()),
                "mean": float(layer_norms_m2l.mean()),
                "max": float(layer_norms_m2l.max()),
            },
            "llava_minus_mmsb": {
                "min": float(layer_norms_l2m.min()),
                "mean": float(layer_norms_l2m.mean()),
                "max": float(layer_norms_l2m.max()),
            },
        },
        "args": {
            "batch_size": batch_size,
            "max_samples": max_samples,
            "seed": seed,
            "max_model_len": max_model_len
        },
    }
    with (run_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved outputs to: {run_dir}")
    print(f"Dataset sizes used: llava={llava_count}, mmsb={mmsb_count}")
    print(f"Activation shape [num_layers, hidden_size]: {tuple(delta_mmsb_minus_llava.shape)}")
    print(
        "Per-layer delta norm stats (mmsb_minus_llava): "
        f"min={metadata['delta_layer_norm_stats']['mmsb_minus_llava']['min']:.6f}, "
        f"mean={metadata['delta_layer_norm_stats']['mmsb_minus_llava']['mean']:.6f}, "
        f"max={metadata['delta_layer_norm_stats']['mmsb_minus_llava']['max']:.6f}"
    )
    return run_dir


if __name__ == "__main__":
    args = _parse_args()

    run_safety_shift(
        model_name=args.model_name,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        seed=args.seed,
        max_model_len=args.max_model_len,
    )
