import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import torch
from tqdm import tqdm

from shiftdc.models import HuggingFace
from shiftdc.steering import ShiftDCSteeringVector
from shiftdc.utils import load_json, prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model_name", type=str, required=True)

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing caption.jsonl, tt_activations.npy, tt_index.jsonl, vl_activations.npy, and vl_index.jsonl.",
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--safety_shift_npy", type=str, required=True)

    parser.add_argument("--layer_start", type=int, default=None)
    parser.add_argument("--layer_end", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--min_token_index", type=int, default=-1)

    parser.add_argument("--max_tokens", type=int, default=128)

    return parser.parse_args()

def _find_decoder_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    # TODO: avoid hardcoding here
    return list(model.language_model.layers)

def generate_steered_hf(
    hf: HuggingFace,
    layer_modules: list[torch.nn.Module],
    device: torch.device,
    prompt_qi: str,
    image_path: str | None,
    steering_vector: ShiftDCSteeringVector,
    min_token_index: int,
    max_tokens: int,
) -> str:
    inputs_qi = hf._build_generation_inputs(question=prompt_qi, image_path=image_path)
    model_inputs = {k: v.to(device) for k, v in inputs_qi.items()}
    prompt_len = model_inputs["input_ids"].shape[1]

    with torch.no_grad():
        with steering_vector.apply(
            layer_modules=layer_modules,
            min_token_index=min_token_index
        ):
            output_ids = hf.model.generate(
                **model_inputs,
                do_sample=False,
                max_new_tokens=max_tokens,
                use_cache=True
            )

    gen_ids = output_ids[0][prompt_len:]
    return hf.tokenizer.decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    ).strip()

def _validate_inputs(
    safety_shift: np.ndarray,
    tt_acts: np.ndarray,
    vl_acts: np.ndarray,
    tt_index: list[dict[str, Any]],
    vl_index: list[dict[str, Any]],
    rows: list[dict[str, Any]]
) -> tuple[int, int]:
    # Check shape dims
    if safety_shift.ndim != 2:
        raise ValueError(f"Safety shift must be [num_layers, hidden_size], got {safety_shift.shape}.")
    if tt_acts.ndim != 3 or vl_acts.ndim != 3:
        raise ValueError(f"TT/VL activations must be [num_samples, num_layers, hidden_size], got {tt_acts.shape}, {vl_acts.shape}.")
    if tt_acts.shape != vl_acts.shape:
        raise ValueError(f"TT/VL activation shape mismatch: {tt_acts.shape} vs {vl_acts.shape}.")

    # If the number of activation rows matches saved indices
    if tt_acts.shape[0] != len(tt_index) or vl_acts.shape[0] != len(vl_index):
        raise ValueError("Activation and index lengths do not match.")
    # If tt and vl activation have the same number of activation rows
    if len(tt_index) != len(vl_index):
        raise ValueError("TT/VL index lengths do not match.")
    # If tt and vl activation rows & data items are in the same order
    for i, (tt_row, vl_row) in enumerate(zip(tt_index, vl_index)):
        for key in ("row_idx", "source_row_idx", "id", "image_path"):
            if tt_row.get(key) != vl_row.get(key):
                raise ValueError(f"TT/VL index mismatch at i={i} on `{key}`.")

        source_row_idx = int(tt_row["source_row_idx"])
        if rows[source_row_idx].get("id") != tt_row.get("id"):
            raise ValueError(f"Caption row id mismatch at i={i}.")

    num_layers = int(tt_acts.shape[1])
    hidden_size = int(tt_acts.shape[2])

    # If safety shift has the same hidden size as activations
    if safety_shift.shape[1] != hidden_size:
        raise ValueError(
            f"Safety hidden size mismatch: {safety_shift.shape[1]} vs {hidden_size}."
        )

    # If safety shift and activations have the same number of layers
    if safety_shift.shape[0] != num_layers:
        raise ValueError(
            f"Layer count mismatch: safety_shift has {safety_shift.shape[0]} layers, activations have {num_layers}."
        )

    return num_layers, hidden_size

def _resolve_layer_range(
    num_layers: int,
    layer_start: int | None,
    layer_end: int | None
) -> tuple[int, int]:
    ls = (num_layers // 2) if layer_start is None else layer_start
    le = (num_layers - 1) if layer_end is None else layer_end
    if ls < 0 or le < 0 or ls > le or ls >= num_layers or le >= num_layers:
        raise ValueError(f"Invalid layer range [{ls}, {le}].")

    return ls, le


def run_shiftdc(
    model_name: str,
    input_dir: str,
    data_dir: str,
    safety_shift_npy: str,
    alpha: float,
    layer_start: int | None,
    layer_end: int | None,
    min_token_index: int,
    max_tokens: int,
) -> Path:
    out_dir = Path(input_dir).resolve()
    caption_path = out_dir / "caption.jsonl"
    data_root = Path(data_dir).resolve()
    out_jsonl = out_dir / "shiftdc.jsonl"
    tt_activations_path = out_dir / "tt_activations.npy"
    tt_index_path = out_dir / "tt_index.jsonl"
    vl_activations_path = out_dir / "vl_activations.npy"
    vl_index_path = out_dir / "vl_index.jsonl"

    rows = load_json(str(caption_path))

    tt_acts = np.load(tt_activations_path)
    vl_acts = np.load(vl_activations_path)
    safety_shift = np.load(Path(safety_shift_npy).resolve())

    tt_index = load_json(str(tt_index_path))
    vl_index = load_json(str(vl_index_path))

    num_layers, hidden_size = _validate_inputs(
        safety_shift,
        tt_acts,
        vl_acts,
        tt_index,
        vl_index,
        rows
    )

    ls, le = _resolve_layer_range(
        num_layers=num_layers,
        layer_start=layer_start,
        layer_end=layer_end
    )

    hf = HuggingFace(model_name)
    try:
        layers = _find_decoder_layers(hf.model)
        if len(layers) != num_layers:
            raise ValueError(
                f"Layer count mismatch: activations have {num_layers}, model has {len(layers)}."
            )

        device = next(hf.model.parameters()).device
        safety_vector = ShiftDCSteeringVector.from_layer_matrix(
            torch.from_numpy(safety_shift),
            range(ls, le + 1),
            device=device,
            dtype=torch.float32,
        )

        processed = 0
        with out_jsonl.open("w") as f:
            for i in tqdm(range(len(tt_index)), desc="ShiftDC"):
                idx = tt_index[i]
                source_row_idx = int(idx["source_row_idx"])
                row = rows[source_row_idx]
                prompt_qi = prompt.NORMAL_SAFE.format(
                    question=row["jailbreak_query"].strip()
                )
                image_path = str(idx["image_path"])

                h_qc = torch.from_numpy(tt_acts[i]).to(device=device, dtype=torch.float32)
                h_qi = torch.from_numpy(vl_acts[i]).to(device=device, dtype=torch.float32)

                correction_vector = safety_vector.compute_corrections(
                    h_qi=h_qi,
                    h_qc=h_qc,
                    alpha=alpha
                )

                response = generate_steered_hf(
                    hf=hf,
                    layer_modules=layers,
                    device=device,
                    prompt_qi=prompt_qi,
                    image_path=image_path,
                    steering_vector=correction_vector,
                    min_token_index=min_token_index,
                    max_tokens=max_tokens
                )

                rec = {
                    "id": row.get("id"),
                    "source_row_idx": source_row_idx,
                    "prompt_qi": prompt_qi,
                    "image_path": image_path,
                    "response": response,
                    "alpha": alpha,
                    "layer_start": ls,
                    "layer_end": le,
                    "min_token_index": min_token_index
                }
                for k in ("policy", "image_type", "redteam_query"):
                    if k in row:
                        rec[k] = row[k]

                f.write(json.dumps(rec) + "\n")
                processed += 1
    finally:
        hf.del_model()

    run_meta = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model_name": model_name,
        "backend": "hf",
        "input_dir": str(out_dir),
        "caption_jsonl": str(caption_path),
        "data_dir": str(data_root),
        "tt_activations_npy": str(tt_activations_path),
        "tt_index_jsonl": str(tt_index_path),
        "vl_activations_npy": str(vl_activations_path),
        "vl_index_jsonl": str(vl_index_path),
        "safety_shift_npy": str(Path(safety_shift_npy).resolve()),
        "tt_activation_shape": list(tt_acts.shape),
        "vl_activation_shape": list(vl_acts.shape),
        "safety_shift_shape": list(safety_shift.shape),
        "alpha": alpha,
        "layer_start": ls,
        "layer_end": le,
        "min_token_index": min_token_index,
        "selected_layer_count": len(safety_vector.layer_vectors),
        "max_tokens": max_tokens,
        "processed": processed
    }
    run_meta_path = out_dir / "run_meta.json"
    with run_meta_path.open("w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"Wrote {out_jsonl}")
    print(f"Wrote {run_meta_path}")

    return out_jsonl


if __name__ == "__main__":
    args = parse_args()

    run_shiftdc(
        model_name=args.model_name,
        input_dir=args.input_dir,
        data_dir=args.data_dir,
        safety_shift_npy=args.safety_shift_npy,
        alpha=args.alpha,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        min_token_index=args.min_token_index,
        max_tokens=args.max_tokens
    )
