import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import torch
from tqdm import tqdm

from shiftdc.models import HuggingFace
from shiftdc.utils import (
    add_prefill_hooks,
    get_parent,
    load_json,
    prompt
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model_name", type=str, required=True)
    parser.add_argument("--caption_jsonl", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--tt_activations_npy", type=str, required=True)
    parser.add_argument("--tt_index_jsonl", type=str, required=True)
    parser.add_argument("--vl_activations_npy", type=str, required=True)
    parser.add_argument("--vl_index_jsonl", type=str, required=True)
    parser.add_argument("--safety_shift_npy", type=str, required=True)
    parser.add_argument("--safety_shift_meta", type=str, required=True)
    parser.add_argument("--layer_start", type=int, default=None)
    parser.add_argument("--layer_end", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=128)

    return parser.parse_args()

def _find_decoder_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    # TODO: avoid hardcoding here
    return list(model.language_model.layers)

def _project_vector(m: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Projects vector m onto vector s."""
    denom = torch.dot(s, s)

    if float(denom.item()) <= 0.0:
        return torch.zeros_like(m)

    return (torch.dot(m, s) / denom) * s

def compute_shift_components(
    h_qi: torch.Tensor,
    h_qc: torch.Tensor,
    safety_shift_by_id: dict[int, torch.Tensor],
    alpha: float,
) -> dict[int, torch.Tensor]:
    corrections: dict[int, torch.Tensor] = {}

    for layer_id, s_vec in safety_shift_by_id.items():
        m_vec = h_qi[layer_id] - h_qc[layer_id]
        corrections[layer_id] = alpha * _project_vector(m_vec, s_vec)

    return corrections


def generate_steered_hf(
    hf: HuggingFace,
    layer_modules: list[torch.nn.Module],
    device: torch.device,
    prompt_qi: str,
    image_path: str | None,
    corrections: dict[int, torch.Tensor],
    max_tokens: int,
) -> str:
    inputs_qi = hf._build_generation_inputs(question=prompt_qi, image_path=image_path)
    model_inputs = {k: v.to(device) for k, v in inputs_qi.items()}
    prompt_len = model_inputs["input_ids"].shape[1]

    with torch.no_grad():
        with add_prefill_hooks(layer_modules=layer_modules, corrections=corrections):
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
) -> None:
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

def _build_layer_maps(
    safety_shift: np.ndarray,
    num_layers: int,
    layer_start: int | None,
    layer_end: int | None
) -> tuple[dict[int, int], dict[int, np.ndarray], int, int]:
    ls = (num_layers // 2) if layer_start is None else layer_start
    le = (num_layers - 1) if layer_end is None else layer_end
    if ls < 0 or le < 0 or ls > le or ls >= num_layers or le >= num_layers:
        raise ValueError(f"Invalid layer range [{ls}, {le}].")

    safety_shift_by_id: dict[int, np.ndarray] = {}
    for layer_id in list(range(ls, le + 1)):
        safety_shift_by_id[layer_id] = safety_shift[layer_id]

    return safety_shift_by_id, ls, le


def run_shiftdc(
    model_name: str,
    caption_jsonl: str,
    data_dir: str,
    tt_activations_npy: str,
    tt_index_jsonl: str,
    vl_activations_npy: str,
    vl_index_jsonl: str,
    safety_shift_npy: str,
    safety_shift_meta: str,
    alpha: float,
    layer_start: int | None,
    layer_end: int | None,
    max_tokens: int,
) -> Path:
    caption_path = Path(caption_jsonl).resolve()
    data_root = Path(data_dir).resolve()
    out_dir = Path(get_parent(str(caption_path)))
    out_jsonl = out_dir / "shiftdc3.jsonl"

    rows = load_json(str(caption_path))

    tt_acts = np.load(Path(tt_activations_npy).resolve())
    vl_acts = np.load(Path(vl_activations_npy).resolve())
    safety_shift = np.load(Path(safety_shift_npy).resolve())

    tt_index = load_json(str(Path(tt_index_jsonl).resolve()))
    vl_index = load_json(str(Path(vl_index_jsonl).resolve()))

    num_layers, hidden_size = _validate_inputs(
        safety_shift,
        tt_acts,
        vl_acts,
        tt_index,
        vl_index,
        rows
    )

    safety_shift_by_id_np, ls, le = _build_layer_maps(
        safety_shift=safety_shift,
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
        safety_shift_by_id = {
            layer_id: torch.tensor(vec, dtype=torch.float32, device=device)
            for layer_id, vec in safety_shift_by_id_np.items()
        }

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

                corrections = compute_shift_components(
                    h_qi=h_qi,
                    h_qc=h_qc,
                    safety_shift_by_id=safety_shift_by_id,
                    alpha=alpha
                )

                response = generate_steered_hf(
                    hf=hf,
                    layer_modules=layers,
                    device=device,
                    prompt_qi=prompt_qi,
                    image_path=image_path,
                    corrections=corrections,
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
                    "layer_end": le
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
        "caption_jsonl": str(caption_path),
        "data_dir": str(data_root),
        "tt_activations_npy": str(Path(tt_activations_npy).resolve()),
        "tt_index_jsonl": str(Path(tt_index_jsonl).resolve()),
        "vl_activations_npy": str(Path(vl_activations_npy).resolve()),
        "vl_index_jsonl": str(Path(vl_index_jsonl).resolve()),
        "safety_shift_npy": str(Path(safety_shift_npy).resolve()),
        "safety_shift_meta": str(Path(safety_shift_meta).resolve()),
        "tt_activation_shape": list(tt_acts.shape),
        "vl_activation_shape": list(vl_acts.shape),
        "safety_shift_shape": list(safety_shift.shape),
        "alpha": alpha,
        "layer_start": ls,
        "layer_end": le,
        "selected_layer_count": len(safety_shift_by_id),
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
        caption_jsonl=args.caption_jsonl,
        data_dir=args.data_dir,
        tt_activations_npy=args.tt_activations_npy,
        tt_index_jsonl=args.tt_index_jsonl,
        vl_activations_npy=args.vl_activations_npy,
        vl_index_jsonl=args.vl_index_jsonl,
        safety_shift_npy=args.safety_shift_npy,
        safety_shift_meta=args.safety_shift_meta,
        alpha=args.alpha,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        max_tokens=args.max_tokens
    )
