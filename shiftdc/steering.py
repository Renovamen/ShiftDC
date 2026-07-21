from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle


@dataclass
class ShiftDCSteeringHandle:
    """Owns the forward hooks installed for one steering application."""

    hooks: list[RemovableHandle]

    def remove(self) -> None:
        """Remove every installed hook."""
        for hook in self.hooks:
            hook.remove()


@dataclass(frozen=True)
class ShiftDCSteeringVector:
    """Applys ShiftDC per-layer."""

    layer_vectors: dict[int, Tensor]

    def __post_init__(self) -> None:
        for layer_idx, vector in self.layer_vectors.items():
            if layer_idx < 0:
                raise ValueError(f"Layer index must be non-negative, got {layer_idx}.")

            if not torch.is_tensor(vector) or vector.ndim != 1:
                shape = tuple(vector.shape) if torch.is_tensor(vector) else None
                raise ValueError(
                    f"Steering vector at layer {layer_idx} must be 1D, got shape={shape}."
                )

    @classmethod
    def from_layer_matrix(
        cls,
        layer_matrix: Tensor,
        layer_indices: Iterable[int],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "ShiftDCSteeringVector":
        """Build per-layer vectors from a ``[num_layers, hidden_size]`` matrix."""
        if layer_matrix.ndim != 2:
            raise ValueError(
                "Steering matrix must be [num_layers, hidden_size], "
                f"got shape={tuple(layer_matrix.shape)}."
            )

        vectors: dict[int, Tensor] = {}
        for layer_idx in layer_indices:
            if layer_idx < 0 or layer_idx >= layer_matrix.shape[0]:
                raise ValueError(
                    f"Layer {layer_idx} out of bounds for steering matrix with "
                    f"{layer_matrix.shape[0]} layers."
                )

            vector = layer_matrix[layer_idx]
            if device is not None or dtype is not None:
                vector = vector.to(device=device, dtype=dtype)
            vectors[layer_idx] = vector
        return cls(vectors)

    def compute_corrections(
        self,
        h_qi: Tensor,
        h_qc: Tensor,
        alpha: float,
    ) -> "ShiftDCSteeringVector":
        """Project ``h_qi - h_qc`` onto this safety vector at each selected layer."""
        if h_qi.ndim != 2 or h_qc.ndim != 2 or h_qi.shape != h_qc.shape:
            raise ValueError(
                "h_qi and h_qc must have identical [num_layers, hidden_size] shapes, "
                f"got {tuple(h_qi.shape)} and {tuple(h_qc.shape)}."
            )

        corrections: dict[int, Tensor] = {}
        for layer_idx, safety_vector in self.layer_vectors.items():
            if layer_idx >= h_qi.shape[0]:
                raise ValueError(
                    f"Layer {layer_idx} out of bounds for activations with "
                    f"{h_qi.shape[0]} layers."
                )
            if safety_vector.numel() != h_qi.shape[1]:
                raise ValueError(
                    f"Hidden size mismatch at layer {layer_idx}: steering vector has "
                    f"{safety_vector.numel()}, activations have {h_qi.shape[1]}."
                )

            m_vector = h_qi[layer_idx] - h_qc[layer_idx]
            corrections[layer_idx] = alpha * self.project_vector(
                m_vector, safety_vector
            )

        return ShiftDCSteeringVector(corrections)

    @staticmethod
    def project_vector(m_vector: Tensor, safety_vector: Tensor) -> Tensor:
        """Projects vector m onto vector s."""
        if m_vector.ndim != 1 or safety_vector.ndim != 1:
            raise ValueError(
                "Projection inputs must both be 1D, got "
                f"{tuple(m_vector.shape)} and {tuple(safety_vector.shape)}."
            )
        if m_vector.numel() != safety_vector.numel():
            raise ValueError(
                "Projection hidden size mismatch: "
                f"{m_vector.numel()} vs {safety_vector.numel()}."
            )

        safety_vector = safety_vector.to(m_vector.device, dtype=m_vector.dtype)
        denominator = torch.dot(safety_vector, safety_vector)

        if float(denominator.item()) <= 0.0:
            return torch.zeros_like(m_vector)

        return (torch.dot(m_vector, safety_vector) / denominator) * safety_vector

    def patch_activations(
        self,
        layer_modules: list[nn.Module],
        min_token_index: int,
    ) -> ShiftDCSteeringHandle:
        """Install subtractive ShiftDC hooks and return a handle for removing them."""
        hooks: list[RemovableHandle] = []

        try:
            for layer_idx, correction in self.layer_vectors.items():
                if layer_idx >= len(layer_modules):
                    raise ValueError(
                        f"Layer {layer_idx} out of bounds for model with "
                        f"{len(layer_modules)} layers."
                    )
                hooks.append(
                    layer_modules[layer_idx].register_forward_hook(
                        _create_shiftdc_hook(correction, min_token_index)
                    )
                )
        except Exception:
            for hook in hooks:
                hook.remove()
            raise

        return ShiftDCSteeringHandle(hooks)

    @contextmanager
    def apply(
        self,
        layer_modules: list[nn.Module],
        min_token_index: int,
    ) -> Generator[None, None, None]:
        """Apply this vector for the duration of a context-managed scope."""
        handle = self.patch_activations(layer_modules, min_token_index)
        try:
            yield
        finally:
            handle.remove()

def _create_shiftdc_hook(correction: Tensor, min_token_index: int):
    """Apply the correction to tokens selected by `min_token_index`.

    During prefill with tokens [t0, t1, t2, t3]:
      min_token_index=0  -> patch [t0, t1, t2, t3]
      min_token_index=-1 -> patch [t3]

    During cached generation each forward usually contains only the
    new token, e.g. [g0], [g1], ..., so:
      min_token_index=0/-1/-2/..  -> patch [g0], [g1], ...
      min_token_index=1/2/.. -> no patch
    """

    def hook_fn(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            return output

        if correction.numel() != hidden.shape[-1]:
            raise ValueError(
                "Correction hidden size does not match hooked output: "
                f"{correction.numel()} vs {hidden.shape[-1]}."
            )

        shift = correction.to(hidden.device, dtype=hidden.dtype).reshape(1, 1, -1)
        patched_hidden = hidden.clone()
        patched_hidden[:, min_token_index:, :] -= shift

        if isinstance(output, tuple):
            return (patched_hidden, *output[1:])
        return patched_hidden

    return hook_fn
