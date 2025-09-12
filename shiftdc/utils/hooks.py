import time
import random
from functools import wraps
from contextlib import contextmanager
from typing import Any, Callable

import torch

def retry(max_retries: int = 3, base_delay: float = 5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise e

                    sleep_time = base_delay * (2 ** attempt) * random.uniform(0.8, 1.2)

                    print(f"Retry {attempt + 1}/{max_retries}: {e}. Waiting {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
        return wrapper
    return decorator

@contextmanager
def time_counter(enable: bool):
    if enable:
        start_time = time.time()
        yield
        elapsed_time = time.time() - start_time

        print("-" * 50)
        print("-- time = {}".format(elapsed_time))
        print("-" * 50)
    else:
        yield


# https://github.com/andyrdt/refusal_direction/blob/main/pipeline/utils/hook_utils.py
def get_shiftdc_hook(
    layer_idx: int,
    correction: torch.Tensor,
    applied: dict[int, bool],
) -> Callable[[torch.nn.Module, tuple[Any, ...], Any], Any]:
    def _hook(
        _module: torch.nn.Module,
        _inputs: tuple[Any, ...],
        output: Any
    ):
        if applied[layer_idx]:
            return output

        hidden = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            return output

        shift = correction.to(hidden.device, dtype=hidden.dtype)
        hidden[:, -1, :] = hidden[:, -1, :] - shift

        applied[layer_idx] = True

        if isinstance(output, tuple):
            return (hidden, *output[1:])
        return hidden

    return _hook


@contextmanager
def add_prefill_hooks(
    layer_modules: list[torch.nn.Module],
    corrections: dict[int, torch.Tensor],
):
    handles = []
    applied = {layer_idx: False for layer_idx in corrections}

    try:
        for layer_idx, correction in corrections.items():
            if layer_idx < 0 or layer_idx >= len(layer_modules):
                raise ValueError(
                    f"Layer {layer_idx} out of bounds for model with {len(layer_modules)} layers."
                )

            hook_fn = get_shiftdc_hook(
                layer_idx=layer_idx,
                correction=correction,
                applied=applied
            )
            handles.append(layer_modules[layer_idx].register_forward_hook(hook_fn))
        yield
    finally:
        for handle in handles:
            handle.remove()
