from .base import BaseVLM
from .vllm import vLLM, LLaVA
from .api import OpenAI
from .hf import HuggingFace

VLLM_MODEL_MAP = {
    "llava-hf/llava-1.5-7b-hf": LLaVA,
    "llava-hf/llava-1.5-13b-hf": LLaVA,
    "llava-hf/llava-v1.6-34b-hf": LLaVA
}

API_MODEL_MAP = {
    "gpt-4o-2024-11-20": OpenAI,
    "gpt-5": OpenAI,
}

SUPPORTED_MODELS = list(VLLM_MODEL_MAP.keys()) + list(API_MODEL_MAP.keys())

def build_vlm(checkpoint: str, image: bool = True, **kwargs) -> BaseVLM:
    if checkpoint in VLLM_MODEL_MAP:
        cls = VLLM_MODEL_MAP[checkpoint]
        return cls(checkpoint, image, **kwargs)

    if checkpoint in API_MODEL_MAP:
        cls = API_MODEL_MAP[checkpoint]
        return cls(checkpoint, **kwargs)

    raise ValueError(f"Model type {checkpoint} is not supported.")
