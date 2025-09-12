
import os
from vllm import LLM, SamplingParams
from abc import abstractmethod
from typing import Any, Sequence

from shiftdc.utils import load_images
from .base import BaseVLM


class vLLM(BaseVLM):
    def __init__(
        self,
        checkpoint: str,
        image: bool = True,
        devices: Sequence[int] | None = None,
        **kwargs
    ):
        super().__init__(checkpoint)
        self.image = image

        engine_kwargs = dict(kwargs)

        self._visible_devices = self._configure_devices(devices)

        if "tensor_parallel_size" not in engine_kwargs and self._visible_devices:
            engine_kwargs["tensor_parallel_size"] = len(self._visible_devices)

        self.engine_kwargs = self._build_engine_kwargs(**engine_kwargs)
        self.model = LLM(**self.engine_kwargs)

    def del_model(self):
        if hasattr(self, "model") and self.model is not None:
            del self.model

    @property
    @abstractmethod
    def valid_checkpoints(self) -> list[str]:
        pass

    @abstractmethod
    def _build_engine_kwargs(self, **kwargs) -> dict[str, Any]:
        pass

    @property
    def _default_sampling_config(self):
        return dict(
            max_tokens=1024,
            temperature=0.0,
            top_p=1.0,
            top_k=-1
            # min_tokens=50
        )

    def _build_sampling_params(self, **kwargs) -> SamplingParams:
        config = dict(self._default_sampling_config)
        config.update(kwargs)
        return SamplingParams(**config)

    def _configure_devices(self, devices: Sequence[int] | None) -> list[int] | None:
        if not devices:
            return None

        device_ids = [int(device) for device in devices]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in device_ids)

        return device_ids

    def _single_turn_conversation(
        self,
        question: str,
        image: Any | None,
        with_image: bool,
        *,
        system_prompt: str | None = None
    ) -> list[dict[str, Any]]:
        conversation: list[dict[str, Any]] = []

        if system_prompt:
            conversation.append({
                "role": "system",
                "content": system_prompt
            })

        content: list[dict[str, Any]] = []
        if with_image:
            content.append({"type": "image_pil", "image_pil": image})
        content.append({"type": "text", "text": question})

        conversation.append({"role": "user", "content": content})

        return conversation

    def _build_conversations(
        self,
        questions: list[str],
        images: list[Any | None],
        with_image: list[bool]
    ) -> list[list[dict[str, Any]]]:
        conversations = [
            self._single_turn_conversation(q, image=img, with_image=img_flag)
            for q, img, img_flag in zip(questions, images, with_image)
        ]
        return conversations

    def _generate_kwargs(self) -> dict[str, Any]:
        return dict(use_tqdm=False)

    @property
    def _image_limit_mm(self):
        """For disabling other modalities to save memory."""
        return dict(image=int(self.image), video=0, audio=0)

    def generate(
        self,
        question: list[str] | str,
        image_path: str | list[str] | None,
        size: tuple[int, int] | None = None,
        **kwargs
    ) -> list[dict[str, Any]]:
        questions = [question] if isinstance(question, str) else question
        if not questions:
            return []

        if image_path is None:
            image_paths = [None] * len(questions)
        elif isinstance(image_path, str):
            image_paths = [image_path] * len(questions)
        else:
            image_paths = list(image_path)
            if len(image_paths) != len(questions):
                raise ValueError(
                    "`image_path` and `question` must have the same length when batched."
                )

        has_images = [path is not None for path in image_paths]
        if not self.image and any(has_images):
            raise ValueError("This model does not accept images, but image paths were provided.")

        images = [
            load_images(path, size=size) if path is not None else None
            for path in image_paths
        ]
        conversations = self._build_conversations(questions, images, has_images)

        sampling_params = self._build_sampling_params(**kwargs)

        outputs = self.model.chat(
            conversations,
            sampling_params=sampling_params,
            **self._generate_kwargs()
        )

        completions = []
        for output, path in zip(outputs, image_paths):
            candidate = output.outputs[0]
            completion = {
                "prompt": getattr(output, "prompt", None),
                "image_path": path,
                "response": candidate.text
            }
            completions.append(completion)

        return completions


class LLaVA(vLLM):
    """LLaVA-1.5, LLaVA-1.6/LLaVA-NeXT"""

    @property
    def valid_checkpoints(self):
        return [
            "llava-hf/llava-1.5-7b-hf",
            "llava-hf/llava-1.5-13b-hf",
            "llava-hf/llava-v1.6-34b-hf"
        ]

    def _build_engine_kwargs(self, **kwargs) -> dict[str, Any]:
        return dict(
            model=self.checkpoint,
            limit_mm_per_prompt=self._image_limit_mm,
            max_model_len=1024,
            **kwargs
        )


class QwenVL(vLLM):
    """Qwen/Qwen-VL-Chat"""

    @property
    def valid_checkpoints(self):
        return ["Qwen/Qwen-VL-Chat"]

    def _build_engine_kwargs(self, **kwargs) -> dict[str, Any]:
        return dict(
            model=self.checkpoint,
            limit_mm_per_prompt=self._image_limit_mm,
            trust_remote_code=True,
            max_model_len=1024,
            max_num_seqs=2,
            **kwargs
        )
