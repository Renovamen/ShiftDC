from typing import Any, Sequence

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

from shiftdc.utils import load_images
from .base import BaseVLM


class HuggingFace(BaseVLM):
    def __init__(
        self,
        checkpoint: str,
        **_: Any,
    ):
        super().__init__(checkpoint)

        self.processor = AutoProcessor.from_pretrained(
            checkpoint,
            trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer is None:
            raise ValueError(f"AutoProcessor for {checkpoint} does not expose a tokenizer.")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = LlavaForConditionalGeneration.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto"
        ).eval()

    @property
    def valid_checkpoints(self) -> list[str]:
        return []

    @property
    def num_hidden_layers(self) -> int:
        config = self.model.config

        num_layers = getattr(
            getattr(config, "text_config", None),
            "num_hidden_layers",
            getattr(config, "num_hidden_layers", None)
        )

        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError("Could not infer `num_hidden_layers` from loaded config.")

        return num_layers

    def del_model(self):
        if hasattr(self, "model") and self.model is not None:
            del self.model

    def _normalize_batch_inputs(
        self,
        question: str | list[str],
        image_path: str | Sequence[str] | None,
    ) -> tuple[list[str], list[str | None]]:
        questions = [question] if isinstance(question, str) else list(question)

        if image_path is None:
            image_paths = [None] * len(questions)
        elif isinstance(image_path, str):
            image_paths = [image_path] * len(questions)
        else:
            image_paths = list(image_path)
            if len(image_paths) != len(questions):
                raise ValueError("`image_path` and `question` must have the same length.")

        return questions, image_paths

    def _build_generation_inputs(
        self,
        question: str,
        image_path: str | None,
        size: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        conversation = [
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": question.replace("<image>", "").strip()
                }]
            }
        ]

        image = None
        if image_path is not None:
            image = load_images(image_path, size=size)
            conversation[0]["content"].append({"type": "image"})

        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )

        return self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )

    def _build_generation_batch_inputs(
        self,
        questions: Sequence[str],
        image_paths: Sequence[str],
        size: tuple[int, int] | None = None,
        max_model_len: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if len(questions) != len(image_paths):
            raise ValueError("`questions` and `image_paths` must have the same length.")
        if len(questions) == 0:
            raise ValueError("Cannot build an empty multimodal batch.")

        prompts: list[str] = []
        for question in questions:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question.replace("<image>", "").strip()
                        },
                        {
                            "type": "image"
                        }
                    ]
                }
            ]
            prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True
            )
            prompts.append(prompt)

        images = load_images(list(image_paths), size=size)
        kwargs: dict[str, Any] = {
            "text": prompts,
            "images": images,
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
        }
        if max_model_len is not None:
            kwargs["max_length"] = max_model_len

        return self.processor(**kwargs)

    @torch.no_grad()
    def generate(
        self,
        question: str | list[str],
        image_path: str | Sequence[str] | None,
        size: tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        questions, image_paths = self._normalize_batch_inputs(question, image_path)
        device = next(self.model.parameters()).device

        max_new_tokens = int(kwargs.pop("max_new_tokens", kwargs.pop("max_tokens", 128)))

        completions: list[dict[str, Any]] = []
        for q, p in zip(questions, image_paths):
            inputs = self._build_generation_inputs(q, p, size=size)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            prompt_len = int(inputs["input_ids"].shape[1])

            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
            response = self.tokenizer.decode(
                output_ids[0][prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            ).strip()

            completions.append(
                {
                    "prompt": q,
                    "image_path": p,
                    "response": response
                }
            )
        return completions

    @torch.no_grad()
    def extract_last_token_activations(
        self,
        prompts: list[str],
        batch_size: int,
        max_model_len: int,
        desc: str,
        image_paths: Sequence[str | None] | None = None,
        size: tuple[int, int] | None = None,
        show_progress: bool = True,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        running_sum: np.ndarray | None = None
        count = 0
        all_activations: list[np.ndarray] = []

        device = next(self.model.parameters()).device
        prompts, normalized_image_paths = self._normalize_batch_inputs(prompts, image_paths)

        has_images = any(p is not None for p in normalized_image_paths)
        total_batches = (len(prompts) + batch_size - 1) // batch_size

        prompt_batches = range(0, len(prompts), batch_size)
        iterator = prompt_batches
        if show_progress:
            iterator = tqdm(prompt_batches, total=total_batches, desc=desc)

        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, len(prompts))
            batch_prompts = prompts[start_idx:end_idx]
            batch_images = normalized_image_paths[start_idx:end_idx]

            if has_images and all(img is not None for img in batch_images):
                batch = self._build_generation_batch_inputs(
                    questions=batch_prompts,
                    image_paths=[str(p) for p in batch_images],
                    size=size,
                    max_model_len=max_model_len,
                )
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

                outputs = self.model(
                    **batch,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True
                )
                hidden_states = outputs.hidden_states
                if hidden_states is None or len(hidden_states) < 2:
                    raise ValueError("Model did not return layer hidden states.")

                hs = torch.stack(hidden_states[1:], dim=0)  # [num_layers, batch_size, seq_len, hidden_size]
                if hs.shape[0] != self.num_hidden_layers:
                    raise ValueError(
                        f"Hidden layer count mismatch: got {hs.shape[0]}, expected {self.num_hidden_layers}."
                    )

                attn_mask = batch["attention_mask"]
                last_token_idx = attn_mask.sum(dim=1) - 1
                for b in range(hs.shape[1]):
                    token_pos = int(last_token_idx[b].item())
                    last_token = hs[:, b, token_pos, :].float().cpu().numpy()

                    if running_sum is None:
                        running_sum = np.zeros_like(last_token, dtype=np.float64)
                    running_sum += last_token

                    all_activations.append(last_token.astype(np.float32, copy=False))
                    count += 1
            elif has_images:
                for prompt, image_path in zip(batch_prompts, batch_images):
                    batch = self._build_generation_inputs(prompt, image_path, size=size)
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

                    outputs = self.model(
                        **batch,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True
                    )
                    hidden_states = outputs.hidden_states
                    if hidden_states is None or len(hidden_states) < 2:
                        raise ValueError("Model did not return layer hidden states.")

                    hs = torch.stack(hidden_states[1:], dim=0)  # [num_layers, 1, seq_len, hidden_size]
                    if hs.shape[0] != self.num_hidden_layers:
                        raise ValueError(
                            f"Hidden layer count mismatch: got {hs.shape[0]}, expected {self.num_hidden_layers}."
                        )

                    token_pos = int((batch["attention_mask"].sum(dim=1) - 1)[0].item())
                    last_token = hs[:, 0, token_pos, :].float().cpu().numpy()

                    if running_sum is None:
                        running_sum = np.zeros_like(last_token, dtype=np.float64)
                    running_sum += last_token

                    all_activations.append(last_token.astype(np.float32, copy=False))
                    count += 1
            else:
                batch = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_model_len
                )
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

                outputs = self.model(
                    **batch,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True
                )
                hidden_states = outputs.hidden_states
                if hidden_states is None or len(hidden_states) < 2:
                    raise ValueError("Model did not return layer hidden states.")

                hs = torch.stack(hidden_states[1:], dim=0)  # [num_layers, batch_size, seq_len, hidden_size]
                if hs.shape[0] != self.num_hidden_layers:
                    raise ValueError(
                        f"Hidden layer count mismatch: got {hs.shape[0]}, expected {self.num_hidden_layers}."
                    )

                attn_mask = batch["attention_mask"]
                last_token_idx = attn_mask.sum(dim=1) - 1

                for b in range(hs.shape[1]):
                    token_pos = int(last_token_idx[b].item())
                    last_token = hs[:, b, token_pos, :].float().cpu().numpy()

                    if running_sum is None:
                        running_sum = np.zeros_like(last_token, dtype=np.float64)
                    running_sum += last_token

                    all_activations.append(last_token.astype(np.float32, copy=False))
                    count += 1

        if running_sum is None or count == 0:
            raise ValueError("No activations extracted.")

        return running_sum, count, np.stack(all_activations, axis=0)
