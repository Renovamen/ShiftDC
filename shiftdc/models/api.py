import os
from typing import Any, Literal, Sequence

from shiftdc.utils import load_images, retry
from .base import BaseVLM


class OpenAI(BaseVLM):
    """OpenAI SDK client"""

    def __init__(
        self,
        checkpoint: str,
        key: str | None = None,
        **_: Any
    ):
        super().__init__(checkpoint)
        self.key = key or os.getenv("OPENAI_API_KEY")
        self._client = None

    @property
    def valid_checkpoints(self) -> list[str]:
        return ["gpt-4o-2024-11-20", "gpt-5"]

    def _load_client(self):
        if self._client is None:
            if not self.key:
                raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY or pass `key=`.")
            try:
                from openai import OpenAI
            except Exception as e:
                raise ImportError("OpenAI SDK is not installed. Please run `pip install openai`." ) from e

            self._client = OpenAI(api_key=self.key)

    def _build_inputs(
        self,
        question: str,
        image_path: str | Sequence[str] | None,
        size: tuple[int, int] | None = None
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": question}]

        if image_path is not None:
            images = [image_path] if isinstance(image_path, str) else list(image_path)
            for path in images:
                content.append(
                    {
                        "type": "input_image",
                        "image_url": load_images(path, fmt="base64", size=size),
                    }
                )

        return [{"role": "user", "content": content}]

    @property
    def _default_sampling_config(self):
        # Responses API uses `max_output_tokens`.
        return {
            "max_output_tokens": 1024,
            "temperature": 0.0,
            "top_p": 1.0,
        }

    def _build_sampling_config(self, **kwargs) -> dict[str, Any]:
        config = dict(self._default_sampling_config)
        config.update(kwargs)
        return config

    @retry(max_retries=3)
    def _send_request(self, messages: Any, **kwargs):
        self._load_client()

        resp = self._client.responses.create(
            model=self.checkpoint,
            input=messages,
            **self._build_sampling_config(**kwargs)
        )

        return resp

    def generate(
        self,
        question: str,
        image_path: str | Sequence[str] | None,
        size: tuple[int, int] | None = None,
        **kwargs,
    ) -> list[dict[Literal["prompt", "image_path", "response"], str]]:
        messages = self._build_inputs(question, image_path, size=size)

        completion = {
            "prompt": question,
            "image_path": image_path
        }

        try:
            resp = self._send_request(messages, **kwargs)
            completion["response"] = resp.output_text.strip()
        except Exception as e:
            completion["response"] = 123
        return [completion]
