from abc import ABC, abstractmethod
from typing import Any, Sequence


class BaseVLM(ABC):
    def __init__(self, checkpoint: str):
        if len(self.valid_checkpoints) > 0 and checkpoint not in self.valid_checkpoints:
            raise ValueError(f"Model type {checkpoint} is not supported.")

        self.checkpoint = checkpoint

    @property
    def name(self) -> str:
        return self.checkpoint.split("/")[-1]

    @property
    @abstractmethod
    def valid_checkpoints(self) -> list[str]:
        pass

    @abstractmethod
    def generate(
        self,
        question: str | list[str],
        image_path: str | Sequence[str] | None,
        size: tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        pass
