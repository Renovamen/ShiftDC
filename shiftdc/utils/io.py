from typing import Literal
from pathlib import Path
from PIL import Image
import io
import os
import json
import base64
import mimetypes

def _load_image(
    image_path: str,
    fmt: Literal["image", "base64"],
    size: tuple[int, int] | None = None
) -> str | Image.Image:
    image = Image.open(image_path).convert("RGB")

    if size:
        image = image.resize(size)

    if fmt == "image":
        return image

    mime_type, _ = mimetypes.guess_type(image_path)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG" if mime_type == "image/jpeg" else "PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"

def load_images(
    image_paths: str | list[str],
    fmt: Literal["image", "base64"] = "image",
    size: tuple[int, int] | None = None
):
    return _load_image(image_paths, fmt=fmt, size=size) if isinstance(image_paths, str) else [
        _load_image(p, fmt=fmt, size=size)
        for p in image_paths
    ]

def load_json(path: str):
    with open(path, "r") as f:
        if str(path).endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)

def make_dir(path: str):
    if os.path.splitext(path)[1]:  # path has a file extension
        os.makedirs(os.path.dirname(path), exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)

def get_parent(path: str):
    return str(Path(path).resolve().parent)
