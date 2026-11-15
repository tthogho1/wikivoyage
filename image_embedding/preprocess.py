"""Image validation and normalisation before embedding (Pillow)."""
from __future__ import annotations

import io
from typing import Optional

from PIL import Image, ImageOps

JPEG_QUALITY = 95


class ImageRejected(Exception):
    """An image that must not be embedded. ``status`` is ``failed`` or ``skipped``."""

    def __init__(self, reason: str, status: str = "skipped") -> None:
        super().__init__(reason)
        self.reason = reason
        self.status = status


def load_and_validate(data: bytes, *, min_side: int = 100, max_side: int = 8192) -> Image.Image:
    """Decode bytes, rejecting empty, corrupt, animated, too small and too large images."""
    if not data:
        raise ImageRejected("zero_bytes", "failed")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        img = Image.open(io.BytesIO(data))
        img.load()
    except Image.DecompressionBombError:
        raise ImageRejected("decompression_bomb", "skipped")
    except Exception:
        raise ImageRejected("corrupt", "failed")
    if getattr(img, "is_animated", False) and getattr(img, "n_frames", 1) > 1:
        raise ImageRejected("animated", "skipped")
    w, h = img.size
    if w < min_side or h < min_side:
        raise ImageRejected(f"too_small:{w}x{h}", "skipped")
    if w > max_side or h > max_side:
        raise ImageRejected(f"too_large:{w}x{h}", "skipped")
    return img


def _has_alpha(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)


def to_rgb(img: Image.Image, *, pad_aspect: Optional[float] = 2.0) -> tuple[Image.Image, list[str]]:
    """EXIF-rotate, flatten transparency onto white, convert to RGB, pad extreme aspect ratios."""
    ops: list[str] = []
    try:
        orientation = img.getexif().get(0x0112, 1)
    except Exception:
        orientation = 1
    if orientation not in (None, 1):
        img = ImageOps.exif_transpose(img)
        ops.append("exif_transpose")
    if _has_alpha(img):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        img = bg
        ops.append("rgba_on_white")
    elif img.mode != "RGB":
        if img.mode.startswith("I"):  # 16/32-bit greyscale: scale into 8 bits first
            img = img.convert("I").point(lambda v: v * (1 / 256)).convert("L")
        img = img.convert("RGB")
        ops.append("convert_rgb")
    w, h = img.size
    if pad_aspect and min(w, h) > 0 and max(w, h) / min(w, h) >= pad_aspect:
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        img = canvas
        ops.append("pad_square")
    return img, ops


def encode_jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()
