"""SigLIP2 image and text encoder producing L2-normalised float32 vectors."""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("wikivoyage.images.embedder")

TEXT_MAX_LENGTH = 64  # SigLIP2 was trained with max_length=64 padded text


def pick_device(device: str = "auto") -> str:
    import torch

    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str, dtype: str = "auto"):
    import torch

    table = {"fp32": torch.float32, "float32": torch.float32, "fp16": torch.float16,
             "float16": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}
    if dtype and dtype != "auto":
        return table[dtype.lower()]
    return {"cuda": torch.bfloat16, "mps": torch.float16}.get(device, torch.float32)


def _features(out: Any):
    """``get_*_features`` returns a tensor in older transformers and an output object in newer ones."""
    import torch

    if isinstance(out, torch.Tensor):
        return out
    for attr in ("pooler_output", "image_embeds", "text_embeds"):
        v = getattr(out, attr, None)
        if isinstance(v, torch.Tensor):
            return v
    raise TypeError(f"unexpected feature output type: {type(out).__name__}")


def _l2(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.clip(norms, 1e-12, None)).astype(np.float32)


class SiglipImageEmbedder:
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        device: str = "auto",
        dtype: str = "auto",
        hf_token: Optional[str] = None,
    ) -> None:
        import torch
        import transformers
        from transformers import AutoModel, AutoProcessor

        self.model_id = model_id
        self.device = pick_device(device)
        self.torch_dtype = pick_dtype(self.device, dtype)
        logger.info("loading %s (revision=%s) on %s/%s", model_id, revision or "main", self.device, self.torch_dtype)
        kw = {"revision": revision, "token": hf_token}
        try:
            self.model = AutoModel.from_pretrained(model_id, dtype=self.torch_dtype, **kw)
        except TypeError:  # transformers < 4.56 only knows torch_dtype
            self.model = AutoModel.from_pretrained(model_id, torch_dtype=self.torch_dtype, **kw)
        self.model.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, **kw)
        self.model_revision = (
            getattr(self.model.config, "_commit_hash", None) or revision or "unknown"
        )
        vision_cfg = getattr(self.model.config, "vision_config", None)
        self.dim = int(getattr(vision_cfg, "hidden_size", 0) or 0)
        self.image_size = getattr(vision_cfg, "image_size", None)
        self.info = {
            "model_id": model_id,
            "model_revision": self.model_revision,
            "dim": self.dim,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "device": self.device,
            "dtype": str(self.torch_dtype).replace("torch.", ""),
            "image_size": self.image_size,
        }

    def _run(self, inputs, fn) -> np.ndarray:
        import torch

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.torch_dtype)
        with torch.inference_mode():
            feats = _features(fn(**inputs))
        arr = _l2(feats.float().cpu().numpy())
        if self.dim and arr.shape[1] != self.dim:
            logger.warning("feature dim %d differs from config %d; using %d", arr.shape[1], self.dim, arr.shape[1])
        self.dim = arr.shape[1]
        self.info["dim"] = self.dim
        return arr

    def embed_images(self, images: list) -> np.ndarray:
        """(n, dim) float32, L2-normalised. ``images`` are RGB PIL images."""
        inputs = self.processor(images=images, return_tensors="pt")
        return self._run(inputs, self.model.get_image_features)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(
            text=texts, padding="max_length", max_length=TEXT_MAX_LENGTH, truncation=True, return_tensors="pt"
        )
        return self._run(inputs, self.model.get_text_features)
