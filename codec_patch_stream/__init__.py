"""Video decode + patch-stream APIs with GPU/CPU native backends."""

import os

from .api_decode import decode_only
from .api_patch import PatchStreamSession, patch_stream
from .types import DecodeConfig, DecodedFrames, PatchStreamConfig, PatchStreamResult

__all__ = [
    "DecodeConfig",
    "DecodedFrames",
    "PatchStreamConfig",
    "PatchStreamResult",
    "PatchStreamSession",
    "decode_only",
    "patch_stream",
]


def _env_enabled(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "on", "yes"}


def _maybe_eager_warmup_hwctx() -> None:
    # Warm up NVDEC hwctx early to reduce runtime conflicts with other FFmpeg-based libs.
    if not _env_enabled("CODEC_EAGER_WARMUP_HWCTX", True):
        return
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    try:
        device_id = int(os.getenv("CODEC_EAGER_WARMUP_DEVICE", "0"))
    except ValueError:
        device_id = 0
    try:
        from . import _codec_patch_stream_native as _dispatch

        has_backend = getattr(_dispatch, "has_backend", None)
        if callable(has_backend) and not bool(has_backend("gpu")):
            return
    except Exception:
        return
    try:
        from . import _codec_patch_stream_gpu as _gpu

        warmup = getattr(_gpu, "warmup_hw_device_ctx", None)
        if callable(warmup):
            warmup(int(device_id))
    except Exception:
        return


_maybe_eager_warmup_hwctx()
