from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .native_backend import load_native_backend

if TYPE_CHECKING:
    import numpy as np


@dataclass
class DecodedFrames:
    frames: torch.Tensor
    sampled_frame_ids: list[int]
    fps: float
    duration_sec: float
    width: int
    height: int

    def asnumpy(self) -> "np.ndarray":
        return self.frames.detach().cpu().numpy()


def decode_uniform_frames(
    video_path: str | Path,
    num_frames: int = 16,
    *,
    backend: str = "auto",
    device_id: int = 0,
    mode: str = "throughput",
) -> DecodedFrames:
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")

    backend_key = str(backend).strip().lower()
    if backend_key == "auto":
        if not torch.cuda.is_available():
            native = load_native_backend("cpu")
            if not hasattr(native, "decode_uniform_frames"):
                raise RuntimeError(
                    "cpu backend does not expose decode_uniform_frames; "
                    "rebuild codec-patch-stream with CODEC_BUILD_NATIVE=1"
                )
            out: Any = native.decode_uniform_frames(
                video_path=str(video_path),
                sequence_length=int(num_frames),
                device_id=int(device_id),
                mode=str(mode),
            )
            return DecodedFrames(
                frames=out["frames"],
                sampled_frame_ids=[int(x) for x in out["sampled_frame_ids"]],
                fps=float(out["fps"]),
                duration_sec=float(out["duration_sec"]),
                width=int(out["width"]),
                height=int(out["height"]),
            )
        try:
            native = load_native_backend("gpu")
        except Exception:
            native = load_native_backend("cpu")
        else:
            if not hasattr(native, "decode_uniform_frames"):
                raise RuntimeError(
                    "GPU backend is available but does not expose decode_uniform_frames; "
                    "rebuild codec-patch-stream or use backend='cpu'"
                )
    else:
        native = load_native_backend(backend_key)
        if not hasattr(native, "decode_uniform_frames"):
            raise RuntimeError(
                f"{backend_key} backend does not expose decode_uniform_frames; "
                "rebuild codec-patch-stream with CODEC_BUILD_NATIVE=1"
            )

    out: Any = native.decode_uniform_frames(
        video_path=str(video_path),
        sequence_length=int(num_frames),
        device_id=int(device_id),
        mode=str(mode),
    )

    return DecodedFrames(
        frames=out["frames"],
        sampled_frame_ids=[int(x) for x in out["sampled_frame_ids"]],
        fps=float(out["fps"]),
        duration_sec=float(out["duration_sec"]),
        width=int(out["width"]),
        height=int(out["height"]),
    )
