from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .native_backend import load_native_backend
from .types import DecodedFrames


def decode_uniform_frames_core(
    video_path: str | Path,
    sequence_length: int,
    *,
    backend: str,
    device_id: int,
    decode_mode: str,
    uniform_strategy: str,
    nvdec_session_pool_size: int | None,
    uniform_auto_ratio: int | None,
    decode_threads: int | None,
    decode_thread_type: str | None,
    reader_cache_size: int | None,
    nvdec_reuse_open_decoder: bool | None,
) -> DecodedFrames:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be > 0")

    backend_key = str(backend).strip().lower()
    if backend_key == "auto" and not torch.cuda.is_available():
        backend_key = "cpu"
    native = load_native_backend(backend_key)

    decode_fn = None
    use_decode_only_native = False
    if hasattr(native, "decode_only_native"):
        decode_fn = native.decode_only_native
        use_decode_only_native = True
    elif hasattr(native, "decode_uniform_frames"):
        decode_fn = native.decode_uniform_frames
    else:
        raise RuntimeError(
            f"{backend_key} backend does not expose decode_only_native/decode_uniform_frames; "
            "rebuild codec-patch-stream with CODEC_BUILD_NATIVE=1"
        )

    if use_decode_only_native:
        out: Any = decode_fn(
            video_path=str(video_path),
            sequence_length=int(sequence_length),
            backend=str(backend_key),
            device_id=int(device_id),
            mode=str(decode_mode),
            uniform_strategy=str(uniform_strategy),
            nvdec_session_pool_size=-1
            if nvdec_session_pool_size is None
            else int(nvdec_session_pool_size),
            uniform_auto_ratio=-1 if uniform_auto_ratio is None else int(uniform_auto_ratio),
            decode_threads=-1 if decode_threads is None else int(decode_threads),
            decode_thread_type=""
            if decode_thread_type is None
            else str(decode_thread_type),
            reader_cache_size=-1 if reader_cache_size is None else int(reader_cache_size),
            nvdec_reuse_open_decoder=-1
            if nvdec_reuse_open_decoder is None
            else int(bool(nvdec_reuse_open_decoder)),
        )
    else:
        out = decode_fn(
            video_path=str(video_path),
            sequence_length=int(sequence_length),
            device_id=int(device_id),
            mode=str(decode_mode),
        )
    return DecodedFrames(
        frames=out["frames"],
        sampled_frame_ids=[int(x) for x in out["sampled_frame_ids"]],
        fps=float(out["fps"]),
        duration_sec=float(out["duration_sec"]),
        width=int(out["width"]),
        height=int(out["height"]),
    )
