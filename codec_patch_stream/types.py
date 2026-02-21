from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch


@dataclass
class DecodedFrames:
    frames: torch.Tensor
    sampled_frame_ids: List[int]
    fps: float
    duration_sec: float
    width: int
    height: int

    def asnumpy(self):
        return self.frames.detach().cpu().numpy()


@dataclass(frozen=True)
class DecodeConfig:
    video_path: str | Path
    sequence_length: int = 16
    backend: str = "auto"
    device_id: int = 0
    decode_mode: str = "throughput"
    uniform_strategy: str = "auto"
    nvdec_session_pool_size: int | None = None
    uniform_auto_ratio: int | None = None
    decode_threads: int | None = None
    decode_thread_type: str | None = None
    reader_cache_size: int | None = None
    nvdec_reuse_open_decoder: bool | None = None


@dataclass(frozen=True)
class PatchStreamConfig:
    video_path: str | Path
    sequence_length: int = 16
    decode_mode: str = "throughput"
    uniform_strategy: str = "auto"
    input_size: int = 224
    min_pixels: int | None = None
    max_pixels: int | None = None
    patch_size: int = 14
    k_keep: int = 2048
    selection_unit: str = "patch"
    static_fallback: bool = False
    static_abs_thresh: float = 2.0
    static_rel_thresh: float = 0.15
    static_uniform_frames: int = 4
    energy_pct: float = 95.0
    output_dtype: str = "bfloat16"
    backend: str = "auto"
    device_id: int = 0
    prefetch_depth: int = 3
    nvdec_session_pool_size: int | None = None
    uniform_auto_ratio: int | None = None
    decode_threads: int | None = None
    decode_thread_type: str | None = None
    reader_cache_size: int | None = None
    nvdec_reuse_open_decoder: bool | None = None


@dataclass
class PatchStreamResult:
    patches: torch.Tensor
    metadata: List[Dict[str, int | float | bool]]
    metadata_tensors: Tuple[torch.Tensor, torch.Tensor]
    sampled_frame_ids: List[int]
    fps: float
    duration_sec: float
