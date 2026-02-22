from __future__ import annotations

from typing import Dict, Iterator, List, Tuple

import torch

from ._patch_stream_core import NativePatchStream
from .types import PatchStreamConfig, PatchStreamResult


class PatchStreamSession(Iterator[Tuple[torch.Tensor, Dict[str, int | float | bool]]]):
    def __init__(self, config: PatchStreamConfig):
        self._stream = NativePatchStream(
            video_path=str(config.video_path),
            sequence_length=int(config.sequence_length),
            uniform_strategy=str(config.uniform_strategy),
            input_size=int(config.input_size),
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            patch_size=int(config.patch_size),
            k_keep=int(config.k_keep),
            selection_unit=str(config.selection_unit),
            static_fallback=bool(config.static_fallback),
            static_abs_thresh=float(config.static_abs_thresh),
            static_rel_thresh=float(config.static_rel_thresh),
            static_uniform_frames=int(config.static_uniform_frames),
            energy_pct=float(config.energy_pct),
            output_dtype=str(config.output_dtype),
            decode_backend=str(config.decode_backend),
            process_backend=str(config.process_backend),
            decode_device_id=int(config.decode_device_id),
            process_device_id=int(config.process_device_id),
            prefetch_depth=int(config.prefetch_depth),
            nvdec_session_pool_size=config.nvdec_session_pool_size,
            uniform_auto_ratio=config.uniform_auto_ratio,
            decode_threads=config.decode_threads,
            decode_thread_type=config.decode_thread_type,
            reader_cache_size=config.reader_cache_size,
            nvdec_reuse_open_decoder=config.nvdec_reuse_open_decoder,
        )

    def __iter__(self) -> "PatchStreamSession":
        return self

    def __next__(self) -> Tuple[torch.Tensor, Dict[str, int | float | bool]]:
        return self._stream.__next__()

    def __len__(self) -> int:
        return int(self._stream.__len__())

    def next_n(self, n: int):
        return self._stream.next_n(int(n))

    def next_n_tensors(self, n: int):
        return self._stream.next_n_tensors(int(n))

    def reset(self) -> None:
        self._stream.reset()

    def close(self) -> None:
        self._stream.close()

    @property
    def patches(self) -> torch.Tensor:
        return self._stream.patches

    @property
    def metadata(self) -> List[Dict[str, int | float | bool]]:
        return self._stream.metadata

    @property
    def metadata_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._stream.metadata_tensors

    @property
    def sampled_frame_ids(self) -> List[int]:
        return self._stream.sampled_frame_ids

    @property
    def fps(self) -> float:
        return self._stream.fps

    @property
    def duration_sec(self) -> float:
        return self._stream.duration_sec

    def collect_all(self) -> PatchStreamResult:
        return PatchStreamResult(
            patches=self.patches,
            metadata=self.metadata,
            metadata_tensors=self.metadata_tensors,
            sampled_frame_ids=self.sampled_frame_ids,
            fps=self.fps,
            duration_sec=self.duration_sec,
        )


def patch_stream(config: PatchStreamConfig) -> PatchStreamSession:
    return PatchStreamSession(config)
