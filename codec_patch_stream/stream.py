from __future__ import annotations

from typing import Dict, Iterator, List, Tuple

import torch

from .native_backend import require_native_backend


class CodecPatchStream(Iterator[Tuple[torch.Tensor, Dict[str, int | float | bool]]]):
    """Thin Python wrapper over the native C++/CUDA stream engine."""

    def __init__(
        self,
        video_path: str,
        sequence_length: int = 16,
        input_size: int = 224,
        patch_size: int = 14,
        k_keep: int = 2048,
        static_fallback: bool = False,
        static_abs_thresh: float = 2.0,
        static_rel_thresh: float = 0.15,
        static_uniform_frames: int = 4,
        energy_pct: float = 95.0,
        output_dtype: str = "bfloat16",
        device_id: int = 0,
        prefetch_depth: int = 3,
    ):
        native = require_native_backend()
        self._native = native.CodecPatchStreamNative(
            video_path=video_path,
            sequence_length=int(sequence_length),
            input_size=int(input_size),
            patch_size=int(patch_size),
            k_keep=int(k_keep),
            static_fallback=bool(static_fallback),
            static_abs_thresh=float(static_abs_thresh),
            static_rel_thresh=float(static_rel_thresh),
            static_uniform_frames=int(static_uniform_frames),
            energy_pct=float(energy_pct),
            output_dtype=str(output_dtype),
            device_id=int(device_id),
            prefetch_depth=int(prefetch_depth),
        )

    def __iter__(self) -> "CodecPatchStream":
        return self

    def __next__(self) -> Tuple[torch.Tensor, Dict[str, int | float | bool]]:
        return self._native.__next__()

    def __len__(self) -> int:
        return int(self._native.__len__())

    def next_n(self, n: int) -> Tuple[torch.Tensor, List[Dict[str, int | float | bool]]]:
        return self._native.next_n(int(n))

    def next_n_tensors(self, n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._native.next_n_tensors(int(n))

    def reset(self) -> None:
        self._native.reset()

    def close(self) -> None:
        self._native.close()

    @property
    def patches(self) -> torch.Tensor:
        return self._native.patches

    @property
    def metadata(self) -> List[Dict[str, int | float | bool]]:
        return list(self._native.metadata)

    @property
    def metadata_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._native.metadata_tensors
