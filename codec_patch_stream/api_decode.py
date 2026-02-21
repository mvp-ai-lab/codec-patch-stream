from __future__ import annotations

from ._decode_core import decode_uniform_frames_core
from .types import DecodeConfig, DecodedFrames


def decode_only(config: DecodeConfig) -> DecodedFrames:
    return decode_uniform_frames_core(
        video_path=config.video_path,
        sequence_length=int(config.sequence_length),
        backend=str(config.backend),
        device_id=int(config.device_id),
        decode_mode=str(config.decode_mode),
        uniform_strategy=str(config.uniform_strategy),
        nvdec_session_pool_size=config.nvdec_session_pool_size,
        uniform_auto_ratio=config.uniform_auto_ratio,
        decode_threads=config.decode_threads,
        decode_thread_type=config.decode_thread_type,
        reader_cache_size=config.reader_cache_size,
        nvdec_reuse_open_decoder=config.nvdec_reuse_open_decoder,
    )
