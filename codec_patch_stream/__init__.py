"""Video patch stream API with GPU/CPU native backends."""

from .decoder import DecodedFrames, decode_uniform_frames
from .stream import CodecPatchStream

__all__ = ["CodecPatchStream", "DecodedFrames", "decode_uniform_frames"]
