#include "decode_core.h"

#include <algorithm>

#include <torch/extension.h>

namespace codec_patch_stream {

namespace {

bool is_supported_uniform_strategy(const std::string& strategy) {
  return strategy == "auto" || strategy == "seek" || strategy == "stream";
}

}  // namespace

DecodeResultCanonical canonicalize_decode_result(const DecodeRequest& req,
                                                 DecodeResultCanonical out) {
  TORCH_CHECK(req.sequence_length > 0, "sequence_length must be > 0");
  TORCH_CHECK(is_supported_uniform_strategy(req.uniform_strategy),
              "uniform_strategy must be one of: auto, seek, stream");

  TORCH_CHECK(out.frames_rgb_u8.defined(), "decoded frames must be defined");
  TORCH_CHECK(out.frames_rgb_u8.dim() == 4 && out.frames_rgb_u8.size(3) == 3,
              "decoded frames must have shape (T,H,W,3)");
  TORCH_CHECK(out.frames_rgb_u8.scalar_type() == at::kByte,
              "decoded frames must be uint8");

  const int64_t t = out.frames_rgb_u8.size(0);
  TORCH_CHECK(static_cast<int64_t>(out.sampled_frame_ids.size()) == t,
              "sampled_frame_ids size mismatch");
  TORCH_CHECK(static_cast<int64_t>(out.is_i_positions.size()) == t,
              "is_i_positions size mismatch");

  TORCH_CHECK(out.width == out.frames_rgb_u8.size(2) && out.height == out.frames_rgb_u8.size(1),
              "width/height metadata mismatch with frame tensor shape");

  for (uint8_t& x : out.is_i_positions) {
    x = x ? 1 : 0;
  }
  return out;
}

}  // namespace codec_patch_stream
