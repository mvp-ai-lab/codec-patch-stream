#include <torch/extension.h>

#include <cstdint>

#include "codec_patch_stream_cpu.h"

namespace codec_patch_stream {

at::Tensor extract_patches_by_indices_cpu(const at::Tensor& frames_rgb,
                                          const at::Tensor& visible_indices,
                                          int64_t patch_size,
                                          int64_t wb) {
  TORCH_CHECK(!frames_rgb.is_cuda(), "frames_rgb must be on CPU");
  TORCH_CHECK(!visible_indices.is_cuda(), "visible_indices must be on CPU");
  TORCH_CHECK(frames_rgb.scalar_type() == at::kFloat, "frames_rgb must be float32");
  TORCH_CHECK(visible_indices.scalar_type() == at::kLong,
              "visible_indices must be int64");
  TORCH_CHECK(frames_rgb.dim() == 4 && frames_rgb.size(3) == 3,
              "frames_rgb shape must be (T,H,W,3)");

  const int64_t t = frames_rgb.size(0);
  const int64_t h = frames_rgb.size(1);
  const int64_t w = frames_rgb.size(2);
  TORCH_CHECK(patch_size > 0, "patch_size must be > 0");
  TORCH_CHECK(h % patch_size == 0 && w % patch_size == 0,
              "H/W must be divisible by patch_size");

  const int64_t hb = h / patch_size;
  TORCH_CHECK(wb == (w / patch_size), "wb mismatch with width/patch_size");

  auto x = frames_rgb.permute({0, 3, 1, 2}).contiguous();  // T,3,H,W
  auto grid = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size);
  auto flat = grid.permute({0, 2, 3, 1, 4, 5})
                  .contiguous()
                  .view({t * hb * wb, 3, patch_size, patch_size});
  return flat.index_select(0, visible_indices.contiguous());
}

}  // namespace codec_patch_stream
