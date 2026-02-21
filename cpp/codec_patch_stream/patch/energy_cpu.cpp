#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <tuple>

#include "codec_patch_stream_cpu.h"

namespace codec_patch_stream {

std::tuple<at::Tensor, at::Tensor> compute_motion_residual_proxy_small_cpu(
    const at::Tensor& luma_small,
    int64_t search_radius) {
  TORCH_CHECK(!luma_small.is_cuda(), "luma_small must be CPU tensor");
  TORCH_CHECK(luma_small.scalar_type() == at::kFloat, "luma_small must be float32");
  TORCH_CHECK(luma_small.dim() == 3, "luma_small must have shape (T,H,W)");
  TORCH_CHECK(search_radius >= 0, "search_radius must be >= 0");

  auto luma = luma_small.contiguous();
  const int64_t t = luma.size(0);
  const int64_t hs = luma.size(1);
  const int64_t ws = luma.size(2);

  auto mv_proxy = torch::zeros_like(luma);
  auto residual_proxy = torch::zeros_like(luma);
  if (t <= 1 || hs <= 0 || ws <= 0) {
    return std::make_tuple(mv_proxy, residual_proxy);
  }

  const float* luma_ptr = luma.data_ptr<float>();
  float* mv_ptr = mv_proxy.data_ptr<float>();
  float* residual_ptr = residual_proxy.data_ptr<float>();

  for (int64_t frame_idx = 1; frame_idx < t; ++frame_idx) {
    const int64_t prev_base = (frame_idx - 1) * hs * ws;
    for (int64_t y = 0; y < hs; ++y) {
      for (int64_t x = 0; x < ws; ++x) {
        const int64_t cur_offset = (frame_idx * hs + y) * ws + x;
        const float cur = luma_ptr[cur_offset];
        float best_err = 1e9f;
        float best_mag = 0.0f;

        for (int64_t dy = -search_radius; dy <= search_radius; ++dy) {
          const int64_t yp = y - dy;
          if (yp < 0 || yp >= hs) {
            continue;
          }
          for (int64_t dx = -search_radius; dx <= search_radius; ++dx) {
            const int64_t xp = x - dx;
            if (xp < 0 || xp >= ws) {
              continue;
            }
            const float prev = luma_ptr[prev_base + yp * ws + xp];
            const float err = std::fabs(cur - prev);
            if (err < best_err) {
              best_err = err;
              best_mag = std::sqrt(static_cast<float>(dx * dx + dy * dy));
            }
          }
        }

        mv_ptr[cur_offset] = best_mag;
        residual_ptr[cur_offset] = best_err;
      }
    }
  }

  return std::make_tuple(mv_proxy, residual_proxy);
}

}  // namespace codec_patch_stream
