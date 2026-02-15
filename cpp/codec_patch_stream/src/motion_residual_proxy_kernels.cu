#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>

#include "codec_patch_stream.h"

namespace codec_patch_stream {
namespace {

__global__ void motion_residual_proxy_kernel(const float* luma,
                                             float* mv_proxy,
                                             float* residual_proxy,
                                             int64_t t,
                                             int64_t hs,
                                             int64_t ws,
                                             int64_t search_radius) {
  const int64_t total = (t - 1) * hs * ws;
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total) {
    return;
  }

  int64_t r = tid;
  const int64_t x = r % ws;
  r /= ws;
  const int64_t y = r % hs;
  r /= hs;
  const int64_t frame_idx = r + 1;

  const int64_t cur_offset = (frame_idx * hs + y) * ws + x;
  const int64_t prev_base = (frame_idx - 1) * hs * ws;

  const float cur = luma[cur_offset];
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

      const float prev = luma[prev_base + yp * ws + xp];
      const float err = fabsf(cur - prev);
      if (err < best_err) {
        best_err = err;
        best_mag = sqrtf(static_cast<float>(dx * dx + dy * dy));
      }
    }
  }

  mv_proxy[cur_offset] = best_mag;
  residual_proxy[cur_offset] = best_err;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> compute_motion_residual_proxy_small_cuda(
    const at::Tensor& luma_small,
    int64_t search_radius) {
  TORCH_CHECK(luma_small.is_cuda(), "luma_small must be CUDA tensor");
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

  const int threads = 256;
  const int64_t total = (t - 1) * hs * ws;
  const int blocks = static_cast<int>((total + threads - 1) / threads);

  auto stream = at::cuda::getDefaultCUDAStream(luma.get_device());
  motion_residual_proxy_kernel<<<blocks, threads, 0, stream.stream()>>>(luma.data_ptr<float>(),
                                                                         mv_proxy.data_ptr<float>(),
                                                                         residual_proxy.data_ptr<float>(),
                                                                         t,
                                                                         hs,
                                                                         ws,
                                                                         search_radius);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_tuple(mv_proxy, residual_proxy);
}

}  // namespace codec_patch_stream
