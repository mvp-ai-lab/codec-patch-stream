#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

#include "codec_patch_stream.h"

namespace codec_patch_stream {
namespace {

__global__ void gather_patch_kernel(const float* frames,
                                    const int64_t* visible_indices,
                                    float* out,
                                    int64_t k,
                                    int64_t h,
                                    int64_t w,
                                    int64_t patch_size,
                                    int64_t patches_per_frame,
                                    int64_t patches_w) {
  const int64_t total = k * 3 * patch_size * patch_size;
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total) {
    return;
  }

  int64_t r = tid;
  const int64_t ox = r % patch_size;
  r /= patch_size;
  const int64_t oy = r % patch_size;
  r /= patch_size;
  const int64_t c = r % 3;
  r /= 3;
  const int64_t k_idx = r;

  const int64_t linear_idx = visible_indices[k_idx];
  const int64_t t = linear_idx / patches_per_frame;
  const int64_t rem = linear_idx % patches_per_frame;
  const int64_t ph_idx = rem / patches_w;
  const int64_t pw_idx = rem % patches_w;

  const int64_t y = ph_idx * patch_size + oy;
  const int64_t x = pw_idx * patch_size + ox;

  const int64_t src_offset = ((t * h + y) * w + x) * 3 + c;
  out[tid] = frames[src_offset];
}

}  // namespace

at::Tensor extract_patches_by_indices_cuda(const at::Tensor& frames_rgb,
                                           const at::Tensor& visible_indices,
                                           int64_t patch_size,
                                           int64_t wb) {
  TORCH_CHECK(frames_rgb.is_cuda(), "frames_rgb must be on CUDA");
  TORCH_CHECK(visible_indices.is_cuda(), "visible_indices must be on CUDA");
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
  const int64_t patches_per_frame = hb * wb;

  const int64_t k = visible_indices.numel();
  auto out = torch::empty({k, 3, patch_size, patch_size},
                          frames_rgb.options().dtype(at::kFloat));
  if (k == 0) {
    return out;
  }

  const int threads = 256;
  const int64_t total = k * 3 * patch_size * patch_size;
  const int blocks = static_cast<int>((total + threads - 1) / threads);

  auto stream = at::cuda::getDefaultCUDAStream(frames_rgb.get_device());
  gather_patch_kernel<<<blocks, threads, 0, stream.stream()>>>(
      frames_rgb.contiguous().data_ptr<float>(),
      visible_indices.contiguous().data_ptr<int64_t>(),
      out.data_ptr<float>(),
      k,
      h,
      w,
      patch_size,
      patches_per_frame,
      wb);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace codec_patch_stream
