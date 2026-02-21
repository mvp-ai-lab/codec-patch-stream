#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "codec_patch_stream.h"

namespace codec_patch_stream {
namespace {

__device__ __forceinline__ uint8_t clamp_u8(float v) {
  if (v < 0.0f) {
    return 0;
  }
  if (v > 255.0f) {
    return 255;
  }
  return static_cast<uint8_t>(v + 0.5f);
}

__global__ void nv12_to_rgb_kernel(const uint8_t* y_plane,
                                   const uint8_t* uv_plane,
                                   int y_pitch,
                                   int uv_pitch,
                                   uint8_t* rgb_out,
                                   int rgb_pitch,
                                   int width,
                                   int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const uint8_t yv = y_plane[y * y_pitch + x];
  const int uv_base = (y >> 1) * uv_pitch + (x >> 1) * 2;
  const float u = static_cast<float>(uv_plane[uv_base + 0]) - 128.0f;
  const float v = static_cast<float>(uv_plane[uv_base + 1]) - 128.0f;
  const float yy = static_cast<float>(yv);

  const float r = yy + 1.402f * v;
  const float g = yy - 0.344136f * u - 0.714136f * v;
  const float b = yy + 1.772f * u;

  uint8_t* out = rgb_out + y * rgb_pitch + x * 3;
  out[0] = clamp_u8(r);
  out[1] = clamp_u8(g);
  out[2] = clamp_u8(b);
}

}  // namespace

void launch_nv12_to_rgb_cuda(const uint8_t* y_plane,
                             const uint8_t* uv_plane,
                             int y_pitch,
                             int uv_pitch,
                             uint8_t* rgb_out,
                             int rgb_pitch,
                             int width,
                             int height,
                             void* stream) {
  dim3 block(32, 8);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  nv12_to_rgb_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      y_plane, uv_plane, y_pitch, uv_pitch, rgb_out, rgb_pitch, width, height);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace codec_patch_stream
