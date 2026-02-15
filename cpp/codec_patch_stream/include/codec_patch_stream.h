#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace codec_patch_stream {

struct StreamConfig {
  int64_t sequence_length = 16;
  int64_t input_size = 224;
  int64_t patch_size = 14;
  int64_t k_keep = 2048;
  bool static_fallback = false;
  double static_abs_thresh = 2.0;
  double static_rel_thresh = 0.15;
  int64_t static_uniform_frames = 4;
  double energy_pct = 95.0;
  int64_t device_id = 0;
  int64_t prefetch_depth = 3;
  at::ScalarType output_dtype = at::kBFloat16;
};

struct PatchMeta {
  int64_t seq_pos = 0;
  int64_t frame_id = 0;
  bool is_i = false;
  int64_t patch_linear_idx = 0;
  int64_t patch_h_idx = 0;
  int64_t patch_w_idx = 0;
  float score = 0.0f;
};

struct DecodeResult {
  at::Tensor frames_rgb_u8;
  std::vector<int64_t> sampled_frame_ids;
  std::vector<uint8_t> is_i_positions;
  int64_t width = 0;
  int64_t height = 0;
};

struct SelectionResult {
  at::Tensor visible_indices;
  at::Tensor scores_flat;
  int64_t hb = 0;
  int64_t wb = 0;
};

DecodeResult decode_sampled_frames_nvdec(const std::string& video_path,
                                         int64_t sequence_length,
                                         int64_t device_id);

void launch_nv12_to_rgb_cuda(const uint8_t* y_plane,
                             const uint8_t* uv_plane,
                             int y_pitch,
                             int uv_pitch,
                             uint8_t* rgb_out,
                             int rgb_pitch,
                             int width,
                             int height,
                             void* stream);

at::Tensor compute_energy_maps_cuda(const at::Tensor& frames_rgb_u8,
                                    const std::vector<uint8_t>& is_i_positions,
                                    double energy_pct);

at::Tensor resize_to_input_cuda(const at::Tensor& frames_or_energy,
                                int64_t input_size,
                                bool is_energy);

SelectionResult compute_visible_indices_cuda(const at::Tensor& energy,
                                             int64_t patch_size,
                                             int64_t k_keep,
                                             bool static_fallback,
                                             double static_abs_thresh,
                                             double static_rel_thresh,
                                             int64_t static_uniform_frames,
                                             const std::vector<uint8_t>& is_i_positions);

at::Tensor extract_patches_by_indices_cuda(const at::Tensor& frames_rgb,
                                           const at::Tensor& visible_indices,
                                           int64_t patch_size,
                                           int64_t wb);

std::tuple<int64_t, int64_t, int64_t> linear_to_thw(int64_t linear_idx,
                                                     int64_t patches_per_frame,
                                                     int64_t patches_w);

class CodecPatchStreamNative {
 public:
  CodecPatchStreamNative(const std::string& video_path, const StreamConfig& cfg);

  int64_t size() const;
  bool has_next() const;
  std::tuple<at::Tensor, PatchMeta> next();
  std::tuple<at::Tensor, std::vector<PatchMeta>> next_n(int64_t n);
  void reset();
  void close();

  const at::Tensor& patch_bank() const;
  const std::vector<PatchMeta>& metadata() const;

 private:
  void prepare();

  std::string video_path_;
  StreamConfig cfg_;
  at::Tensor patch_bank_;
  std::vector<PatchMeta> meta_;
  int64_t cursor_ = 0;
  bool closed_ = false;
};

const char* version();

}  // namespace codec_patch_stream
