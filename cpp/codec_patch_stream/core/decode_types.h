#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <string>
#include <vector>

namespace codec_patch_stream {

struct DecodeRequest {
  std::string video_path;
  int64_t sequence_length = 16;
  std::string decode_backend = "auto";  // auto | gpu | cpu
  int64_t decode_device_id = 0;
  std::string uniform_strategy = "auto";       // auto | seek | stream
  int64_t nvdec_session_pool_size = -1;        // <0 means library default
  int64_t uniform_auto_ratio = -1;             // <0 means library default
  int64_t decode_threads = -1;                 // <0 means library default
  std::string decode_thread_type;              // empty means library default
  int64_t reader_cache_size = -1;              // <0 means library default
  int64_t nvdec_reuse_open_decoder = -1;       // -1 default policy, 0 off, 1 on
};

struct DecodeResultCanonical {
  at::Tensor frames_rgb_u8;
  at::Tensor mv_magnitude_maps;  // CPU may provide, GPU keeps empty tensor
  std::vector<int64_t> sampled_frame_ids;
  std::vector<uint8_t> is_i_positions;
  double fps = 0.0;
  double duration_sec = 0.0;
  int64_t width = 0;
  int64_t height = 0;
};

}  // namespace codec_patch_stream
