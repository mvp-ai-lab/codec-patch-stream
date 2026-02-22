#include "decode_core.h"

#include <cstdlib>
#include <optional>
#include <vector>

#include "codec_patch_stream.h"

namespace codec_patch_stream {
namespace {

class ScopedEnv {
 public:
  ScopedEnv(const char* key, const std::string& value) : key_(key) {
    const char* old = std::getenv(key_);
    if (old != nullptr) {
      old_ = std::string(old);
    }
    setenv(key_, value.c_str(), 1);
  }

  ~ScopedEnv() {
    if (old_.has_value()) {
      setenv(key_, old_->c_str(), 1);
    } else {
      unsetenv(key_);
    }
  }

 private:
  const char* key_;
  std::optional<std::string> old_;
};

}  // namespace

DecodeResultCanonical decode_only_native_gpu(const DecodeRequest& req) {
  std::vector<ScopedEnv> env_overrides;
  env_overrides.emplace_back("CODEC_NVDEC_SESSION_POOL_SIZE",
                             std::to_string(req.nvdec_session_pool_size >= 0 ? req.nvdec_session_pool_size : 2));
  env_overrides.emplace_back("CODEC_DECODE_UNIFORM_AUTO_RATIO",
                             std::to_string(req.uniform_auto_ratio >= 0 ? req.uniform_auto_ratio : 12));
  env_overrides.emplace_back("CODEC_DECODE_READER_CACHE_SIZE",
                             std::to_string(req.reader_cache_size >= 0 ? req.reader_cache_size : 8));
  env_overrides.emplace_back("CODEC_DECODE_DISABLE_READER_CACHE",
                             (req.reader_cache_size >= 0 && req.reader_cache_size <= 0) ? "1" : "0");
  env_overrides.emplace_back("CODEC_NVDEC_REUSE_OPEN_DECODER",
                             req.nvdec_reuse_open_decoder >= 0
                                 ? (req.nvdec_reuse_open_decoder ? "1" : "0")
                                 : "");
  constexpr const char* resolved_mode = "throughput";

  DecodeResult raw;
  if (req.uniform_strategy == "stream") {
    raw = decode_sampled_frames_nvdec(
        req.video_path, req.sequence_length, req.decode_device_id, resolved_mode);
  } else {
    const char* planner_mode = req.uniform_strategy == "seek" ? "seek" : "auto";
    ScopedEnv env_mode("CODEC_DECODE_UNIFORM_NVDEC_MODE", planner_mode);
    raw = decode_uniform_frames_nvdec(
        req.video_path, req.sequence_length, req.decode_device_id, resolved_mode);
  }

  DecodeResultCanonical out;
  out.frames_rgb_u8 = raw.frames_rgb_u8;
  out.sampled_frame_ids = std::move(raw.sampled_frame_ids);
  out.is_i_positions = std::move(raw.is_i_positions);
  out.fps = raw.fps;
  out.duration_sec = raw.duration_sec;
  out.width = raw.width;
  out.height = raw.height;
  return canonicalize_decode_result(req, std::move(out));
}

}  // namespace codec_patch_stream
