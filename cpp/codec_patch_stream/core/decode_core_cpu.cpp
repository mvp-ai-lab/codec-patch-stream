#include "decode_core.h"

#include <cstdlib>
#include <optional>
#include <vector>

#include "codec_patch_stream_cpu.h"

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

DecodeResultCanonical decode_only_native_cpu(const DecodeRequest& req) {
  std::vector<ScopedEnv> env_overrides;
  env_overrides.emplace_back("CODEC_DECODE_THREADS",
                             std::to_string(req.decode_threads >= 0 ? req.decode_threads : 0));
  env_overrides.emplace_back("CODEC_DECODE_THREAD_TYPE",
                             req.decode_thread_type.empty() ? "auto" : req.decode_thread_type);
  env_overrides.emplace_back("CODEC_DECODE_READER_CACHE_SIZE",
                             std::to_string(req.reader_cache_size >= 0 ? req.reader_cache_size : 8));
  env_overrides.emplace_back("CODEC_DECODE_DISABLE_READER_CACHE",
                             (req.reader_cache_size >= 0 && req.reader_cache_size <= 0) ? "1" : "0");

  DecodeResult raw = decode_uniform_frames_ffmpeg_cpu(req.video_path, req.sequence_length);
  DecodeResultCanonical out;
  out.frames_rgb_u8 = raw.frames_rgb_u8;
  out.mv_magnitude_maps = raw.mv_magnitude_maps;
  out.sampled_frame_ids = std::move(raw.sampled_frame_ids);
  out.is_i_positions = std::move(raw.is_i_positions);
  out.fps = raw.fps;
  out.duration_sec = raw.duration_sec;
  out.width = raw.width;
  out.height = raw.height;
  return canonicalize_decode_result(req, std::move(out));
}

}  // namespace codec_patch_stream
