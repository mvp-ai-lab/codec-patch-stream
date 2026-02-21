#include <pybind11/pybind11.h>

#include <string>

#include "codec_patch_stream_cpu.h"
#include "decode_core.h"
#include "codec_patch_stream_pybind_backend_common.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  codec_patch_stream_pybind_common::bind_backend_module(
      m,
      "codec_patch_stream cpu native backend",
      "cpu",
      [](const std::string& video_path,
         int64_t sequence_length,
         int64_t /*device_id*/,
         const std::string& /*mode*/) {
        return codec_patch_stream::decode_uniform_frames_ffmpeg_cpu(video_path, sequence_length);
      },
      [](const std::string& video_path,
         int64_t sequence_length,
         int64_t device_id,
         const std::string& mode,
         const std::string& uniform_strategy,
         const std::string& backend_name,
         int64_t nvdec_session_pool_size,
         int64_t uniform_auto_ratio,
         int64_t decode_threads,
         const std::string& decode_thread_type,
         int64_t reader_cache_size,
         int64_t nvdec_reuse_open_decoder) {
        codec_patch_stream::DecodeRequest req;
        req.video_path = video_path;
        req.sequence_length = sequence_length;
        req.backend = backend_name;
        req.device_id = device_id;
        req.decode_mode = mode;
        req.uniform_strategy = uniform_strategy;
        req.nvdec_session_pool_size = nvdec_session_pool_size;
        req.uniform_auto_ratio = uniform_auto_ratio;
        req.decode_threads = decode_threads;
        req.decode_thread_type = decode_thread_type;
        req.reader_cache_size = reader_cache_size;
        req.nvdec_reuse_open_decoder = nvdec_reuse_open_decoder;
        return codec_patch_stream::decode_only_native_cpu(req);
      });
}
