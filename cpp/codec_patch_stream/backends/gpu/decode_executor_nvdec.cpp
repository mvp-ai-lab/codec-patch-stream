#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixfmt.h>
}

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "codec_patch_stream.h"

namespace codec_patch_stream {
namespace {

std::string ff_err_str(int errnum) {
  char buf[AV_ERROR_MAX_STRING_SIZE];
  std::memset(buf, 0, sizeof(buf));
  av_strerror(errnum, buf, sizeof(buf));
  return std::string(buf);
}

std::string cu_err_str(CUresult cu_err) {
  const char* name = nullptr;
  const char* msg = nullptr;
  cuGetErrorName(cu_err, &name);
  cuGetErrorString(cu_err, &msg);
  std::string out = name ? std::string(name) : "CUDA_ERROR_UNKNOWN";
  if (msg && msg[0] != '\0') {
    out += " (";
    out += msg;
    out += ")";
  }
  return out;
}

void check_ff(int ret, const char* where) {
  if (ret < 0) {
    throw std::runtime_error(std::string(where) + " failed: " + ff_err_str(ret));
  }
}

// Try to reuse an existing CUDA context (typically PyTorch's primary
// context) by manually constructing the AVHWDeviceContext instead of
// going through av_hwdevice_ctx_create.  This avoids the host-side
// memory allocation that av_hwdevice_ctx_create performs internally,
// which can fail with "Cannot allocate memory" when host resources
// are fragmented after large model loading.
//
// Returns true on success (*hw_device_ctx is set), false on failure.
bool try_reuse_existing_cuda_ctx(AVBufferRef** hw_device_ctx,
                                 int64_t device_id,
                                 std::string* fail_reason = nullptr) {
  // Activate the target device so cuCtxGetCurrent returns its context.
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_id));

  // Force PyTorch to initialise CUDA on this device (no-op if already done).
  at::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(device_id));

  CUcontext cu_ctx = nullptr;
  CUresult cu_err = cuCtxGetCurrent(&cu_ctx);
  if (cu_err != CUDA_SUCCESS || cu_ctx == nullptr) {
    if (fail_reason) {
      *fail_reason = "cuCtxGetCurrent failed: " + cu_err_str(cu_err);
    }
    return false;
  }

  AVBufferRef* ctx = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_CUDA);
  if (!ctx) {
    if (fail_reason) {
      *fail_reason = "av_hwdevice_ctx_alloc(cuda) returned null";
    }
    return false;
  }

  auto* device_ctx =
      reinterpret_cast<AVHWDeviceContext*>(ctx->data);
  auto* cuda_device_ctx =
      reinterpret_cast<AVCUDADeviceContext*>(device_ctx->hwctx);

  // Hand our existing CUcontext to FFmpeg.  av_hwdevice_ctx_init will
  // use it directly instead of creating a new one.
  cuda_device_ctx->cuda_ctx = cu_ctx;

  int ret = av_hwdevice_ctx_init(ctx);
  if (ret < 0) {
    if (fail_reason) {
      *fail_reason = "av_hwdevice_ctx_init(cuda) failed: " + ff_err_str(ret);
    }
    av_buffer_unref(&ctx);
    return false;
  }

  *hw_device_ctx = ctx;
  return true;
}

void create_hw_device_ctx_cuda(AVBufferRef** hw_device_ctx, int64_t device_id) {
  char device_name[16];
  std::snprintf(device_name, sizeof(device_name), "%lld",
                static_cast<long long>(device_id));

  std::string errors;
  auto append_error = [&errors](const char* stage, int code) {
    if (!errors.empty()) {
      errors += "; ";
    }
    errors += stage;
    errors += "=";
    errors += ff_err_str(code);
  };

  auto try_create = [&](const char* stage, const char* dev, int flags) -> bool {
    const int ret = av_hwdevice_ctx_create(
        hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, dev, nullptr, flags);
    if (ret >= 0) {
      return true;
    }
    append_error(stage, ret);
    return false;
  };

  // Prefer legacy creation first (same behavior as ffmpeg CLI default).
  // Some driver/runtime combinations fail primary-context mode once a runtime
  // context is already active in-process.
  if (try_create("legacy_device", device_name, 0)) {
    return;
  }
  if (try_create("legacy_default", nullptr, 0)) {
    return;
  }
  if (try_create("primary_device", device_name, AV_CUDA_USE_PRIMARY_CONTEXT)) {
    return;
  }
  if (try_create("primary_default", nullptr, AV_CUDA_USE_PRIMARY_CONTEXT)) {
    return;
  }

  // Final fallback: reuse an existing CUDA context from PyTorch.
  std::string reuse_fail_reason;
  if (try_reuse_existing_cuda_ctx(hw_device_ctx, device_id, &reuse_fail_reason)) {
    return;
  }
  if (!errors.empty()) {
    errors += "; ";
  }
  errors += "reuse_existing_ctx=failed";
  if (!reuse_fail_reason.empty()) {
    errors += " (";
    errors += reuse_fail_reason;
    errors += ")";
  }

  throw std::runtime_error("av_hwdevice_ctx_create(cuda) failed: " + errors);
}

std::mutex& hw_ctx_cache_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<int64_t, AVBufferRef*>& hw_ctx_cache() {
  static std::unordered_map<int64_t, AVBufferRef*> cache;
  return cache;
}

AVBufferRef* get_or_create_cached_hw_device_ctx(int64_t device_id) {
  {
    std::lock_guard<std::mutex> lock(hw_ctx_cache_mutex());
    auto it = hw_ctx_cache().find(device_id);
    if (it != hw_ctx_cache().end() && it->second) {
      return av_buffer_ref(it->second);
    }
  }

  AVBufferRef* created = nullptr;
  create_hw_device_ctx_cuda(&created, device_id);

  std::lock_guard<std::mutex> lock(hw_ctx_cache_mutex());
  auto it = hw_ctx_cache().find(device_id);
  if (it == hw_ctx_cache().end() || !it->second) {
    hw_ctx_cache()[device_id] = av_buffer_ref(created);
  }
  return created;
}

enum AVPixelFormat get_hw_format(AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts) {
  for (const enum AVPixelFormat* p = pix_fmts; *p != AV_PIX_FMT_NONE; ++p) {
    if (*p == AV_PIX_FMT_CUDA) {
      return *p;
    }
  }
  (void)ctx;
  throw std::runtime_error("CUDA hw pixel format is not supported by this decoder");
}

std::vector<int64_t> sample_frame_ids(int64_t total_frames, int64_t sequence_length) {
  if (total_frames <= 0) {
    throw std::invalid_argument("total_frames must be > 0");
  }
  if (sequence_length <= 0) {
    throw std::invalid_argument("sequence_length must be > 0");
  }

  std::vector<int64_t> out(static_cast<size_t>(sequence_length));
  if (sequence_length == 1) {
    out[0] = 0;
    return out;
  }
  const double step = static_cast<double>(total_frames - 1) / static_cast<double>(sequence_length - 1);
  for (int64_t i = 0; i < sequence_length; ++i) {
    out[static_cast<size_t>(i)] = static_cast<int64_t>(0 + i * step);
  }
  return out;
}

int64_t estimate_total_frames(AVFormatContext* fmt_ctx,
                              AVStream* video_stream,
                              AVCodecContext* dec_ctx) {
  if (video_stream->nb_frames > 0) {
    return video_stream->nb_frames;
  }

  AVRational fr = video_stream->avg_frame_rate;
  if (fr.num <= 0 || fr.den <= 0) {
    fr = dec_ctx->framerate;
  }
  const double fps = (fr.num > 0 && fr.den > 0) ? av_q2d(fr) : 0.0;
  if (fps > 0.0) {
    if (video_stream->duration > 0) {
      const double sec = video_stream->duration * av_q2d(video_stream->time_base);
      const int64_t est = static_cast<int64_t>(std::llround(sec * fps));
      if (est > 0) {
        return est;
      }
    }
    if (fmt_ctx->duration > 0) {
      const double sec = static_cast<double>(fmt_ctx->duration) / AV_TIME_BASE;
      const int64_t est = static_cast<int64_t>(std::llround(sec * fps));
      if (est > 0) {
        return est;
      }
    }
  }

  throw std::runtime_error(
      "Unable to infer total frame count. Please remux video with frame count metadata.");
}

double infer_fps(AVStream* video_stream, AVCodecContext* dec_ctx) {
  AVRational fr = video_stream->avg_frame_rate;
  if (fr.num <= 0 || fr.den <= 0) {
    fr = dec_ctx->framerate;
  }
  const double fps = (fr.num > 0 && fr.den > 0) ? av_q2d(fr) : 0.0;
  return fps > 0.0 ? fps : 0.0;
}

double infer_duration_sec(AVFormatContext* fmt_ctx,
                          AVStream* video_stream,
                          double fps,
                          int64_t total_frames) {
  if (video_stream->duration > 0) {
    const double sec = video_stream->duration * av_q2d(video_stream->time_base);
    if (sec > 0.0) {
      return sec;
    }
  }
  if (fmt_ctx->duration > 0) {
    const double sec = static_cast<double>(fmt_ctx->duration) / AV_TIME_BASE;
    if (sec > 0.0) {
      return sec;
    }
  }
  if (fps > 0.0 && total_frames > 0) {
    return static_cast<double>(total_frames) / fps;
  }
  return 0.0;
}

double infer_stream_fps_only(AVStream* video_stream) {
  AVRational fr = video_stream->avg_frame_rate;
  if (fr.num <= 0 || fr.den <= 0) {
    fr = video_stream->r_frame_rate;
  }
  const double fps = (fr.num > 0 && fr.den > 0) ? av_q2d(fr) : 0.0;
  return fps > 0.0 ? fps : 0.0;
}

int64_t estimate_total_frames_quick(AVFormatContext* fmt_ctx, AVStream* video_stream) {
  if (video_stream->nb_frames > 0) {
    return video_stream->nb_frames;
  }
  const double fps = infer_stream_fps_only(video_stream);
  if (fps <= 0.0) {
    return -1;
  }

  if (video_stream->duration > 0) {
    const double sec = video_stream->duration * av_q2d(video_stream->time_base);
    if (sec > 0.0) {
      const int64_t est = static_cast<int64_t>(std::llround(sec * fps));
      if (est > 0) {
        return est;
      }
    }
  }
  if (fmt_ctx->duration > 0) {
    const double sec = static_cast<double>(fmt_ctx->duration) / AV_TIME_BASE;
    if (sec > 0.0) {
      const int64_t est = static_cast<int64_t>(std::llround(sec * fps));
      if (est > 0) {
        return est;
      }
    }
  }
  return -1;
}

struct IndexedFrameInfo {
  int64_t pts = 0;
  int64_t dts = 0;
  bool is_key = false;
};

int64_t normalize_packet_pts(const AVPacket* pkt, int64_t fallback) {
  int64_t pts = pkt->pts;
  if (pts == AV_NOPTS_VALUE && pkt->dts != AV_NOPTS_VALUE) {
    pts = pkt->dts;
  }
  if (pts == AV_NOPTS_VALUE) {
    pts = fallback;
  }
  return pts;
}

std::vector<IndexedFrameInfo> build_sorted_frame_index(AVFormatContext* fmt_ctx,
                                                       int video_stream_idx) {
  AVPacket* pkt = av_packet_alloc();
  if (!pkt) {
    throw std::runtime_error("failed to allocate AVPacket for indexing");
  }

  std::vector<IndexedFrameInfo> out;
  int64_t fallback_pts = 0;
  try {
    while (av_read_frame(fmt_ctx, pkt) >= 0) {
      if (pkt->stream_index == video_stream_idx) {
        const int64_t pts = normalize_packet_pts(pkt, fallback_pts);
        const int64_t dts = (pkt->dts == AV_NOPTS_VALUE) ? pts : pkt->dts;
        out.push_back(IndexedFrameInfo{
            pts,
            dts,
            (pkt->flags & AV_PKT_FLAG_KEY) != 0,
        });
        ++fallback_pts;
      }
      av_packet_unref(pkt);
    }
    av_packet_free(&pkt);
  } catch (...) {
    av_packet_free(&pkt);
    throw;
  }

  if (out.empty()) {
    throw std::runtime_error("video stream has no packets");
  }

  std::stable_sort(out.begin(),
                   out.end(),
                   [](const IndexedFrameInfo& a, const IndexedFrameInfo& b) {
                     if (a.pts != b.pts) {
                       return a.pts < b.pts;
                     }
                     return a.dts < b.dts;
                   });
  return out;
}

int64_t parse_env_int64(const char* key, int64_t fallback) {
  const char* raw = std::getenv(key);
  if (!raw || !raw[0]) {
    return fallback;
  }
  char* end = nullptr;
  long long v = std::strtoll(raw, &end, 10);
  if (end == raw || (end && *end != '\0')) {
    return fallback;
  }
  return static_cast<int64_t>(v);
}

std::string parse_env_string(const char* key, const std::string& fallback) {
  const char* raw = std::getenv(key);
  if (!raw || !raw[0]) {
    return fallback;
  }
  return std::string(raw);
}

bool parse_env_bool(const char* key, bool fallback) {
  const std::string raw = parse_env_string(key, fallback ? "1" : "0");
  if (raw == "1" || raw == "true" || raw == "TRUE" || raw == "on" || raw == "ON") {
    return true;
  }
  if (raw == "0" || raw == "false" || raw == "FALSE" || raw == "off" || raw == "OFF") {
    return false;
  }
  return fallback;
}

enum class NvdecDecodeMode {
  kThroughput = 0,
  kLatency = 1,
};

std::string normalize_ascii_lower(std::string raw) {
  std::transform(raw.begin(), raw.end(), raw.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return raw;
}

NvdecDecodeMode parse_nvdec_decode_mode(const std::string& raw, NvdecDecodeMode fallback) {
  const std::string key = normalize_ascii_lower(raw);
  if (key.empty() || key == "auto") {
    return fallback;
  }
  if (key == "throughput" || key == "tp") {
    return NvdecDecodeMode::kThroughput;
  }
  if (key == "latency" || key == "low_latency") {
    return NvdecDecodeMode::kLatency;
  }
  throw std::invalid_argument("mode must be one of: throughput, latency, auto");
}

NvdecDecodeMode resolve_nvdec_decode_mode(const std::string& mode_override) {
  const std::string env_mode = parse_env_string("CODEC_NVDEC_MODE", "throughput");
  const NvdecDecodeMode env_resolved =
      parse_nvdec_decode_mode(env_mode, NvdecDecodeMode::kThroughput);
  return parse_nvdec_decode_mode(mode_override, env_resolved);
}

size_t decoder_pool_size_for_mode(NvdecDecodeMode mode) {
  if (mode == NvdecDecodeMode::kLatency) {
    return 1;
  }
  const int64_t configured = parse_env_int64("CODEC_NVDEC_SESSION_POOL_SIZE", 2);
  return static_cast<size_t>(configured > 0 ? configured : 1);
}

void maybe_print_decode_profile(
    const char* tag,
    const std::vector<std::pair<const char*, double>>& stages_ms,
    int64_t sequence_length,
    int64_t width,
    int64_t height,
    int64_t total_frames) {
  if (!parse_env_bool("CODEC_DECODE_PROFILE", false)) {
    return;
  }
  std::fprintf(stderr,
               "[codec-patch-stream][%s] seq=%lld size=%lldx%lld total=%lld",
               tag,
               static_cast<long long>(sequence_length),
               static_cast<long long>(width),
               static_cast<long long>(height),
               static_cast<long long>(total_frames));
  for (const auto& kv : stages_ms) {
    std::fprintf(stderr, " %s=%.3fms", kv.first, kv.second);
  }
  std::fprintf(stderr, "\n");
}

struct NvdecSeekProfileStats {
  int64_t seek_calls = 0;
  double seek_ms = 0.0;
  double flush_ms = 0.0;

  int64_t pop_calls = 0;
  double pop_total_ms = 0.0;

  int64_t read_calls = 0;
  double read_ms = 0.0;

  int64_t send_calls = 0;
  double send_ms = 0.0;

  int64_t receive_calls = 0;
  double receive_ms = 0.0;
  int64_t receive_ok = 0;
  int64_t receive_eagain = 0;
  int64_t receive_eof = 0;

  int64_t first_frame_latency_samples = 0;
  double first_frame_latency_ms = 0.0;

  int64_t copy_calls = 0;
  double copy_ms = 0.0;

  int64_t sync_calls = 0;
  double sync_ms = 0.0;
};

void maybe_print_nvdec_profile_detail(const char* tag,
                                      const NvdecSeekProfileStats& s,
                                      int64_t sequence_length,
                                      int64_t width,
                                      int64_t height,
                                      int64_t total_frames) {
  if (!parse_env_bool("CODEC_DECODE_PROFILE", false)) {
    return;
  }
  if (!parse_env_bool("CODEC_DECODE_PROFILE_VERBOSE", false)) {
    return;
  }
  std::fprintf(stderr,
               "[codec-patch-stream][%s] "
               "seq=%lld size=%lldx%lld total=%lld "
               "seek_calls=%lld seek_ms=%.3f flush_ms=%.3f "
               "pop_calls=%lld pop_total_ms=%.3f "
               "read_calls=%lld read_ms=%.3f "
               "send_calls=%lld send_ms=%.3f "
               "recv_calls=%lld recv_ms=%.3f recv_ok=%lld recv_eagain=%lld recv_eof=%lld "
               "first_frame_latency_samples=%lld first_frame_latency_ms=%.3f "
               "copy_calls=%lld copy_ms=%.3f "
               "sync_calls=%lld sync_ms=%.3f\n",
               tag,
               static_cast<long long>(sequence_length),
               static_cast<long long>(width),
               static_cast<long long>(height),
               static_cast<long long>(total_frames),
               static_cast<long long>(s.seek_calls),
               s.seek_ms,
               s.flush_ms,
               static_cast<long long>(s.pop_calls),
               s.pop_total_ms,
               static_cast<long long>(s.read_calls),
               s.read_ms,
               static_cast<long long>(s.send_calls),
               s.send_ms,
               static_cast<long long>(s.receive_calls),
               s.receive_ms,
               static_cast<long long>(s.receive_ok),
               static_cast<long long>(s.receive_eagain),
               static_cast<long long>(s.receive_eof),
               static_cast<long long>(s.first_frame_latency_samples),
               s.first_frame_latency_ms,
               static_cast<long long>(s.copy_calls),
               s.copy_ms,
               static_cast<long long>(s.sync_calls),
               s.sync_ms);
}

struct CodecStreamSignature {
  AVCodecID codec_id = AV_CODEC_ID_NONE;
  int format = AV_PIX_FMT_NONE;
  int width = 0;
  int height = 0;
  int profile = FF_PROFILE_UNKNOWN;
  int level = FF_LEVEL_UNKNOWN;
  int video_delay = 0;
  int extradata_size = 0;
  uint64_t extradata_hash = 0;
};

uint64_t hash_bytes_fnv1a(const uint8_t* data, int size) {
  uint64_t h = 1469598103934665603ULL;
  for (int i = 0; i < size; ++i) {
    h ^= static_cast<uint64_t>(data[i]);
    h *= 1099511628211ULL;
  }
  return h;
}

CodecStreamSignature make_codec_stream_signature(const AVCodecParameters* codecpar) {
  CodecStreamSignature sig;
  sig.codec_id = codecpar->codec_id;
  sig.format = codecpar->format;
  sig.width = codecpar->width;
  sig.height = codecpar->height;
  sig.profile = codecpar->profile;
  sig.level = codecpar->level;
  sig.video_delay = codecpar->video_delay;
  sig.extradata_size = codecpar->extradata_size;
  if (codecpar->extradata && codecpar->extradata_size > 0) {
    sig.extradata_hash = hash_bytes_fnv1a(codecpar->extradata, codecpar->extradata_size);
  }
  return sig;
}

bool same_codec_stream_signature(const CodecStreamSignature& a,
                                 const CodecStreamSignature& b) {
  return a.codec_id == b.codec_id && a.format == b.format && a.width == b.width &&
         a.height == b.height && a.profile == b.profile && a.level == b.level &&
         a.video_delay == b.video_delay && a.extradata_size == b.extradata_size &&
         a.extradata_hash == b.extradata_hash;
}

struct NvdecPersistentDecoder {
  explicit NvdecPersistentDecoder(int64_t device) : device_id(device) {}

  ~NvdecPersistentDecoder() {
    if (frame) {
      av_frame_free(&frame);
    }
    if (pkt) {
      av_packet_free(&pkt);
    }
    if (dec_ctx) {
      avcodec_free_context(&dec_ctx);
    }
    if (hw_device_ctx) {
      av_buffer_unref(&hw_device_ctx);
    }
  }

  int64_t device_id = 0;
  AVCodecContext* dec_ctx = nullptr;
  AVBufferRef* hw_device_ctx = nullptr;
  AVPacket* pkt = nullptr;
  AVFrame* frame = nullptr;
  bool decoder_open = false;
  bool has_signature = false;
  CodecStreamSignature signature;
  std::mutex mutex;
};

std::mutex& persistent_decoder_cache_mutex() {
  static std::mutex m;
  return m;
}

struct NvdecPersistentDecoderPool {
  std::vector<std::shared_ptr<NvdecPersistentDecoder>> decoders;
  size_t rr_cursor = 0;
};

std::unordered_map<int64_t, NvdecPersistentDecoderPool>& persistent_decoder_cache() {
  static std::unordered_map<int64_t, NvdecPersistentDecoderPool> c;
  return c;
}

std::shared_ptr<NvdecPersistentDecoder> create_persistent_decoder(int64_t device_id) {
  auto created = std::make_shared<NvdecPersistentDecoder>(device_id);
  created->hw_device_ctx = get_or_create_cached_hw_device_ctx(device_id);
  if (!created->hw_device_ctx) {
    throw std::runtime_error("failed to create persistent hw device ctx");
  }
  created->pkt = av_packet_alloc();
  created->frame = av_frame_alloc();
  if (!created->pkt || !created->frame) {
    throw std::runtime_error("failed to allocate persistent packet/frame");
  }
  return created;
}

std::shared_ptr<NvdecPersistentDecoder> acquire_persistent_decoder(int64_t device_id,
                                                                   NvdecDecodeMode mode,
                                                                   int64_t* slot_idx = nullptr) {
  const size_t required = decoder_pool_size_for_mode(mode);
  std::lock_guard<std::mutex> lock(persistent_decoder_cache_mutex());
  auto& pool = persistent_decoder_cache()[device_id];
  while (pool.decoders.size() < required) {
    pool.decoders.push_back(create_persistent_decoder(device_id));
  }
  if (pool.decoders.empty()) {
    throw std::runtime_error("persistent decoder pool is empty");
  }

  size_t selected = 0;
  if (mode == NvdecDecodeMode::kThroughput && pool.decoders.size() > 1) {
    selected = pool.rr_cursor % pool.decoders.size();
    pool.rr_cursor = (pool.rr_cursor + 1) % pool.decoders.size();
  }

  if (slot_idx) {
    *slot_idx = static_cast<int64_t>(selected);
  }
  return pool.decoders[selected];
}

void reopen_persistent_decoder_for_stream(NvdecPersistentDecoder& persistent,
                                          const AVCodec* codec,
                                          const AVCodecParameters* codecpar,
                                          NvdecDecodeMode mode) {
  if (!persistent.dec_ctx) {
    persistent.dec_ctx = avcodec_alloc_context3(codec);
    if (!persistent.dec_ctx) {
      throw std::runtime_error("avcodec_alloc_context3 failed");
    }
  } else if (persistent.dec_ctx->codec && persistent.dec_ctx->codec->id != codec->id) {
    if (persistent.decoder_open) {
      avcodec_close(persistent.dec_ctx);
      persistent.decoder_open = false;
    }
    avcodec_free_context(&persistent.dec_ctx);
    persistent.dec_ctx = avcodec_alloc_context3(codec);
    persistent.has_signature = false;
  }
  if (!persistent.dec_ctx) {
    throw std::runtime_error("avcodec_alloc_context3 failed");
  }

  const CodecStreamSignature sig = make_codec_stream_signature(codecpar);
  const bool default_reuse_open_decoder = (mode == NvdecDecodeMode::kThroughput);
  const bool can_reuse_open_decoder =
      parse_env_bool("CODEC_NVDEC_REUSE_OPEN_DECODER", default_reuse_open_decoder) &&
      persistent.decoder_open &&
      persistent.has_signature && same_codec_stream_signature(persistent.signature, sig);

  if (can_reuse_open_decoder) {
    avcodec_flush_buffers(persistent.dec_ctx);
    if (persistent.pkt) {
      av_packet_unref(persistent.pkt);
    }
    if (persistent.frame) {
      av_frame_unref(persistent.frame);
    }
    return;
  }

  if (persistent.decoder_open) {
    // NVDEC close->reconfigure on the same AVCodecContext can be unstable in
    // some driver/ffmpeg combinations. Use a fresh codec context for reopen.
    avcodec_free_context(&persistent.dec_ctx);
    persistent.dec_ctx = avcodec_alloc_context3(codec);
    if (!persistent.dec_ctx) {
      throw std::runtime_error("avcodec_alloc_context3 failed");
    }
    persistent.decoder_open = false;
    persistent.has_signature = false;
  }

  check_ff(avcodec_parameters_to_context(persistent.dec_ctx, codecpar),
           "avcodec_parameters_to_context");
  persistent.dec_ctx->get_format = get_hw_format;

  if (persistent.dec_ctx->hw_device_ctx) {
    av_buffer_unref(&persistent.dec_ctx->hw_device_ctx);
  }
  persistent.dec_ctx->hw_device_ctx = av_buffer_ref(persistent.hw_device_ctx);
  if (!persistent.dec_ctx->hw_device_ctx) {
    throw std::runtime_error("av_buffer_ref(hw_device_ctx) failed");
  }

  check_ff(avcodec_open2(persistent.dec_ctx, codec, nullptr), "avcodec_open2");
  persistent.decoder_open = true;
  persistent.signature = sig;
  persistent.has_signature = true;
  if (persistent.pkt) {
    av_packet_unref(persistent.pkt);
  }
  if (persistent.frame) {
    av_frame_unref(persistent.frame);
  }
}

struct CachedFrameIndex {
  std::vector<IndexedFrameInfo> indexed_frames;
  std::vector<int64_t> key_indices;
  std::unordered_map<int64_t, int64_t> pts_to_frame_idx;
  int64_t width = 0;
  int64_t height = 0;
};

struct FrameIndexCacheEntry {
  std::shared_ptr<const CachedFrameIndex> value;
  uint64_t touch = 0;
};

std::mutex& frame_index_cache_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::string, FrameIndexCacheEntry>& frame_index_cache() {
  static std::unordered_map<std::string, FrameIndexCacheEntry> c;
  return c;
}

uint64_t& frame_index_cache_clock() {
  static uint64_t tick = 0;
  return tick;
}

size_t frame_index_cache_limit() {
  const int64_t v = parse_env_int64("CODEC_DECODE_READER_CACHE_SIZE", 8);
  if (v <= 0) {
    return 0;
  }
  return static_cast<size_t>(v);
}

int64_t effective_uniform_auto_stream_ratio(int64_t base_ratio, int64_t sequence_length) {
  // Dense sampling tends to favor stream decode on NVDEC because repeated
  // seek+flush incurs high fixed overhead. Keep low-seq behavior unchanged.
  const bool enable_dense_adapt =
      parse_env_bool("CODEC_DECODE_UNIFORM_AUTO_DENSE_ADAPT", true);
  if (!enable_dense_adapt) {
    return base_ratio;
  }
  const int64_t dense_ratio = std::max<int64_t>(
      int64_t{12}, std::min<int64_t>(int64_t{128}, sequence_length / 4));
  return std::max(base_ratio, dense_ratio);
}

void evict_frame_index_cache_locked(size_t limit) {
  if (limit == 0) {
    frame_index_cache().clear();
    return;
  }
  while (frame_index_cache().size() > limit) {
    auto evict_it = frame_index_cache().begin();
    for (auto it = frame_index_cache().begin(); it != frame_index_cache().end(); ++it) {
      if (it->second.touch < evict_it->second.touch) {
        evict_it = it;
      }
    }
    frame_index_cache().erase(evict_it);
  }
}

std::shared_ptr<const CachedFrameIndex> build_cached_frame_index(AVFormatContext* fmt_ctx,
                                                                 int video_stream_idx,
                                                                 int64_t width,
                                                                 int64_t height) {
  auto idx = std::make_shared<CachedFrameIndex>();
  idx->indexed_frames = build_sorted_frame_index(fmt_ctx, video_stream_idx);
  idx->width = width;
  idx->height = height;

  idx->key_indices.reserve(idx->indexed_frames.size());
  idx->pts_to_frame_idx.reserve(idx->indexed_frames.size());
  for (int64_t i = 0; i < static_cast<int64_t>(idx->indexed_frames.size()); ++i) {
    if (idx->indexed_frames[static_cast<size_t>(i)].is_key) {
      idx->key_indices.push_back(i);
    }
    idx->pts_to_frame_idx.emplace(idx->indexed_frames[static_cast<size_t>(i)].pts, i);
  }
  if (idx->key_indices.empty()) {
    idx->key_indices.push_back(0);
  }
  return idx;
}

std::shared_ptr<const CachedFrameIndex> get_or_build_cached_frame_index(
    const std::string& video_path,
    AVFormatContext* fmt_ctx,
    int video_stream_idx,
    int64_t width,
    int64_t height,
    bool* cache_hit = nullptr) {
  const bool disable_cache = parse_env_bool("CODEC_DECODE_DISABLE_READER_CACHE", false);
  const size_t limit = frame_index_cache_limit();
  if (disable_cache || limit == 0) {
    if (cache_hit) {
      *cache_hit = false;
    }
    return build_cached_frame_index(fmt_ctx, video_stream_idx, width, height);
  }

  {
    std::lock_guard<std::mutex> lock(frame_index_cache_mutex());
    auto it = frame_index_cache().find(video_path);
    if (it != frame_index_cache().end() && it->second.value &&
        it->second.value->width == width && it->second.value->height == height) {
      it->second.touch = ++frame_index_cache_clock();
      if (cache_hit) {
        *cache_hit = true;
      }
      return it->second.value;
    }
  }

  auto built = build_cached_frame_index(fmt_ctx, video_stream_idx, width, height);

  {
    std::lock_guard<std::mutex> lock(frame_index_cache_mutex());
    auto& entry = frame_index_cache()[video_path];
    entry.value = built;
    entry.touch = ++frame_index_cache_clock();
    evict_frame_index_cache_locked(limit);
    if (cache_hit) {
      *cache_hit = false;
    }
    return entry.value;
  }
}

int64_t locate_keyframe_index(int64_t pos, const std::vector<int64_t>& key_indices) {
  if (key_indices.empty()) {
    return 0;
  }
  if (pos <= key_indices.front()) {
    return key_indices.front();
  }
  if (pos >= key_indices.back()) {
    return key_indices.back();
  }
  auto it = std::upper_bound(key_indices.begin(), key_indices.end(), pos);
  if (it == key_indices.begin()) {
    return key_indices.front();
  }
  --it;
  return *it;
}

void seek_to_frame_index(AVFormatContext* fmt_ctx,
                         AVCodecContext* dec_ctx,
                         int video_stream_idx,
                         int64_t target_idx,
                         int64_t target_pts,
                         NvdecSeekProfileStats* stats = nullptr) {
  auto elapsed_ms = [](const std::chrono::steady_clock::time_point& s,
                       const std::chrono::steady_clock::time_point& e) {
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(e - s)
        .count();
  };
  const auto t_seek_begin = std::chrono::steady_clock::now();
  int ret = av_seek_frame(fmt_ctx, video_stream_idx, target_pts, AVSEEK_FLAG_BACKWARD);
  if (ret < 0) {
    ret = av_seek_frame(fmt_ctx, video_stream_idx, target_pts, 0);
  }
  if (ret < 0) {
    ret = av_seek_frame(
        fmt_ctx, video_stream_idx, target_idx, AVSEEK_FLAG_BACKWARD | AVSEEK_FLAG_FRAME);
  }
  if (ret < 0) {
    throw std::runtime_error("failed to seek to frame index " + std::to_string(target_idx));
  }
  const auto t_seek_end = std::chrono::steady_clock::now();
  if (stats) {
    stats->seek_calls += 1;
    stats->seek_ms += elapsed_ms(t_seek_begin, t_seek_end);
  }
  const auto t_flush_begin = std::chrono::steady_clock::now();
  avcodec_flush_buffers(dec_ctx);
  const auto t_flush_end = std::chrono::steady_clock::now();
  if (stats) {
    stats->flush_ms += elapsed_ms(t_flush_begin, t_flush_end);
  }
}

bool pop_next_decoded_frame(AVFormatContext* fmt_ctx,
                            AVCodecContext* dec_ctx,
                            int video_stream_idx,
                            AVPacket* pkt,
                            AVFrame* frame,
                            bool* draining,
                            NvdecSeekProfileStats* stats = nullptr) {
  auto elapsed_ms = [](const std::chrono::steady_clock::time_point& s,
                       const std::chrono::steady_clock::time_point& e) {
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(e - s)
        .count();
  };
  const auto t_pop_begin = std::chrono::steady_clock::now();
  if (stats) {
    stats->pop_calls += 1;
  }
  while (true) {
    const auto t_recv_begin = std::chrono::steady_clock::now();
    const int recv_ret = avcodec_receive_frame(dec_ctx, frame);
    const auto t_recv_end = std::chrono::steady_clock::now();
    if (stats) {
      stats->receive_calls += 1;
      stats->receive_ms += elapsed_ms(t_recv_begin, t_recv_end);
    }
    if (recv_ret == 0) {
      if (stats) {
        stats->receive_ok += 1;
        stats->pop_total_ms += elapsed_ms(t_pop_begin, std::chrono::steady_clock::now());
      }
      return true;
    }
    if (recv_ret == AVERROR_EOF) {
      if (stats) {
        stats->receive_eof += 1;
        stats->pop_total_ms += elapsed_ms(t_pop_begin, std::chrono::steady_clock::now());
      }
      return false;
    }
    if (recv_ret != AVERROR(EAGAIN)) {
      check_ff(recv_ret, "avcodec_receive_frame");
    }
    if (stats) {
      stats->receive_eagain += 1;
    }

    if (*draining) {
      if (stats) {
        stats->pop_total_ms += elapsed_ms(t_pop_begin, std::chrono::steady_clock::now());
      }
      return false;
    }

    bool sent_packet = false;
    while (!sent_packet) {
      const auto t_read_begin = std::chrono::steady_clock::now();
      const int read_ret = av_read_frame(fmt_ctx, pkt);
      const auto t_read_end = std::chrono::steady_clock::now();
      if (stats) {
        stats->read_calls += 1;
        stats->read_ms += elapsed_ms(t_read_begin, t_read_end);
      }
      if (read_ret < 0) {
        if (read_ret == AVERROR_EOF) {
          check_ff(avcodec_send_packet(dec_ctx, nullptr), "avcodec_send_packet(flush)");
          *draining = true;
          break;
        }
        check_ff(read_ret, "av_read_frame");
      }
      if (read_ret >= 0) {
        if (pkt->stream_index == video_stream_idx) {
          const auto t_send_begin = std::chrono::steady_clock::now();
          check_ff(avcodec_send_packet(dec_ctx, pkt), "avcodec_send_packet");
          const auto t_send_end = std::chrono::steady_clock::now();
          if (stats) {
            stats->send_calls += 1;
            stats->send_ms += elapsed_ms(t_send_begin, t_send_end);
          }
          sent_packet = true;
        }
        av_packet_unref(pkt);
      }
    }
  }
}

void copy_cuda_nv12_frame_to_rgb(AVFrame* frame,
                                 at::Tensor* out_frames,
                                 int64_t out_idx,
                                 int64_t width,
                                 int64_t height,
                                 void* cuda_stream_ptr) {
  if (frame->format != AV_PIX_FMT_CUDA) {
    throw std::runtime_error("decoded frame is not AV_PIX_FMT_CUDA");
  }
  if (!frame->data[0] || !frame->data[1]) {
    throw std::runtime_error("CUDA frame plane pointers are null");
  }
  if (!frame->hw_frames_ctx) {
    throw std::runtime_error("CUDA frame missing hw_frames_ctx");
  }

  auto* hw_frames_ctx = reinterpret_cast<AVHWFramesContext*>(frame->hw_frames_ctx->data);
  if (!hw_frames_ctx || hw_frames_ctx->sw_format != AV_PIX_FMT_NV12) {
    throw std::runtime_error("Only NV12 CUDA frames are supported in this implementation");
  }

  uint8_t* dst = out_frames->data_ptr<uint8_t>() + out_idx * height * width * 3;
  launch_nv12_to_rgb_cuda(reinterpret_cast<const uint8_t*>(frame->data[0]),
                          reinterpret_cast<const uint8_t*>(frame->data[1]),
                          frame->linesize[0],
                          frame->linesize[1],
                          dst,
                          static_cast<int>(width * 3),
                          static_cast<int>(width),
                          static_cast<int>(height),
                          cuda_stream_ptr);
}

}  // namespace

DecodeResult decode_sampled_frames_nvdec(const std::string& video_path,
                                         int64_t sequence_length,
                                         int64_t device_id,
                                         const std::string& mode) {
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_id));
  const NvdecDecodeMode decode_mode = resolve_nvdec_decode_mode(mode);
  const auto t_begin = std::chrono::steady_clock::now();

  AVFormatContext* fmt_ctx = nullptr;
  std::shared_ptr<NvdecPersistentDecoder> persistent;
  std::unique_lock<std::mutex> persistent_lock;
  check_ff(avformat_open_input(&fmt_ctx, video_path.c_str(), nullptr, nullptr),
           "avformat_open_input");
  const auto t_after_open_input = std::chrono::steady_clock::now();

  try {
    int video_stream_idx =
        av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_idx < 0) {
      check_ff(avformat_find_stream_info(fmt_ctx, nullptr), "avformat_find_stream_info");
      video_stream_idx =
          av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    }
    check_ff(video_stream_idx, "av_find_best_stream");

    AVStream* video_stream = fmt_ctx->streams[video_stream_idx];
    const AVCodec* codec = avcodec_find_decoder(video_stream->codecpar->codec_id);
    if (!codec) {
      throw std::runtime_error("avcodec_find_decoder returned null");
    }

    persistent = acquire_persistent_decoder(device_id, decode_mode);
    persistent_lock = std::unique_lock<std::mutex>(persistent->mutex);
    reopen_persistent_decoder_for_stream(*persistent, codec, video_stream->codecpar, decode_mode);
    AVCodecContext* dec_ctx = persistent->dec_ctx;
    AVPacket* pkt = persistent->pkt;
    AVFrame* frame = persistent->frame;
    if (!dec_ctx || !pkt || !frame) {
      throw std::runtime_error("persistent decoder is not fully initialized");
    }
    const auto t_after_codec_open = std::chrono::steady_clock::now();

    const int64_t total_frames = estimate_total_frames(fmt_ctx, video_stream, dec_ctx);
    const double fps = infer_fps(video_stream, dec_ctx);
    const double duration_sec = infer_duration_sec(fmt_ctx, video_stream, fps, total_frames);
    std::vector<int64_t> frame_ids = sample_frame_ids(total_frames, sequence_length);

    std::vector<int64_t> uniq_frame_ids = frame_ids;
    std::sort(uniq_frame_ids.begin(), uniq_frame_ids.end());
    uniq_frame_ids.erase(std::unique(uniq_frame_ids.begin(), uniq_frame_ids.end()),
                         uniq_frame_ids.end());

    const int64_t width = dec_ctx->width;
    const int64_t height = dec_ctx->height;
    if (width <= 0 || height <= 0) {
      throw std::runtime_error("decoder produced invalid width/height");
    }

    auto frame_opts = torch::TensorOptions()
                          .dtype(at::kByte)
                          .device(torch::Device(torch::kCUDA, static_cast<int>(device_id)));
    at::Tensor uniq_frames = torch::empty(
        {static_cast<int64_t>(uniq_frame_ids.size()), height, width, 3}, frame_opts);

    std::vector<uint8_t> captured(uniq_frame_ids.size(), 0);
    std::vector<uint8_t> key_flags(uniq_frame_ids.size(), 0);

    const auto t_after_setup = std::chrono::steady_clock::now();
    const bool verbose_seek_profile = parse_env_bool("CODEC_DECODE_PROFILE_VERBOSE", false);
    NvdecSeekProfileStats seek_stats;
    NvdecSeekProfileStats* seek_stats_ptr = verbose_seek_profile ? &seek_stats : nullptr;

    auto elapsed_ms = [](const std::chrono::steady_clock::time_point& s,
                         const std::chrono::steady_clock::time_point& e) {
      return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(e - s)
          .count();
    };

    auto stream = at::cuda::getStreamFromPool(false, static_cast<c10::DeviceIndex>(device_id));
    void* cuda_stream_ptr = reinterpret_cast<void*>(stream.stream());

    int64_t decoded_frame_idx = 0;
    size_t next_target = 0;
    size_t captured_count = 0;
    bool done = uniq_frame_ids.empty();
    const auto t_decode_loop_begin = std::chrono::steady_clock::now();

    auto process_frame = [&](AVFrame* f) {
      if (!done && next_target < uniq_frame_ids.size() &&
          decoded_frame_idx == uniq_frame_ids[next_target]) {
        const int64_t pos = static_cast<int64_t>(next_target);
        const auto t_copy_begin = std::chrono::steady_clock::now();
        copy_cuda_nv12_frame_to_rgb(f, &uniq_frames, pos, width, height, cuda_stream_ptr);
        const auto t_copy_end = std::chrono::steady_clock::now();
        if (seek_stats_ptr) {
          seek_stats_ptr->copy_calls += 1;
          seek_stats_ptr->copy_ms += elapsed_ms(t_copy_begin, t_copy_end);
        }

        key_flags[next_target] = f->key_frame ? static_cast<uint8_t>(1)
                                              : static_cast<uint8_t>(0);
        if (!captured[next_target]) {
          captured[next_target] = 1;
          ++captured_count;
        }
        ++next_target;
        done = captured_count == uniq_frame_ids.size();
      }
      ++decoded_frame_idx;
    };

    while (true) {
      const auto t_read_begin = std::chrono::steady_clock::now();
      const int read_ret = av_read_frame(fmt_ctx, pkt);
      const auto t_read_end = std::chrono::steady_clock::now();
      if (seek_stats_ptr) {
        seek_stats_ptr->read_calls += 1;
        seek_stats_ptr->read_ms += elapsed_ms(t_read_begin, t_read_end);
      }
      if (read_ret < 0) {
        break;
      }
      if (pkt->stream_index == video_stream_idx) {
        const auto t_send_begin = std::chrono::steady_clock::now();
        check_ff(avcodec_send_packet(dec_ctx, pkt), "avcodec_send_packet");
        const auto t_send_end = std::chrono::steady_clock::now();
        if (seek_stats_ptr) {
          seek_stats_ptr->send_calls += 1;
          seek_stats_ptr->send_ms += elapsed_ms(t_send_begin, t_send_end);
        }
        while (true) {
          const auto t_recv_begin = std::chrono::steady_clock::now();
          const int ret = avcodec_receive_frame(dec_ctx, frame);
          const auto t_recv_end = std::chrono::steady_clock::now();
          if (seek_stats_ptr) {
            seek_stats_ptr->receive_calls += 1;
            seek_stats_ptr->receive_ms += elapsed_ms(t_recv_begin, t_recv_end);
          }
          if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            if (seek_stats_ptr) {
              if (ret == AVERROR(EAGAIN)) {
                seek_stats_ptr->receive_eagain += 1;
              } else {
                seek_stats_ptr->receive_eof += 1;
              }
            }
            break;
          }
          check_ff(ret, "avcodec_receive_frame");
          if (seek_stats_ptr) {
            seek_stats_ptr->receive_ok += 1;
            if (seek_stats_ptr->first_frame_latency_samples == 0) {
              seek_stats_ptr->first_frame_latency_samples = 1;
              seek_stats_ptr->first_frame_latency_ms +=
                  elapsed_ms(t_decode_loop_begin, std::chrono::steady_clock::now());
            }
          }
          process_frame(frame);
          av_frame_unref(frame);
          if (done) {
            break;
          }
        }
      }
      av_packet_unref(pkt);
      if (done) {
        break;
      }
    }

    if (!done) {
      const auto t_send_begin = std::chrono::steady_clock::now();
      check_ff(avcodec_send_packet(dec_ctx, nullptr), "avcodec_send_packet(flush)");
      const auto t_send_end = std::chrono::steady_clock::now();
      if (seek_stats_ptr) {
        seek_stats_ptr->send_calls += 1;
        seek_stats_ptr->send_ms += elapsed_ms(t_send_begin, t_send_end);
      }
      while (true) {
        const auto t_recv_begin = std::chrono::steady_clock::now();
        const int ret = avcodec_receive_frame(dec_ctx, frame);
        const auto t_recv_end = std::chrono::steady_clock::now();
        if (seek_stats_ptr) {
          seek_stats_ptr->receive_calls += 1;
          seek_stats_ptr->receive_ms += elapsed_ms(t_recv_begin, t_recv_end);
        }
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
          if (seek_stats_ptr) {
            if (ret == AVERROR(EAGAIN)) {
              seek_stats_ptr->receive_eagain += 1;
            } else {
              seek_stats_ptr->receive_eof += 1;
            }
          }
          break;
        }
        check_ff(ret, "avcodec_receive_frame(flush)");
        if (seek_stats_ptr) {
          seek_stats_ptr->receive_ok += 1;
          if (seek_stats_ptr->first_frame_latency_samples == 0) {
            seek_stats_ptr->first_frame_latency_samples = 1;
            seek_stats_ptr->first_frame_latency_ms +=
                elapsed_ms(t_decode_loop_begin, std::chrono::steady_clock::now());
          }
        }
        process_frame(frame);
        av_frame_unref(frame);
        if (done) {
          break;
        }
      }
    }

    const auto t_sync_begin = std::chrono::steady_clock::now();
    C10_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
    const auto t_sync_end = std::chrono::steady_clock::now();
    if (seek_stats_ptr) {
      seek_stats_ptr->sync_calls += 1;
      seek_stats_ptr->sync_ms += elapsed_ms(t_sync_begin, t_sync_end);
    }
    const auto t_after_decode = std::chrono::steady_clock::now();
    if (seek_stats_ptr) {
      seek_stats_ptr->pop_calls = 0;
      seek_stats_ptr->pop_total_ms = 0.0;
    }

    for (size_t i = 0; i < captured.size(); ++i) {
      if (!captured[i]) {
        throw std::runtime_error("failed to decode one or more sampled frames");
      }
    }

    std::vector<int64_t> gather_pos(frame_ids.size(), 0);
    std::vector<uint8_t> is_i_positions(frame_ids.size(), 0);
    for (size_t i = 0; i < frame_ids.size(); ++i) {
      const int64_t fid = frame_ids[i];
      auto it = std::lower_bound(uniq_frame_ids.begin(), uniq_frame_ids.end(), fid);
      if (it == uniq_frame_ids.end() || *it != fid) {
        throw std::runtime_error("sampled frame id not found in decoded frame set");
      }
      const size_t pos = static_cast<size_t>(std::distance(uniq_frame_ids.begin(), it));
      gather_pos[i] = static_cast<int64_t>(pos);
      is_i_positions[i] = key_flags[pos];
    }

    bool any_i = false;
    for (uint8_t v : is_i_positions) {
      if (v != 0) {
        any_i = true;
        break;
      }
    }
    if (!any_i && !is_i_positions.empty()) {
      is_i_positions[0] = 1;
    }

    bool need_gather = frame_ids.size() != uniq_frame_ids.size();
    if (!need_gather) {
      for (size_t i = 0; i < frame_ids.size(); ++i) {
        if (frame_ids[i] != uniq_frame_ids[i]) {
          need_gather = true;
          break;
        }
      }
    }

    at::Tensor sampled_frames;
    if (need_gather) {
      auto gather = torch::from_blob(gather_pos.data(),
                                     {static_cast<int64_t>(gather_pos.size())},
                                     torch::TensorOptions().dtype(at::kLong))
                        .clone()
                        .to(torch::Device(torch::kCUDA, static_cast<int>(device_id)));
      sampled_frames = uniq_frames.index_select(0, gather);
    } else {
      sampled_frames = uniq_frames;
    }
    const auto t_after_post = std::chrono::steady_clock::now();

    const auto t_before_cleanup = std::chrono::steady_clock::now();
    av_frame_unref(frame);
    av_packet_unref(pkt);
    const auto t_after_cleanup_state = std::chrono::steady_clock::now();
    if (fmt_ctx) {
      avformat_close_input(&fmt_ctx);
    }
    const auto t_after_close_input = std::chrono::steady_clock::now();

    maybe_print_decode_profile(
        "decode_sampled_nvdec",
        {
            {"open_input", elapsed_ms(t_begin, t_after_open_input)},
            {"codec_prepare", elapsed_ms(t_after_open_input, t_after_codec_open)},
            {"setup", elapsed_ms(t_after_codec_open, t_after_setup)},
            {"decode", elapsed_ms(t_after_setup, t_after_decode)},
            {"post", elapsed_ms(t_after_decode, t_after_post)},
            {"cleanup_state", elapsed_ms(t_before_cleanup, t_after_cleanup_state)},
            {"cleanup_input", elapsed_ms(t_after_cleanup_state, t_after_close_input)},
            {"total", elapsed_ms(t_begin, t_after_close_input)},
        },
        sequence_length,
        width,
        height,
        total_frames);
    if (seek_stats_ptr) {
      maybe_print_nvdec_profile_detail(
          "decode_sampled_nvdec_detail",
          *seek_stats_ptr,
          sequence_length,
          width,
          height,
          total_frames);
    }

    DecodeResult out;
    out.frames_rgb_u8 = sampled_frames;
    out.sampled_frame_ids = std::move(frame_ids);
    out.is_i_positions = std::move(is_i_positions);
    out.fps = fps;
    out.duration_sec = duration_sec;
    out.width = width;
    out.height = height;
    return out;

  } catch (...) {
    if (persistent && persistent_lock.owns_lock()) {
      if (persistent->frame) {
        av_frame_unref(persistent->frame);
      }
      if (persistent->pkt) {
        av_packet_unref(persistent->pkt);
      }
      if (persistent->dec_ctx && persistent->decoder_open) {
        avcodec_flush_buffers(persistent->dec_ctx);
      }
    }
    if (fmt_ctx) {
      avformat_close_input(&fmt_ctx);
    }
    throw;
  }
}
DecodeResult decode_uniform_frames_nvdec(const std::string& video_path,
                                         int64_t sequence_length,
                                         int64_t device_id,
                                         const std::string& mode) {
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_id));
  const NvdecDecodeMode decode_mode = resolve_nvdec_decode_mode(mode);
  const auto t_begin = std::chrono::steady_clock::now();

  AVFormatContext* fmt_ctx = nullptr;
  std::shared_ptr<NvdecPersistentDecoder> persistent;
  std::unique_lock<std::mutex> persistent_lock;
  check_ff(avformat_open_input(&fmt_ctx, video_path.c_str(), nullptr, nullptr),
           "avformat_open_input");
  const auto t_after_open_input = std::chrono::steady_clock::now();

  try {
    int video_stream_idx =
        av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_idx < 0) {
      check_ff(avformat_find_stream_info(fmt_ctx, nullptr), "avformat_find_stream_info");
      video_stream_idx =
          av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    }
    check_ff(video_stream_idx, "av_find_best_stream");

    AVStream* video_stream = fmt_ctx->streams[video_stream_idx];
    const std::string uniform_mode =
        parse_env_string("CODEC_DECODE_UNIFORM_NVDEC_MODE", "auto");
    const bool auto_uniform_mode = (uniform_mode == "auto");
    int64_t quick_total_frames = -1;
    int64_t auto_stream_ratio = -1;
    if (uniform_mode == "stream") {
      if (fmt_ctx) {
        avformat_close_input(&fmt_ctx);
      }
      return decode_sampled_frames_nvdec(video_path, sequence_length, device_id, mode);
    }
    if (uniform_mode != "auto" && uniform_mode != "seek") {
      throw std::invalid_argument(
          "CODEC_DECODE_UNIFORM_NVDEC_MODE must be one of: auto, seek, stream");
    }
    if (uniform_mode == "auto") {
      quick_total_frames = estimate_total_frames_quick(fmt_ctx, video_stream);
      const int64_t base_stream_ratio =
          parse_env_int64("CODEC_DECODE_UNIFORM_AUTO_RATIO", 12);
      auto_stream_ratio =
          effective_uniform_auto_stream_ratio(base_stream_ratio, sequence_length);
      if (quick_total_frames > 0 && auto_stream_ratio > 0 &&
          quick_total_frames <= sequence_length * auto_stream_ratio) {
        if (fmt_ctx) {
          avformat_close_input(&fmt_ctx);
        }
        return decode_sampled_frames_nvdec(video_path, sequence_length, device_id, mode);
      }
    }

    const AVCodec* codec = avcodec_find_decoder(video_stream->codecpar->codec_id);
    if (!codec) {
      throw std::runtime_error("avcodec_find_decoder returned null");
    }

    persistent = acquire_persistent_decoder(device_id, decode_mode);
    persistent_lock = std::unique_lock<std::mutex>(persistent->mutex);
    reopen_persistent_decoder_for_stream(*persistent, codec, video_stream->codecpar, decode_mode);
    AVCodecContext* dec_ctx = persistent->dec_ctx;
    AVPacket* pkt = persistent->pkt;
    AVFrame* frame = persistent->frame;
    if (!dec_ctx || !pkt || !frame) {
      throw std::runtime_error("persistent decoder is not fully initialized");
    }
    const auto t_after_codec_open = std::chrono::steady_clock::now();

    const int64_t fallback_total_frames = estimate_total_frames(fmt_ctx, video_stream, dec_ctx);
    if (auto_uniform_mode && quick_total_frames <= 0 && auto_stream_ratio > 0 &&
        fallback_total_frames > 0 &&
        fallback_total_frames <= sequence_length * auto_stream_ratio) {
      if (frame) {
        av_frame_unref(frame);
      }
      if (pkt) {
        av_packet_unref(pkt);
      }
      if (persistent_lock.owns_lock()) {
        persistent_lock.unlock();
      }
      if (fmt_ctx) {
        avformat_close_input(&fmt_ctx);
      }
      return decode_sampled_frames_nvdec(video_path, sequence_length, device_id, mode);
    }
    const double fps = infer_fps(video_stream, dec_ctx);
    const double duration_sec =
        infer_duration_sec(fmt_ctx, video_stream, fps, fallback_total_frames);

    const int64_t width = dec_ctx->width;
    const int64_t height = dec_ctx->height;
    if (width <= 0 || height <= 0) {
      throw std::runtime_error("decoder produced invalid width/height");
    }

    bool cache_hit = false;
    auto cached_index =
        get_or_build_cached_frame_index(video_path, fmt_ctx, video_stream_idx, width, height, &cache_hit);
    (void)cache_hit;
    const auto& indexed_frames = cached_index->indexed_frames;
    const auto& key_indices = cached_index->key_indices;
    const auto& pts_to_frame_idx = cached_index->pts_to_frame_idx;
    const int64_t total_frames = static_cast<int64_t>(indexed_frames.size());

    std::vector<int64_t> frame_ids = sample_frame_ids(total_frames, sequence_length);
    std::vector<int64_t> uniq_frame_ids = frame_ids;
    std::sort(uniq_frame_ids.begin(), uniq_frame_ids.end());
    uniq_frame_ids.erase(std::unique(uniq_frame_ids.begin(), uniq_frame_ids.end()),
                         uniq_frame_ids.end());

    auto frame_opts = torch::TensorOptions()
                          .dtype(at::kByte)
                          .device(torch::Device(torch::kCUDA, static_cast<int>(device_id)));
    at::Tensor uniq_frames = torch::empty(
        {static_cast<int64_t>(uniq_frame_ids.size()), height, width, 3}, frame_opts);
    std::vector<uint8_t> key_flags(uniq_frame_ids.size(), 0);
    const auto t_after_setup = std::chrono::steady_clock::now();
    const bool verbose_seek_profile = parse_env_bool("CODEC_DECODE_PROFILE_VERBOSE", false);
    NvdecSeekProfileStats seek_stats;
    NvdecSeekProfileStats* seek_stats_ptr = verbose_seek_profile ? &seek_stats : nullptr;

    auto elapsed_ms = [](const std::chrono::steady_clock::time_point& s,
                         const std::chrono::steady_clock::time_point& e) {
      return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(e - s)
          .count();
    };

    auto stream = at::cuda::getStreamFromPool(false, static_cast<c10::DeviceIndex>(device_id));
    void* cuda_stream_ptr = reinterpret_cast<void*>(stream.stream());

    int64_t curr_decoded_idx = -1;
    int64_t curr_key_idx = -1;
    bool draining = false;
    bool waiting_first_frame_after_seek = false;
    std::chrono::steady_clock::time_point t_seek_done;

    for (size_t out_pos = 0; out_pos < uniq_frame_ids.size(); ++out_pos) {
      const int64_t target_idx = uniq_frame_ids[out_pos];
      const int64_t target_key_idx = locate_keyframe_index(target_idx, key_indices);

      if (curr_decoded_idx < 0 || target_idx < curr_decoded_idx ||
          target_key_idx != curr_key_idx) {
        const int64_t target_key_pts =
            indexed_frames[static_cast<size_t>(target_key_idx)].pts;
        seek_to_frame_index(
            fmt_ctx, dec_ctx, video_stream_idx, target_key_idx, target_key_pts, seek_stats_ptr);
        curr_decoded_idx = target_key_idx - 1;
        curr_key_idx = target_key_idx;
        draining = false;
        waiting_first_frame_after_seek = true;
        t_seek_done = std::chrono::steady_clock::now();
      }

      while (true) {
        if (!pop_next_decoded_frame(
                fmt_ctx, dec_ctx, video_stream_idx, pkt, frame, &draining, seek_stats_ptr)) {
          throw std::runtime_error("failed to decode target frame " +
                                   std::to_string(target_idx));
        }
        if (seek_stats_ptr && waiting_first_frame_after_seek) {
          seek_stats_ptr->first_frame_latency_samples += 1;
          seek_stats_ptr->first_frame_latency_ms +=
              elapsed_ms(t_seek_done, std::chrono::steady_clock::now());
          waiting_first_frame_after_seek = false;
        }

        int64_t next_idx = curr_decoded_idx + 1;
        const int64_t frame_pts = frame->best_effort_timestamp;
        if (frame_pts != AV_NOPTS_VALUE) {
          auto it = pts_to_frame_idx.find(frame_pts);
          if (it != pts_to_frame_idx.end()) {
            const int64_t mapped_idx = it->second;
            if (mapped_idx >= next_idx && mapped_idx <= target_idx) {
              next_idx = mapped_idx;
            }
          }
        }
        curr_decoded_idx = next_idx;

        if (curr_decoded_idx < target_idx) {
          av_frame_unref(frame);
          continue;
        }
        if (curr_decoded_idx > target_idx) {
          av_frame_unref(frame);
          throw std::runtime_error("decoder overshot target frame " +
                                   std::to_string(target_idx));
        }

        const auto t_copy_begin = std::chrono::steady_clock::now();
        copy_cuda_nv12_frame_to_rgb(
            frame, &uniq_frames, static_cast<int64_t>(out_pos), width, height, cuda_stream_ptr);
        const auto t_copy_end = std::chrono::steady_clock::now();
        if (seek_stats_ptr) {
          seek_stats_ptr->copy_calls += 1;
          seek_stats_ptr->copy_ms += elapsed_ms(t_copy_begin, t_copy_end);
        }
        key_flags[out_pos] = indexed_frames[static_cast<size_t>(target_idx)].is_key
                                 ? static_cast<uint8_t>(1)
                                 : static_cast<uint8_t>(0);
        av_frame_unref(frame);
        break;
      }
    }

    const auto t_sync_begin = std::chrono::steady_clock::now();
    C10_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
    const auto t_sync_end = std::chrono::steady_clock::now();
    if (seek_stats_ptr) {
      seek_stats_ptr->sync_calls += 1;
      seek_stats_ptr->sync_ms += elapsed_ms(t_sync_begin, t_sync_end);
    }
    const auto t_after_decode = std::chrono::steady_clock::now();

    std::vector<int64_t> gather_pos(frame_ids.size(), 0);
    std::vector<uint8_t> is_i_positions(frame_ids.size(), 0);
    for (size_t i = 0; i < frame_ids.size(); ++i) {
      const int64_t fid = frame_ids[i];
      auto it = std::lower_bound(uniq_frame_ids.begin(), uniq_frame_ids.end(), fid);
      if (it == uniq_frame_ids.end() || *it != fid) {
        throw std::runtime_error("sampled frame id not found in decoded frame set");
      }
      const size_t pos = static_cast<size_t>(std::distance(uniq_frame_ids.begin(), it));
      gather_pos[i] = static_cast<int64_t>(pos);
      is_i_positions[i] = key_flags[pos];
    }

    bool any_i = false;
    for (uint8_t v : is_i_positions) {
      if (v != 0) {
        any_i = true;
        break;
      }
    }
    if (!any_i && !is_i_positions.empty()) {
      is_i_positions[0] = 1;
    }

    bool need_gather = frame_ids.size() != uniq_frame_ids.size();
    if (!need_gather) {
      for (size_t i = 0; i < frame_ids.size(); ++i) {
        if (frame_ids[i] != uniq_frame_ids[i]) {
          need_gather = true;
          break;
        }
      }
    }

    at::Tensor sampled_frames;
    if (need_gather) {
      auto gather = torch::from_blob(gather_pos.data(),
                                     {static_cast<int64_t>(gather_pos.size())},
                                     torch::TensorOptions().dtype(at::kLong))
                        .clone()
                        .to(torch::Device(torch::kCUDA, static_cast<int>(device_id)));
      sampled_frames = uniq_frames.index_select(0, gather);
    } else {
      sampled_frames = uniq_frames;
    }
    const auto t_after_post = std::chrono::steady_clock::now();

    const auto t_before_cleanup = std::chrono::steady_clock::now();
    av_frame_unref(frame);
    av_packet_unref(pkt);
    const auto t_after_cleanup_state = std::chrono::steady_clock::now();
    if (fmt_ctx) {
      avformat_close_input(&fmt_ctx);
    }
    const auto t_after_close_input = std::chrono::steady_clock::now();

    maybe_print_decode_profile(
        "decode_uniform_nvdec",
        {
            {"open_input", elapsed_ms(t_begin, t_after_open_input)},
            {"codec_prepare", elapsed_ms(t_after_open_input, t_after_codec_open)},
            {"index_and_setup", elapsed_ms(t_after_codec_open, t_after_setup)},
            {"decode", elapsed_ms(t_after_setup, t_after_decode)},
            {"post", elapsed_ms(t_after_decode, t_after_post)},
            {"cleanup_state", elapsed_ms(t_before_cleanup, t_after_cleanup_state)},
            {"cleanup_input", elapsed_ms(t_after_cleanup_state, t_after_close_input)},
            {"total", elapsed_ms(t_begin, t_after_close_input)},
        },
        sequence_length,
        width,
        height,
        total_frames);
    if (seek_stats_ptr) {
      maybe_print_nvdec_profile_detail(
          "decode_uniform_nvdec_detail",
          *seek_stats_ptr,
          sequence_length,
          width,
          height,
          total_frames);
    }

    DecodeResult out;
    out.frames_rgb_u8 = sampled_frames;
    out.sampled_frame_ids = std::move(frame_ids);
    out.is_i_positions = std::move(is_i_positions);
    out.fps = fps;
    out.duration_sec = duration_sec;
    out.width = width;
    out.height = height;
    return out;

  } catch (...) {
    if (persistent && persistent_lock.owns_lock()) {
      if (persistent->frame) {
        av_frame_unref(persistent->frame);
      }
      if (persistent->pkt) {
        av_packet_unref(persistent->pkt);
      }
      if (persistent->dec_ctx && persistent->decoder_open) {
        avcodec_flush_buffers(persistent->dec_ctx);
      }
    }
    if (fmt_ctx) {
      avformat_close_input(&fmt_ctx);
    }
    throw;
  }
}

bool warmup_nvdec_hw_device_ctx(int64_t device_id) {
  AVBufferRef* ctx = get_or_create_cached_hw_device_ctx(device_id);
  if (!ctx) {
    return false;
  }
  av_buffer_unref(&ctx);
  return true;
}

}  // namespace codec_patch_stream
