#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixfmt.h>
}

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>
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

void check_ff(int ret, const char* where) {
  if (ret < 0) {
    throw std::runtime_error(std::string(where) + " failed: " + ff_err_str(ret));
  }
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

}  // namespace

DecodeResult decode_sampled_frames_nvdec(const std::string& video_path,
                                         int64_t sequence_length,
                                         int64_t device_id) {
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_id));

  AVFormatContext* fmt_ctx = nullptr;
  check_ff(avformat_open_input(&fmt_ctx, video_path.c_str(), nullptr, nullptr),
           "avformat_open_input");

  AVPacket* pkt = nullptr;
  AVFrame* frame = nullptr;
  AVCodecContext* dec_ctx = nullptr;
  AVBufferRef* hw_device_ctx = nullptr;

  try {
    check_ff(avformat_find_stream_info(fmt_ctx, nullptr), "avformat_find_stream_info");

    const int video_stream_idx =
        av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    check_ff(video_stream_idx, "av_find_best_stream");

    AVStream* video_stream = fmt_ctx->streams[video_stream_idx];
    const AVCodec* codec = avcodec_find_decoder(video_stream->codecpar->codec_id);
    if (!codec) {
      throw std::runtime_error("avcodec_find_decoder returned null");
    }

    dec_ctx = avcodec_alloc_context3(codec);
    if (!dec_ctx) {
      throw std::runtime_error("avcodec_alloc_context3 failed");
    }

    check_ff(avcodec_parameters_to_context(dec_ctx, video_stream->codecpar),
             "avcodec_parameters_to_context");
    dec_ctx->get_format = get_hw_format;

    char device_name[16];
    std::snprintf(device_name, sizeof(device_name), "%lld",
                  static_cast<long long>(device_id));
    check_ff(av_hwdevice_ctx_create(&hw_device_ctx,
                                    AV_HWDEVICE_TYPE_CUDA,
                                    device_name,
                                    nullptr,
                                    0),
             "av_hwdevice_ctx_create(cuda)");
    dec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
    if (!dec_ctx->hw_device_ctx) {
      throw std::runtime_error("av_buffer_ref(hw_device_ctx) failed");
    }

    check_ff(avcodec_open2(dec_ctx, codec, nullptr), "avcodec_open2");

    const int64_t total_frames = estimate_total_frames(fmt_ctx, video_stream, dec_ctx);
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

    pkt = av_packet_alloc();
    frame = av_frame_alloc();
    if (!pkt || !frame) {
      throw std::runtime_error("failed to allocate AVPacket/AVFrame");
    }

    auto stream = at::cuda::getDefaultCUDAStream(static_cast<c10::DeviceIndex>(device_id));

    int64_t decoded_frame_idx = 0;
    size_t next_target = 0;
    size_t captured_count = 0;
    bool done = uniq_frame_ids.empty();

    auto process_frame = [&](AVFrame* f) {
      if (!done && next_target < uniq_frame_ids.size() &&
          decoded_frame_idx == uniq_frame_ids[next_target]) {
        if (f->format != AV_PIX_FMT_CUDA) {
          throw std::runtime_error("decoded frame is not AV_PIX_FMT_CUDA");
        }

        if (!f->data[0] || !f->data[1]) {
          throw std::runtime_error("CUDA frame plane pointers are null");
        }

        if (!f->hw_frames_ctx) {
          throw std::runtime_error("CUDA frame missing hw_frames_ctx");
        }

        auto* hw_frames_ctx = reinterpret_cast<AVHWFramesContext*>(f->hw_frames_ctx->data);
        if (!hw_frames_ctx || hw_frames_ctx->sw_format != AV_PIX_FMT_NV12) {
          throw std::runtime_error(
              "Only NV12 CUDA frames are supported in this implementation");
        }

        const int64_t pos = static_cast<int64_t>(next_target);
        uint8_t* dst = uniq_frames.data_ptr<uint8_t>() + pos * height * width * 3;
        launch_nv12_to_rgb_cuda(reinterpret_cast<const uint8_t*>(f->data[0]),
                                reinterpret_cast<const uint8_t*>(f->data[1]),
                                f->linesize[0],
                                f->linesize[1],
                                dst,
                                static_cast<int>(width * 3),
                                static_cast<int>(width),
                                static_cast<int>(height),
                                reinterpret_cast<void*>(stream.stream()));

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

    while (av_read_frame(fmt_ctx, pkt) >= 0) {
      if (pkt->stream_index == video_stream_idx) {
        check_ff(avcodec_send_packet(dec_ctx, pkt), "avcodec_send_packet");
        while (true) {
          const int ret = avcodec_receive_frame(dec_ctx, frame);
          if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
          }
          check_ff(ret, "avcodec_receive_frame");
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
      check_ff(avcodec_send_packet(dec_ctx, nullptr), "avcodec_send_packet(flush)");
      while (true) {
        const int ret = avcodec_receive_frame(dec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
          break;
        }
        check_ff(ret, "avcodec_receive_frame(flush)");
        process_frame(frame);
        av_frame_unref(frame);
        if (done) {
          break;
        }
      }
    }

    C10_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));

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
    if (fmt_ctx) {
      avformat_close_input(&fmt_ctx);
    }

    DecodeResult out;
    out.frames_rgb_u8 = sampled_frames;
    out.sampled_frame_ids = std::move(frame_ids);
    out.is_i_positions = std::move(is_i_positions);
    out.width = width;
    out.height = height;
    return out;

  } catch (...) {
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
    if (fmt_ctx) {
      avformat_close_input(&fmt_ctx);
    }
    throw;
  }
}

}  // namespace codec_patch_stream
