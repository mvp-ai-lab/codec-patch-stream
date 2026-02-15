#include <torch/extension.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/motion_vector.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "codec_patch_stream_cpu.h"

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
  const double step = static_cast<double>(total_frames - 1) /
                      static_cast<double>(sequence_length - 1);
  for (int64_t i = 0; i < sequence_length; ++i) {
    out[static_cast<size_t>(i)] = static_cast<int64_t>(i * step);
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

DecodeResult decode_sampled_frames_ffmpeg_cpu(const std::string& video_path,
                                              int64_t sequence_length) {
  AVFormatContext* fmt_ctx = nullptr;
  check_ff(avformat_open_input(&fmt_ctx, video_path.c_str(), nullptr, nullptr),
           "avformat_open_input");

  AVPacket* pkt = nullptr;
  AVFrame* frame = nullptr;
  AVCodecContext* dec_ctx = nullptr;
  SwsContext* sws_ctx = nullptr;

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
    dec_ctx->flags2 |= AV_CODEC_FLAG2_EXPORT_MVS;
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

    auto frame_opts = torch::TensorOptions().dtype(at::kByte).device(torch::kCPU);
    at::Tensor uniq_frames =
        torch::empty({static_cast<int64_t>(uniq_frame_ids.size()), height, width, 3},
                     frame_opts);
    auto mv_opts = torch::TensorOptions().dtype(at::kFloat).device(torch::kCPU);
    at::Tensor uniq_mv_magnitude =
        torch::zeros({static_cast<int64_t>(uniq_frame_ids.size()), height, width}, mv_opts);

    std::vector<uint8_t> captured(uniq_frame_ids.size(), 0);
    std::vector<uint8_t> key_flags(uniq_frame_ids.size(), 0);

    pkt = av_packet_alloc();
    frame = av_frame_alloc();
    if (!pkt || !frame) {
      throw std::runtime_error("failed to allocate AVPacket/AVFrame");
    }

    int64_t decoded_frame_idx = 0;
    size_t next_target = 0;
    size_t captured_count = 0;
    bool any_codec_mv = false;
    bool done = uniq_frame_ids.empty();

    auto process_frame = [&](AVFrame* f) {
      if (!done && next_target < uniq_frame_ids.size() &&
          decoded_frame_idx == uniq_frame_ids[next_target]) {
        if (f->width <= 0 || f->height <= 0) {
          throw std::runtime_error("decoded frame has invalid shape");
        }
        if (f->width != width || f->height != height) {
          throw std::runtime_error("variable frame size is not supported");
        }

        sws_ctx = sws_getCachedContext(sws_ctx,
                                       f->width,
                                       f->height,
                                       static_cast<AVPixelFormat>(f->format),
                                       static_cast<int>(width),
                                       static_cast<int>(height),
                                       AV_PIX_FMT_RGB24,
                                       SWS_BILINEAR,
                                       nullptr,
                                       nullptr,
                                       nullptr);
        if (!sws_ctx) {
          throw std::runtime_error("sws_getCachedContext failed");
        }

        const int64_t pos = static_cast<int64_t>(next_target);
        uint8_t* dst = uniq_frames.data_ptr<uint8_t>() + pos * height * width * 3;
        uint8_t* dst_data[4] = {dst, nullptr, nullptr, nullptr};
        int dst_linesize[4] = {static_cast<int>(width * 3), 0, 0, 0};
        const int scaled_h =
            sws_scale(sws_ctx, f->data, f->linesize, 0, f->height, dst_data, dst_linesize);
        if (scaled_h <= 0) {
          throw std::runtime_error("sws_scale failed");
        }

        // Exported codec motion vectors (if present).
        float* mv_dst = uniq_mv_magnitude.data_ptr<float>() + pos * height * width;
        std::fill_n(mv_dst, static_cast<size_t>(height * width), 0.0f);
        AVFrameSideData* mv_side = av_frame_get_side_data(f, AV_FRAME_DATA_MOTION_VECTORS);
        if (mv_side && mv_side->data &&
            mv_side->size >= static_cast<int>(sizeof(AVMotionVector))) {
          bool frame_has_mv = false;
          const int mv_count = mv_side->size / static_cast<int>(sizeof(AVMotionVector));
          auto* vectors = reinterpret_cast<const AVMotionVector*>(mv_side->data);
          for (int i = 0; i < mv_count; ++i) {
            const AVMotionVector& mv = vectors[i];
            if (mv.motion_scale == 0) {
              continue;
            }

            const float mx = static_cast<float>(mv.motion_x) /
                             static_cast<float>(mv.motion_scale);
            const float my = static_cast<float>(mv.motion_y) /
                             static_cast<float>(mv.motion_scale);
            const float mag = std::sqrt(mx * mx + my * my);

            const int x0 = std::max<int>(0, mv.dst_x);
            const int y0 = std::max<int>(0, mv.dst_y);
            const int x1 = std::min<int>(static_cast<int>(width),
                                         static_cast<int>(mv.dst_x) + static_cast<int>(mv.w));
            const int y1 = std::min<int>(static_cast<int>(height),
                                         static_cast<int>(mv.dst_y) + static_cast<int>(mv.h));
            if (x1 <= x0 || y1 <= y0) {
              continue;
            }
            frame_has_mv = true;

            for (int y = y0; y < y1; ++y) {
              float* row = mv_dst + static_cast<int64_t>(y) * width;
              for (int x = x0; x < x1; ++x) {
                row[x] = std::max(row[x], mag);
              }
            }
          }
          any_codec_mv = any_codec_mv || frame_has_mv;
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
    at::Tensor sampled_mv_magnitude;
    if (need_gather) {
      auto gather = torch::from_blob(gather_pos.data(),
                                     {static_cast<int64_t>(gather_pos.size())},
                                     torch::TensorOptions().dtype(at::kLong))
                        .clone();
      sampled_frames = uniq_frames.index_select(0, gather);
      sampled_mv_magnitude = uniq_mv_magnitude.index_select(0, gather);
    } else {
      sampled_frames = uniq_frames;
      sampled_mv_magnitude = uniq_mv_magnitude;
    }
    if (!any_codec_mv) {
      sampled_mv_magnitude = at::Tensor();
    }

    if (sws_ctx) {
      sws_freeContext(sws_ctx);
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
    if (fmt_ctx) {
      avformat_close_input(&fmt_ctx);
    }

    DecodeResult out;
    out.frames_rgb_u8 = sampled_frames;
    out.mv_magnitude_maps = sampled_mv_magnitude;
    out.sampled_frame_ids = std::move(frame_ids);
    out.is_i_positions = std::move(is_i_positions);
    out.width = width;
    out.height = height;
    return out;

  } catch (...) {
    if (sws_ctx) {
      sws_freeContext(sws_ctx);
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
    if (fmt_ctx) {
      avformat_close_input(&fmt_ctx);
    }
    throw;
  }
}

}  // namespace codec_patch_stream
