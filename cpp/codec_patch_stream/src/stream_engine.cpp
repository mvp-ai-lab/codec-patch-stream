#include <ATen/ops/roll.h>
#include <ATen/ops/upsample_bilinear2d.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "codec_patch_stream.h"

namespace codec_patch_stream {
namespace {

int64_t percentile_rank_index(int64_t n, double pct) {
  if (n <= 0) {
    return 0;
  }
  const double clamped = std::max(1.0, std::min(100.0, pct));
  const double rank = clamped / 100.0 * static_cast<double>(n - 1);
  const int64_t idx = static_cast<int64_t>(std::llround(rank));
  return std::max<int64_t>(0, std::min<int64_t>(n - 1, idx));
}

at::Tensor normalize_per_frame_percentile_cuda(const at::Tensor& maps, double pct) {
  TORCH_CHECK(maps.is_cuda(), "maps must be CUDA tensor");
  TORCH_CHECK(maps.scalar_type() == at::kFloat, "maps must be float32");
  TORCH_CHECK(maps.dim() == 3, "maps must have shape (T,H,W)");

  const int64_t t = maps.size(0);
  const int64_t h = maps.size(1);
  const int64_t w = maps.size(2);
  const int64_t n = h * w;
  if (t == 0 || n == 0) {
    return maps.clone();
  }

  const int64_t kth = percentile_rank_index(n, pct) + 1;
  auto flat = maps.reshape({t, n});
  auto kth_vals = std::get<0>(flat.kthvalue(kth, 1, false));
  auto denom = kth_vals.clamp_min(1e-6f).view({t, 1, 1});
  return (maps / denom).clamp(0.0f, 1.0f).contiguous();
}

std::tuple<at::Tensor, at::Tensor> compute_motion_residual_proxy_small_cuda(
    const at::Tensor& luma_small, int64_t search_radius) {
  using torch::indexing::Slice;

  TORCH_CHECK(luma_small.is_cuda(), "luma_small must be CUDA tensor");
  TORCH_CHECK(luma_small.scalar_type() == at::kFloat, "luma_small must be float32");
  TORCH_CHECK(luma_small.dim() == 3, "luma_small must have shape (T,H,W)");

  const int64_t t = luma_small.size(0);
  const int64_t hs = luma_small.size(1);
  const int64_t ws = luma_small.size(2);

  auto mv_proxy = torch::zeros_like(luma_small);
  auto residual_proxy = torch::zeros_like(luma_small);
  if (t <= 1) {
    return std::make_tuple(mv_proxy, residual_proxy);
  }

  auto cur = luma_small.index({Slice(1, t), Slice(), Slice()});
  auto prev = luma_small.index({Slice(0, t - 1), Slice(), Slice()});

  auto best_err = torch::full_like(cur, 1e9f);
  auto best_mag = torch::zeros_like(cur);

  for (int64_t dy = -search_radius; dy <= search_radius; ++dy) {
    for (int64_t dx = -search_radius; dx <= search_radius; ++dx) {
      auto shifted = torch::roll(prev, {dy, dx}, {1, 2});
      auto err = (cur - shifted).abs();

      if (dy > 0) {
        err.index_put_({Slice(), Slice(0, dy), Slice()}, 1e9f);
      } else if (dy < 0) {
        err.index_put_({Slice(), Slice(hs + dy, hs), Slice()}, 1e9f);
      }
      if (dx > 0) {
        err.index_put_({Slice(), Slice(), Slice(0, dx)}, 1e9f);
      } else if (dx < 0) {
        err.index_put_({Slice(), Slice(), Slice(ws + dx, ws)}, 1e9f);
      }

      auto better = err < best_err;
      best_err = torch::where(better, err, best_err);

      const float mag = std::sqrt(static_cast<float>(dx * dx + dy * dy));
      best_mag = torch::where(better, torch::full_like(best_mag, mag), best_mag);
    }
  }

  mv_proxy.index_put_({Slice(1, t), Slice(), Slice()}, best_mag);
  residual_proxy.index_put_({Slice(1, t), Slice(), Slice()}, best_err);
  return std::make_tuple(mv_proxy.contiguous(), residual_proxy.contiguous());
}

}  // namespace

at::Tensor compute_energy_maps_cuda(const at::Tensor& frames_rgb_u8,
                                    const std::vector<uint8_t>& is_i_positions,
                                    double energy_pct) {
  using torch::indexing::Slice;

  TORCH_CHECK(frames_rgb_u8.is_cuda(), "frames_rgb_u8 must be CUDA tensor");
  TORCH_CHECK(frames_rgb_u8.scalar_type() == at::kByte,
              "frames_rgb_u8 must be uint8 tensor");
  TORCH_CHECK(frames_rgb_u8.dim() == 4 && frames_rgb_u8.size(3) == 3,
              "frames_rgb_u8 must have shape (T,H,W,3)");

  const int64_t t = frames_rgb_u8.size(0);
  const int64_t h = frames_rgb_u8.size(1);
  const int64_t w = frames_rgb_u8.size(2);
  TORCH_CHECK(static_cast<int64_t>(is_i_positions.size()) == t,
              "is_i_positions size mismatch");

  auto rgb = frames_rgb_u8.to(at::kFloat);
  auto luma = 0.299f * rgb.index({Slice(), Slice(), Slice(), 0}) +
              0.587f * rgb.index({Slice(), Slice(), Slice(), 1}) +
              0.114f * rgb.index({Slice(), Slice(), Slice(), 2});

  constexpr int64_t kDownsample = 4;
  constexpr int64_t kMinSide = 32;
  int64_t hs = std::max<int64_t>(kMinSide, h / kDownsample);
  int64_t ws = std::max<int64_t>(kMinSide, w / kDownsample);
  hs = std::min<int64_t>(h, hs);
  ws = std::min<int64_t>(w, ws);
  hs = std::max<int64_t>(1, hs);
  ws = std::max<int64_t>(1, ws);

  auto luma_small = at::upsample_bilinear2d(
                        luma.unsqueeze(1), {hs, ws}, /*align_corners=*/false, c10::nullopt, c10::nullopt)
                        .index({Slice(), 0, Slice(), Slice()})
                        .contiguous();

  const int64_t max_radius = std::max<int64_t>(1, std::min<int64_t>(hs, ws) / 2);
  const int64_t search_radius = std::max<int64_t>(1, std::min<int64_t>(4, max_radius));
  auto proxies = compute_motion_residual_proxy_small_cuda(luma_small, search_radius);
  auto mv_proxy_small = std::get<0>(proxies);
  auto residual_proxy_small = std::get<1>(proxies);

  auto mv_norm_small = normalize_per_frame_percentile_cuda(mv_proxy_small, energy_pct);
  auto residual_norm_small = normalize_per_frame_percentile_cuda(residual_proxy_small, energy_pct);
  auto fused_small = (mv_norm_small + residual_norm_small) * 0.5f;

  auto fused = at::upsample_bilinear2d(fused_small.unsqueeze(1),
                                       {h, w},
                                       /*align_corners=*/false,
                                       c10::nullopt,
                                       c10::nullopt)
                   .index({Slice(), 0, Slice(), Slice()})
                   .clamp(0.0f, 1.0f)
                   .contiguous();

  return fused;
}

at::Tensor resize_to_input_cuda(const at::Tensor& frames_or_energy,
                                int64_t input_size,
                                bool is_energy) {
  TORCH_CHECK(frames_or_energy.is_cuda(), "input must be CUDA tensor");
  TORCH_CHECK(input_size > 0, "input_size must be > 0");

  at::Tensor x;

  if (is_energy) {
    TORCH_CHECK(frames_or_energy.dim() == 3,
                "energy tensor must have shape (T,H,W)");
    x = frames_or_energy.unsqueeze(1);
  } else {
    TORCH_CHECK(frames_or_energy.dim() == 4 && frames_or_energy.size(3) == 3,
                "frame tensor must have shape (T,H,W,3)");
    x = frames_or_energy.permute({0, 3, 1, 2});
  }

  auto resized = at::upsample_bilinear2d(
      x, {input_size, input_size}, /*align_corners=*/false, c10::nullopt, c10::nullopt);

  if (is_energy) {
    return resized
        .index({torch::indexing::Slice(),
                0,
                torch::indexing::Slice(),
                torch::indexing::Slice()})
        .contiguous();
  }
  return resized.permute({0, 2, 3, 1}).contiguous();
}

std::tuple<int64_t, int64_t, int64_t> linear_to_thw(int64_t linear_idx,
                                                     int64_t patches_per_frame,
                                                     int64_t patches_w) {
  if (patches_per_frame <= 0 || patches_w <= 0) {
    throw std::invalid_argument("patches_per_frame and patches_w must be > 0");
  }
  const int64_t t = linear_idx / patches_per_frame;
  const int64_t rem = linear_idx % patches_per_frame;
  const int64_t h = rem / patches_w;
  const int64_t w = rem % patches_w;
  return std::make_tuple(t, h, w);
}

CodecPatchStreamNative::CodecPatchStreamNative(const std::string& video_path,
                                               const StreamConfig& cfg)
    : video_path_(video_path), cfg_(cfg) {
  prepare();
}

void CodecPatchStreamNative::prepare() {
  auto decoded = decode_sampled_frames_nvdec(video_path_, cfg_.sequence_length, cfg_.device_id);

  auto energy =
      compute_energy_maps_cuda(decoded.frames_rgb_u8, decoded.is_i_positions, cfg_.energy_pct);
  energy = resize_to_input_cuda(energy, cfg_.input_size, true);

  auto resized_frames =
      resize_to_input_cuda(decoded.frames_rgb_u8.to(at::kFloat), cfg_.input_size, false);

  auto selection = compute_visible_indices_cuda(energy,
                                                cfg_.patch_size,
                                                cfg_.k_keep,
                                                cfg_.static_fallback,
                                                cfg_.static_abs_thresh,
                                                cfg_.static_rel_thresh,
                                                cfg_.static_uniform_frames,
                                                decoded.is_i_positions);

  auto patches_f32 = extract_patches_by_indices_cuda(resized_frames.contiguous(),
                                                     selection.visible_indices.contiguous(),
                                                     cfg_.patch_size,
                                                     selection.wb);
  patch_bank_ = patches_f32.to(cfg_.output_dtype);

  meta_.clear();
  const int64_t ppf = selection.hb * selection.wb;

  auto vis_cpu = selection.visible_indices.to(torch::kCPU, /*non_blocking=*/false).contiguous();
  auto selected_scores_cpu = selection.scores_flat.index_select(0, selection.visible_indices)
                                 .to(torch::kCPU, /*non_blocking=*/false)
                                 .contiguous();

  const int64_t n = vis_cpu.numel();
  meta_.reserve(static_cast<size_t>(n));

  auto vis_ptr = vis_cpu.data_ptr<int64_t>();
  auto score_ptr = selected_scores_cpu.data_ptr<float>();

  for (int64_t i = 0; i < n; ++i) {
    const int64_t idx = vis_ptr[i];
    const int64_t seq_pos = idx / ppf;
    const int64_t rem = idx % ppf;
    const int64_t ph_idx = rem / selection.wb;
    const int64_t pw_idx = rem % selection.wb;

    PatchMeta m;
    m.seq_pos = seq_pos;
    m.frame_id = decoded.sampled_frame_ids[static_cast<size_t>(seq_pos)];
    m.is_i = decoded.is_i_positions[static_cast<size_t>(seq_pos)] != 0;
    m.patch_linear_idx = idx;
    m.patch_h_idx = ph_idx;
    m.patch_w_idx = pw_idx;
    m.score = score_ptr[i];
    meta_.push_back(m);
  }

  cursor_ = 0;
  closed_ = false;
}

int64_t CodecPatchStreamNative::size() const {
  if (!patch_bank_.defined()) {
    return 0;
  }
  return patch_bank_.size(0);
}

bool CodecPatchStreamNative::has_next() const {
  return !closed_ && cursor_ < size();
}

std::tuple<at::Tensor, PatchMeta> CodecPatchStreamNative::next() {
  if (!has_next()) {
    throw std::out_of_range("patch stream exhausted");
  }
  const int64_t i = cursor_;
  ++cursor_;
  return std::make_tuple(patch_bank_.index({i}), meta_[static_cast<size_t>(i)]);
}

std::tuple<at::Tensor, std::vector<PatchMeta>> CodecPatchStreamNative::next_n(int64_t n) {
  if (n <= 0 || !patch_bank_.defined() || closed_ || cursor_ >= size()) {
    auto empty = torch::empty({0, 3, cfg_.patch_size, cfg_.patch_size},
                              torch::TensorOptions()
                                  .dtype(cfg_.output_dtype)
                                  .device(torch::Device(torch::kCUDA, cfg_.device_id)));
    return std::make_tuple(empty, std::vector<PatchMeta>{});
  }

  const int64_t end = std::min<int64_t>(size(), cursor_ + n);
  auto patches = patch_bank_.slice(0, cursor_, end);
  std::vector<PatchMeta> metas(meta_.begin() + cursor_, meta_.begin() + end);
  cursor_ = end;
  return std::make_tuple(patches, std::move(metas));
}

void CodecPatchStreamNative::reset() { cursor_ = 0; }

void CodecPatchStreamNative::close() {
  patch_bank_ = at::Tensor();
  meta_.clear();
  cursor_ = 0;
  closed_ = true;
}

const at::Tensor& CodecPatchStreamNative::patch_bank() const { return patch_bank_; }

const std::vector<PatchMeta>& CodecPatchStreamNative::metadata() const { return meta_; }

const char* version() { return "0.2.0"; }

}  // namespace codec_patch_stream
