#include <ATen/ops/upsample_bilinear2d.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "codec_patch_stream.h"
#include "decode_core.h"

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

std::vector<PatchMeta> build_patch_meta_from_cpu_tensors(const at::Tensor& fields_cpu,
                                                          const at::Tensor& scores_cpu) {
  TORCH_CHECK(fields_cpu.scalar_type() == at::kLong,
              "metadata fields must be int64 on CPU");
  TORCH_CHECK(scores_cpu.scalar_type() == at::kFloat,
              "metadata scores must be float32 on CPU");
  TORCH_CHECK(fields_cpu.dim() == 2 && fields_cpu.size(1) == 6,
              "metadata fields must have shape (N,6)");
  TORCH_CHECK(scores_cpu.dim() == 1 && scores_cpu.size(0) == fields_cpu.size(0),
              "metadata scores must have shape (N)");

  const int64_t n = fields_cpu.size(0);
  std::vector<PatchMeta> out;
  out.reserve(static_cast<size_t>(n));

  auto fields_ptr = fields_cpu.data_ptr<int64_t>();
  auto scores_ptr = scores_cpu.data_ptr<float>();
  for (int64_t i = 0; i < n; ++i) {
    const int64_t* row = fields_ptr + i * 6;
    PatchMeta m;
    m.seq_pos = row[0];
    m.frame_id = row[1];
    m.is_i = row[2] != 0;
    m.patch_linear_idx = row[3];
    m.patch_h_idx = row[4];
    m.patch_w_idx = row[5];
    m.score = scores_ptr[i];
    out.push_back(m);
  }
  return out;
}

int64_t squared_pixels_from_input_size(int64_t input_size) {
  TORCH_CHECK(input_size > 0, "input_size must be > 0");
  if (input_size > std::numeric_limits<int64_t>::max() / input_size) {
    return std::numeric_limits<int64_t>::max();
  }
  return input_size * input_size;
}

std::tuple<int64_t, int64_t> smart_resize_shape(int64_t height,
                                                int64_t width,
                                                int64_t patch_size,
                                                int64_t min_pixels,
                                                int64_t max_pixels,
                                                int64_t align_patch_size) {
  TORCH_CHECK(height > 0 && width > 0, "invalid shape: h=", height, ", w=", width);
  TORCH_CHECK(patch_size > 0, "patch_size must be > 0");

  const int64_t align_size = align_patch_size > 0 ? align_patch_size : patch_size;
  TORCH_CHECK(align_size > 0, "align_size must be > 0");

  const double pixels = static_cast<double>(height) * static_cast<double>(width);
  double scale = 1.0;
  if (min_pixels > 0 && pixels < static_cast<double>(min_pixels)) {
    scale = std::sqrt(static_cast<double>(min_pixels) / pixels);
  }
  if (max_pixels > 0 && pixels > static_cast<double>(max_pixels)) {
    scale = std::sqrt(static_cast<double>(max_pixels) / pixels);
  }

  auto aligned = [align_size](double x) -> int64_t {
    const int64_t rounded = static_cast<int64_t>(
        std::llround(x / static_cast<double>(align_size)));
    return std::max<int64_t>(align_size, rounded * align_size);
  };

  const int64_t resized_h = aligned(static_cast<double>(height) * scale);
  const int64_t resized_w = aligned(static_cast<double>(width) * scale);
  return std::make_tuple(resized_h, resized_w);
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
                                int64_t target_h,
                                int64_t target_w,
                                bool is_energy) {
  TORCH_CHECK(frames_or_energy.is_cuda(), "input must be CUDA tensor");
  TORCH_CHECK(target_h > 0 && target_w > 0, "target_h/target_w must be > 0");

  at::Tensor x;
  int64_t input_h = 0;
  int64_t input_w = 0;

  if (is_energy) {
    TORCH_CHECK(frames_or_energy.dim() == 3,
                "energy tensor must have shape (T,H,W)");
    input_h = frames_or_energy.size(1);
    input_w = frames_or_energy.size(2);
    x = frames_or_energy.unsqueeze(1);
  } else {
    TORCH_CHECK(frames_or_energy.dim() == 4 && frames_or_energy.size(3) == 3,
                "frame tensor must have shape (T,H,W,3)");
    input_h = frames_or_energy.size(1);
    input_w = frames_or_energy.size(2);
    x = frames_or_energy.permute({0, 3, 1, 2});
  }

  if (input_h == target_h && input_w == target_w) {
    return frames_or_energy.contiguous();
  }

  auto resized = at::upsample_bilinear2d(
      x, {target_h, target_w}, /*align_corners=*/false, c10::nullopt, c10::nullopt);

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
  DecodeRequest req;
  req.video_path = video_path_;
  req.sequence_length = cfg_.sequence_length;
  req.backend = "gpu";
  req.device_id = cfg_.device_id;
  req.decode_mode = cfg_.decode_mode;
  req.uniform_strategy = cfg_.uniform_strategy;
  req.nvdec_session_pool_size = cfg_.nvdec_session_pool_size;
  req.uniform_auto_ratio = cfg_.uniform_auto_ratio;
  req.decode_threads = cfg_.decode_threads;
  req.decode_thread_type = cfg_.decode_thread_type;
  req.reader_cache_size = cfg_.reader_cache_size;
  req.nvdec_reuse_open_decoder = cfg_.nvdec_reuse_open_decoder;
  auto decoded = decode_only_native_gpu(req);

  const int64_t default_pixels = squared_pixels_from_input_size(cfg_.input_size);
  const int64_t min_pixels = cfg_.min_pixels > 0 ? cfg_.min_pixels : default_pixels;
  const int64_t max_pixels = cfg_.max_pixels > 0 ? cfg_.max_pixels : default_pixels;
  const auto [target_h, target_w] =
      smart_resize_shape(decoded.frames_rgb_u8.size(1),
                         decoded.frames_rgb_u8.size(2),
                         cfg_.patch_size,
                         min_pixels,
                         max_pixels,
                         cfg_.patch_size * 2);

  auto energy =
      compute_energy_maps_cuda(decoded.frames_rgb_u8, decoded.is_i_positions, cfg_.energy_pct);
  energy = resize_to_input_cuda(energy, target_h, target_w, true);

  auto resized_frames =
      resize_to_input_cuda(decoded.frames_rgb_u8.to(at::kFloat), target_h, target_w, false);

  auto selection = compute_visible_indices_cuda(energy,
                                                cfg_.patch_size,
                                                cfg_.k_keep,
                                                cfg_.selection_unit,
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
  metadata_cached_ = false;
  const int64_t ppf = selection.hb * selection.wb;
  auto long_opts = torch::TensorOptions()
                       .dtype(at::kLong)
                       .device(torch::Device(torch::kCUDA, cfg_.device_id));
  auto cuda_device = torch::Device(torch::kCUDA, cfg_.device_id);

  auto visible = selection.visible_indices.contiguous();
  const int64_t t = static_cast<int64_t>(decoded.sampled_frame_ids.size());
  auto patch_frames = torch::arange(t, long_opts).repeat_interleave(ppf).contiguous();
  auto seq_pos = patch_frames.index_select(0, visible);

  auto rem = visible.remainder(ppf);
  auto patch_h_lut = torch::arange(selection.hb, long_opts)
                         .repeat_interleave(selection.wb)
                         .contiguous();
  auto patch_w_lut = torch::arange(selection.wb, long_opts).repeat({selection.hb}).contiguous();
  auto ph_idx = patch_h_lut.index_select(0, rem);
  auto pw_idx = patch_w_lut.index_select(0, rem);

  auto frame_ids_cpu = torch::from_blob(const_cast<int64_t*>(decoded.sampled_frame_ids.data()),
                                        {t},
                                        torch::TensorOptions().dtype(at::kLong))
                           .clone();
  auto frame_ids_gpu = frame_ids_cpu.to(cuda_device).contiguous();
  auto frame_ids = frame_ids_gpu.index_select(0, seq_pos);

  auto is_i_cpu = torch::from_blob(const_cast<uint8_t*>(decoded.is_i_positions.data()),
                                   {t},
                                   torch::TensorOptions().dtype(at::kByte))
                      .clone();
  auto is_i_gpu = is_i_cpu.to(cuda_device).toType(at::kLong).contiguous();
  auto is_i = is_i_gpu.index_select(0, seq_pos);

  metadata_fields_gpu_ =
      torch::stack({seq_pos, frame_ids, is_i, visible, ph_idx, pw_idx}, 1).contiguous();
  metadata_scores_gpu_ =
      selection.scores_flat.index_select(0, visible).toType(at::kFloat).contiguous();

  sampled_frame_ids_ = decoded.sampled_frame_ids;
  fps_ = decoded.fps;
  duration_sec_ = decoded.duration_sec;

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
  if (!metadata_cached_) {
    materialize_metadata_cpu_cache();
  }
  const int64_t i = cursor_;
  ++cursor_;
  return std::make_tuple(patch_bank_.index({i}), meta_.at(static_cast<size_t>(i)));
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
  std::vector<PatchMeta> metas;
  if (metadata_cached_) {
    metas = std::vector<PatchMeta>(meta_.begin() + cursor_, meta_.begin() + end);
  } else {
    metas = metadata_slice_cpu(cursor_, end);
  }
  cursor_ = end;
  return std::make_tuple(patches, std::move(metas));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> CodecPatchStreamNative::next_n_tensors(int64_t n) {
  if (n <= 0 || !patch_bank_.defined() || closed_ || cursor_ >= size()) {
    auto empty_patches = torch::empty({0, 3, cfg_.patch_size, cfg_.patch_size},
                                      torch::TensorOptions()
                                          .dtype(cfg_.output_dtype)
                                          .device(torch::Device(torch::kCUDA, cfg_.device_id)));
    auto empty_fields = torch::empty({0, 6},
                                     torch::TensorOptions()
                                         .dtype(at::kLong)
                                         .device(torch::Device(torch::kCUDA, cfg_.device_id)));
    auto empty_scores = torch::empty({0},
                                     torch::TensorOptions()
                                         .dtype(at::kFloat)
                                         .device(torch::Device(torch::kCUDA, cfg_.device_id)));
    return std::make_tuple(empty_patches, empty_fields, empty_scores);
  }

  const int64_t end = std::min<int64_t>(size(), cursor_ + n);
  auto patches = patch_bank_.slice(0, cursor_, end);
  auto fields = metadata_fields_gpu_.slice(0, cursor_, end);
  auto scores = metadata_scores_gpu_.slice(0, cursor_, end);
  cursor_ = end;
  return std::make_tuple(patches, fields, scores);
}

void CodecPatchStreamNative::reset() { cursor_ = 0; }

void CodecPatchStreamNative::close() {
  patch_bank_ = at::Tensor();
  metadata_fields_gpu_ = at::Tensor();
  metadata_scores_gpu_ = at::Tensor();
  sampled_frame_ids_.clear();
  fps_ = 0.0;
  duration_sec_ = 0.0;
  meta_.clear();
  metadata_cached_ = false;
  cursor_ = 0;
  closed_ = true;
}

const at::Tensor& CodecPatchStreamNative::patch_bank() const { return patch_bank_; }

const std::vector<PatchMeta>& CodecPatchStreamNative::metadata() const {
  materialize_metadata_cpu_cache();
  return meta_;
}

const at::Tensor& CodecPatchStreamNative::metadata_fields_gpu() const {
  return metadata_fields_gpu_;
}

const at::Tensor& CodecPatchStreamNative::metadata_scores_gpu() const {
  return metadata_scores_gpu_;
}

const std::vector<int64_t>& CodecPatchStreamNative::sampled_frame_ids() const {
  return sampled_frame_ids_;
}

double CodecPatchStreamNative::fps() const { return fps_; }

double CodecPatchStreamNative::duration_sec() const { return duration_sec_; }

std::vector<PatchMeta> CodecPatchStreamNative::metadata_slice_cpu(int64_t begin,
                                                                  int64_t end) const {
  if (!metadata_fields_gpu_.defined() || end <= begin) {
    return {};
  }
  auto fields_cpu =
      metadata_fields_gpu_.slice(0, begin, end).to(torch::kCPU, /*non_blocking=*/false).contiguous();
  auto scores_cpu =
      metadata_scores_gpu_.slice(0, begin, end).to(torch::kCPU, /*non_blocking=*/false).contiguous();
  return build_patch_meta_from_cpu_tensors(fields_cpu, scores_cpu);
}

void CodecPatchStreamNative::materialize_metadata_cpu_cache() const {
  if (metadata_cached_) {
    return;
  }
  meta_.clear();
  if (size() <= 0 || !metadata_fields_gpu_.defined()) {
    metadata_cached_ = true;
    return;
  }
  meta_ = metadata_slice_cpu(0, size());
  metadata_cached_ = true;
}

const char* version() { return "0.3.0"; }

}  // namespace codec_patch_stream
