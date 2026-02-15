#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

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

float normalize_static_abs_thresh(double raw) {
  double v = raw;
  if (v > 1.0) {
    v /= 255.0;
  }
  v = std::max(0.0, std::min(1.0, v));
  return static_cast<float>(v);
}

}  // namespace

SelectionResult compute_visible_indices_cuda(const at::Tensor& energy,
                                             int64_t patch_size,
                                             int64_t k_keep,
                                             bool static_fallback,
                                             double static_abs_thresh,
                                             double static_rel_thresh,
                                             int64_t static_uniform_frames,
                                             const std::vector<uint8_t>& is_i_positions) {
  TORCH_CHECK(energy.is_cuda(), "energy must be on CUDA");
  TORCH_CHECK(energy.scalar_type() == at::kFloat, "energy must be float32");
  TORCH_CHECK(energy.dim() == 3, "energy must have shape (T,H,W)");

  const int64_t t = energy.size(0);
  const int64_t h = energy.size(1);
  const int64_t w = energy.size(2);
  TORCH_CHECK(static_cast<int64_t>(is_i_positions.size()) == t,
              "is_i_positions size mismatch");
  TORCH_CHECK(patch_size > 0, "patch_size must be > 0");
  TORCH_CHECK(h % patch_size == 0 && w % patch_size == 0,
              "H/W must be divisible by patch_size");

  const int64_t hb = h / patch_size;
  const int64_t wb = w / patch_size;
  const int64_t patches_per_frame = hb * wb;
  const int64_t l = t * hb * wb;
  const int64_t k_global = std::max<int64_t>(0, std::min<int64_t>(k_keep, l));

  auto scores_flat = energy.view({t, hb, patch_size, wb, patch_size})
                         .sum({2, 4})
                         .reshape({l})
                         .contiguous();

  auto long_opts = torch::TensorOptions().dtype(at::kLong).device(energy.device());
  auto float_opts = torch::TensorOptions().dtype(at::kFloat).device(energy.device());
  if (k_global == 0) {
    auto empty = torch::empty({0}, long_opts);
    return SelectionResult{empty, scores_flat, hb, wb};
  }

  auto is_i_frame_cpu = torch::from_blob(const_cast<uint8_t*>(is_i_positions.data()),
                                         {t},
                                         torch::TensorOptions().dtype(at::kByte))
                            .clone();
  auto is_i_frame = is_i_frame_cpu.to(energy.device()).toType(at::kBool);

  auto patch_ids = torch::arange(l, long_opts);
  auto patch_frames =
      torch::arange(t, long_opts).repeat_interleave(patches_per_frame).contiguous();
  auto patch_is_i = is_i_frame.index_select(0, patch_frames);

  auto i_indices = patch_ids.masked_select(patch_is_i);
  auto p_indices = patch_ids.masked_select(patch_is_i.logical_not());

  const int64_t i_count = i_indices.numel();
  const int64_t p_count = p_indices.numel();

  if (i_count == 0) {
    auto global_topk =
        std::get<1>(at::topk(scores_flat, k_global, 0, true, false)).toType(at::kLong);
    auto sorted = std::get<0>(at::sort(global_topk, 0, false));
    return SelectionResult{sorted, scores_flat, hb, wb};
  }

  if (i_count >= k_global) {
    auto i_scores = scores_flat.index_select(0, i_indices);
    auto i_topk_rel =
        std::get<1>(at::topk(i_scores, k_global, 0, true, false)).toType(at::kLong);
    auto i_selected = i_indices.index_select(0, i_topk_rel);
    auto sorted_i = std::get<0>(at::sort(i_selected, 0, false));
    return SelectionResult{sorted_i, scores_flat, hb, wb};
  }

  const int64_t remain = k_global - i_count;
  if (p_count == 0 || remain <= 0) {
    auto sorted_i = std::get<0>(at::sort(i_indices, 0, false));
    return SelectionResult{sorted_i, scores_flat, hb, wb};
  }

  auto p_scores = scores_flat.index_select(0, p_indices);
  auto p_scores_for_topk = p_scores;

  if (static_fallback && static_uniform_frames > 0) {
    const float abs_thresh = normalize_static_abs_thresh(static_abs_thresh);
    const float rel_thresh = static_cast<float>(std::max(0.0, static_rel_thresh));

    auto patch_mean = scores_flat / static_cast<float>(patch_size * patch_size);
    const int64_t k95 = percentile_rank_index(l, 95.0) + 1;
    const int64_t k50 = percentile_rank_index(l, 50.0) + 1;
    auto p95 = std::get<0>(patch_mean.kthvalue(k95, 0, false));
    auto p50 = std::get<0>(patch_mean.kthvalue(k50, 0, false));

    auto eps = torch::full({}, 1e-6f, float_opts);
    auto t_abs = torch::full({}, abs_thresh, float_opts);
    auto t_rel = torch::full({}, rel_thresh, float_opts);
    auto rel_contrast = (p95 - p50) / at::maximum(p95, eps);
    auto static_cond = (p95 < t_abs) | ((p95 < (t_abs * 2.0f)) & (rel_contrast < t_rel));
    auto static_scale = static_cond.toType(at::kFloat);

    auto p_frame_ids = torch::arange(t, long_opts).masked_select(is_i_frame.logical_not());
    const int64_t p_frame_count = p_frame_ids.numel();
    if (p_frame_count > 0) {
      const int64_t f_u =
          std::max<int64_t>(1, std::min<int64_t>(static_uniform_frames, p_frame_count));
      auto uniform_pos = (torch::arange(f_u, long_opts) * p_frame_count) / f_u;
      auto uniform_frames = p_frame_ids.index_select(0, uniform_pos);

      auto p_patch_frames = patch_frames.index_select(0, p_indices);
      auto uniform_hit = p_patch_frames.unsqueeze(1).eq(uniform_frames.unsqueeze(0)).any(1);

      auto bonus = (p_scores.max() - p_scores.min() + 1.0f) * static_scale;
      p_scores_for_topk = p_scores + uniform_hit.toType(at::kFloat) * bonus;
    }
  }

  auto p_topk_rel =
      std::get<1>(at::topk(p_scores_for_topk, remain, 0, true, false)).toType(at::kLong);
  auto p_selected = p_indices.index_select(0, p_topk_rel);

  auto merged = at::cat({i_indices, p_selected}, 0);
  auto sorted_merged = std::get<0>(at::sort(merged, 0, false));
  return SelectionResult{sorted_merged, scores_flat, hb, wb};
}

}  // namespace codec_patch_stream
