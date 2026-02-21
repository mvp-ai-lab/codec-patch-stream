#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "codec_patch_stream_cpu.h"

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

at::Tensor select_units_with_iframe_priority_cpu(const at::Tensor& unit_scores,
                                                 const at::Tensor& unit_indices,
                                                 const at::Tensor& unit_frames,
                                                 const at::Tensor& is_i_frame,
                                                 int64_t k_keep_units,
                                                 int64_t unit_area,
                                                 bool static_fallback,
                                                 double static_abs_thresh,
                                                 double static_rel_thresh,
                                                 int64_t static_uniform_frames) {
  auto long_opts = torch::TensorOptions().dtype(at::kLong).device(torch::kCPU);
  auto float_opts = torch::TensorOptions().dtype(at::kFloat).device(torch::kCPU);

  const int64_t unit_count = unit_indices.numel();
  if (unit_count <= 0 || k_keep_units <= 0) {
    return torch::empty({0}, long_opts);
  }

  const int64_t k_global = std::max<int64_t>(0, std::min<int64_t>(k_keep_units, unit_count));
  if (k_global <= 0) {
    return torch::empty({0}, long_opts);
  }

  auto unit_is_i = is_i_frame.index_select(0, unit_frames);
  auto i_indices = unit_indices.masked_select(unit_is_i);
  auto p_indices = unit_indices.masked_select(unit_is_i.logical_not());

  const int64_t i_count = i_indices.numel();
  const int64_t p_count = p_indices.numel();

  if (i_count == 0) {
    auto global_topk = std::get<1>(at::topk(unit_scores, k_global, 0, true, false)).toType(at::kLong);
    return std::get<0>(at::sort(global_topk, 0, false));
  }

  if (i_count >= k_global) {
    auto i_scores = unit_scores.index_select(0, i_indices);
    auto i_topk_rel = std::get<1>(at::topk(i_scores, k_global, 0, true, false)).toType(at::kLong);
    auto i_selected = i_indices.index_select(0, i_topk_rel);
    return std::get<0>(at::sort(i_selected, 0, false));
  }

  const int64_t remain = k_global - i_count;
  if (p_count == 0 || remain <= 0) {
    return std::get<0>(at::sort(i_indices, 0, false));
  }

  auto p_scores = unit_scores.index_select(0, p_indices);
  auto p_scores_for_topk = p_scores;

  if (static_fallback && static_uniform_frames > 0) {
    const float abs_thresh = normalize_static_abs_thresh(static_abs_thresh);
    const float rel_thresh = static_cast<float>(std::max(0.0, static_rel_thresh));

    auto unit_mean = unit_scores / static_cast<float>(std::max<int64_t>(1, unit_area));
    const int64_t k95 = percentile_rank_index(unit_count, 95.0) + 1;
    const int64_t k50 = percentile_rank_index(unit_count, 50.0) + 1;
    auto p95 = std::get<0>(unit_mean.kthvalue(k95, 0, false));
    auto p50 = std::get<0>(unit_mean.kthvalue(k50, 0, false));

    auto eps = torch::full({}, 1e-6f, float_opts);
    auto t_abs = torch::full({}, abs_thresh, float_opts);
    auto t_rel = torch::full({}, rel_thresh, float_opts);
    auto rel_contrast = (p95 - p50) / at::maximum(p95, eps);
    auto static_cond = (p95 < t_abs) | ((p95 < (t_abs * 2.0f)) & (rel_contrast < t_rel));
    auto static_scale = static_cond.toType(at::kFloat);

    const int64_t t = is_i_frame.numel();
    auto p_frame_ids = torch::arange(t, long_opts).masked_select(is_i_frame.logical_not());
    const int64_t p_frame_count = p_frame_ids.numel();

    if (p_frame_count > 0) {
      const int64_t f_u = std::max<int64_t>(1, std::min<int64_t>(static_uniform_frames, p_frame_count));
      auto uniform_pos = (torch::arange(f_u, long_opts) * p_frame_count) / f_u;
      auto uniform_frames = p_frame_ids.index_select(0, uniform_pos);

      auto p_unit_frames = unit_frames.index_select(0, p_indices);
      auto uniform_hit = p_unit_frames.unsqueeze(1).eq(uniform_frames.unsqueeze(0)).any(1);

      auto bonus = (p_scores.max() - p_scores.min() + 1.0f) * static_scale;
      p_scores_for_topk = p_scores + uniform_hit.toType(at::kFloat) * bonus;
    }
  }

  auto p_topk_rel = std::get<1>(at::topk(p_scores_for_topk, remain, 0, true, false)).toType(at::kLong);
  auto p_selected = p_indices.index_select(0, p_topk_rel);

  auto merged = at::cat({i_indices, p_selected}, 0);
  return std::get<0>(at::sort(merged, 0, false));
}

}  // namespace

SelectionResult compute_visible_indices_cpu(const at::Tensor& energy,
                                            int64_t patch_size,
                                            int64_t k_keep,
                                            int64_t selection_unit,
                                            bool static_fallback,
                                            double static_abs_thresh,
                                            double static_rel_thresh,
                                            int64_t static_uniform_frames,
                                            const std::vector<uint8_t>& is_i_positions) {
  TORCH_CHECK(!energy.is_cuda(), "energy must be on CPU");
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

  auto long_opts = torch::TensorOptions().dtype(at::kLong).device(torch::kCPU);
  if (k_global == 0) {
    auto empty = torch::empty({0}, long_opts);
    return SelectionResult{empty, scores_flat, hb, wb};
  }

  auto is_i_frame = torch::from_blob(const_cast<uint8_t*>(is_i_positions.data()),
                                     {t},
                                     torch::TensorOptions().dtype(at::kByte))
                        .clone()
                        .toType(at::kBool);

  if (selection_unit == 0) {
    auto patch_ids = torch::arange(l, long_opts);
    auto patch_frames = torch::arange(t, long_opts).repeat_interleave(patches_per_frame).contiguous();

    auto visible = select_units_with_iframe_priority_cpu(scores_flat,
                                                         patch_ids,
                                                         patch_frames,
                                                         is_i_frame,
                                                         k_global,
                                                         patch_size * patch_size,
                                                         static_fallback,
                                                         static_abs_thresh,
                                                         static_rel_thresh,
                                                         static_uniform_frames);
    return SelectionResult{visible, scores_flat, hb, wb};
  }

  TORCH_CHECK(selection_unit == 1, "selection_unit must be 0 (patch) or 1 (block2x2)");
  TORCH_CHECK(hb % 2 == 0 && wb % 2 == 0,
              "selection_unit=block2x2 requires patch-grid height/width divisible by 2");
  TORCH_CHECK(k_global % 4 == 0,
              "selection_unit=block2x2 requires k_keep divisible by 4");

  const int64_t hb_block = hb / 2;
  const int64_t wb_block = wb / 2;
  const int64_t blocks_per_frame = hb_block * wb_block;
  const int64_t block_count = t * blocks_per_frame;
  const int64_t k_blocks = k_global / 4;

  auto block_scores = scores_flat.view({t, hb, wb})
                          .view({t, hb_block, 2, wb_block, 2})
                          .sum({2, 4})
                          .reshape({block_count})
                          .contiguous();

  auto block_ids = torch::arange(block_count, long_opts);
  auto block_frames = torch::arange(t, long_opts).repeat_interleave(blocks_per_frame).contiguous();

  auto selected_blocks = select_units_with_iframe_priority_cpu(block_scores,
                                                               block_ids,
                                                               block_frames,
                                                               is_i_frame,
                                                               k_blocks,
                                                               patch_size * patch_size * 4,
                                                               static_fallback,
                                                               static_abs_thresh,
                                                               static_rel_thresh,
                                                               static_uniform_frames);

  if (selected_blocks.numel() == 0) {
    auto empty = torch::empty({0}, long_opts);
    return SelectionResult{empty, scores_flat, hb, wb};
  }

  auto frame_idx = at::floor_divide(selected_blocks, blocks_per_frame);
  auto rem = selected_blocks.remainder(blocks_per_frame);
  auto block_h = at::floor_divide(rem, wb_block);
  auto block_w = rem.remainder(wb_block);

  auto base = frame_idx * patches_per_frame + (block_h * 2) * wb + (block_w * 2);
  auto visible = torch::stack({base, base + 1, base + wb, base + wb + 1}, 1)
                     .reshape({-1})
                     .contiguous();

  return SelectionResult{visible, scores_flat, hb, wb};
}

}  // namespace codec_patch_stream
