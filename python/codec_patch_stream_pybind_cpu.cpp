#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <utility>

#include "codec_patch_stream_cpu.h"

namespace py = pybind11;

namespace {

at::ScalarType parse_dtype(const std::string& dtype) {
  std::string key = dtype;
  std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  if (key == "bfloat16" || key == "bf16") {
    return at::kBFloat16;
  }
  if (key == "float16" || key == "fp16") {
    return at::kHalf;
  }
  if (key == "float32" || key == "fp32") {
    return at::kFloat;
  }
  throw std::invalid_argument("Unsupported output_dtype: " + dtype);
}

py::dict meta_to_dict(const codec_patch_stream::PatchMeta& m) {
  py::dict d;
  d["seq_pos"] = m.seq_pos;
  d["frame_id"] = m.frame_id;
  d["is_i"] = m.is_i;
  d["patch_linear_idx"] = m.patch_linear_idx;
  d["patch_h_idx"] = m.patch_h_idx;
  d["patch_w_idx"] = m.patch_w_idx;
  d["score"] = m.score;
  return d;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "codec_patch_stream cpu native backend";

  m.def("version", []() { return codec_patch_stream::version(); });
  m.def("linear_to_thw",
        &codec_patch_stream::linear_to_thw,
        "Map linear patch index to (t,h,w)");

  py::class_<codec_patch_stream::CodecPatchStreamNative>(m, "CodecPatchStreamNative")
      .def(py::init([](const std::string& video_path,
                       int64_t sequence_length,
                       int64_t input_size,
                       int64_t patch_size,
                       int64_t k_keep,
                       bool static_fallback,
                       double static_abs_thresh,
                       double static_rel_thresh,
                       int64_t static_uniform_frames,
                       double energy_pct,
                       const std::string& output_dtype,
                       int64_t device_id,
                       int64_t prefetch_depth) {
             codec_patch_stream::StreamConfig cfg;
             cfg.sequence_length = sequence_length;
             cfg.input_size = input_size;
             cfg.patch_size = patch_size;
             cfg.k_keep = k_keep;
             cfg.static_fallback = static_fallback;
             cfg.static_abs_thresh = static_abs_thresh;
             cfg.static_rel_thresh = static_rel_thresh;
             cfg.static_uniform_frames = static_uniform_frames;
             cfg.energy_pct = energy_pct;
             cfg.output_dtype = parse_dtype(output_dtype);
             cfg.device_id = device_id;
             cfg.prefetch_depth = prefetch_depth;
             return std::make_unique<codec_patch_stream::CodecPatchStreamNative>(video_path,
                                                                                 cfg);
           }),
           py::arg("video_path"),
           py::arg("sequence_length") = 16,
           py::arg("input_size") = 224,
           py::arg("patch_size") = 14,
           py::arg("k_keep") = 2048,
           py::arg("static_fallback") = false,
           py::arg("static_abs_thresh") = 2.0,
           py::arg("static_rel_thresh") = 0.15,
           py::arg("static_uniform_frames") = 4,
           py::arg("energy_pct") = 95.0,
           py::arg("output_dtype") = "bfloat16",
           py::arg("device_id") = 0,
           py::arg("prefetch_depth") = 3)
      .def("__len__", &codec_patch_stream::CodecPatchStreamNative::size)
      .def("reset", &codec_patch_stream::CodecPatchStreamNative::reset)
      .def("close", &codec_patch_stream::CodecPatchStreamNative::close)
      .def_property_readonly("patches",
                             &codec_patch_stream::CodecPatchStreamNative::patch_bank)
      .def_property_readonly("metadata", [](codec_patch_stream::CodecPatchStreamNative& self) {
        py::list out;
        for (const auto& m : self.metadata()) {
          out.append(meta_to_dict(m));
        }
        return out;
      })
      .def_property_readonly("metadata_tensors",
                             [](codec_patch_stream::CodecPatchStreamNative& self) {
                               return py::make_tuple(self.metadata_fields_gpu(),
                                                     self.metadata_scores_gpu());
                             })
      .def("next_n", [](codec_patch_stream::CodecPatchStreamNative& self, int64_t n) {
        auto out = self.next_n(n);
        py::list metas;
        for (const auto& m : std::get<1>(out)) {
          metas.append(meta_to_dict(m));
        }
        return py::make_tuple(std::get<0>(out), metas);
      })
      .def("next_n_tensors", [](codec_patch_stream::CodecPatchStreamNative& self, int64_t n) {
        auto out = self.next_n_tensors(n);
        return py::make_tuple(std::get<0>(out), std::get<1>(out), std::get<2>(out));
      })
      .def("__iter__", [](codec_patch_stream::CodecPatchStreamNative& self)
                          -> codec_patch_stream::CodecPatchStreamNative& { return self; },
           py::return_value_policy::reference_internal)
      .def("__next__", [](codec_patch_stream::CodecPatchStreamNative& self) {
        try {
          auto out = self.next();
          return py::make_tuple(std::get<0>(out), meta_to_dict(std::get<1>(out)));
        } catch (const std::out_of_range&) {
          throw py::stop_iteration();
        }
      });
}
