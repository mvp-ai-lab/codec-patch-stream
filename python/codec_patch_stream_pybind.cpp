#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <utility>

namespace py = pybind11;

namespace {

std::string normalize_backend(std::string backend, const char* field_name = "backend") {
  std::transform(backend.begin(), backend.end(), backend.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (backend != "auto" && backend != "gpu" && backend != "cpu") {
    throw std::invalid_argument(std::string(field_name) + " must be one of: auto, gpu, cpu");
  }
  return backend;
}

py::object import_optional(const char* module_name) {
  try {
    return py::module_::import(module_name);
  } catch (const py::error_already_set&) {
    return py::none();
  }
}

py::object import_gpu_module() {
  return import_optional("codec_patch_stream._codec_patch_stream_gpu");
}

py::object import_cpu_module() {
  return import_optional("codec_patch_stream._codec_patch_stream_cpu");
}

bool torch_cuda_is_available() {
  try {
    auto torch = py::module_::import("torch");
    return torch.attr("cuda").attr("is_available")().cast<bool>();
  } catch (const std::exception&) {
    return false;
  }
}

std::string resolve_decode_backend(std::string decode_backend) {
  decode_backend = normalize_backend(std::move(decode_backend), "decode_backend");
  if (decode_backend == "auto") {
    return torch_cuda_is_available() ? "gpu" : "cpu";
  }
  return decode_backend;
}

std::string resolve_process_backend(std::string process_backend,
                                    const std::string& resolved_decode_backend) {
  process_backend = normalize_backend(std::move(process_backend), "process_backend");
  if (process_backend == "auto") {
    return resolved_decode_backend;
  }
  return process_backend;
}

void validate_device_id(const char* field_name, int64_t device_id) {
  if (device_id < 0) {
    throw std::invalid_argument(std::string(field_name) + " must be >= 0");
  }
}

py::object resolve_decode_module(const std::string& resolved_decode_backend) {
  if (resolved_decode_backend == "gpu") {
    auto gpu = import_gpu_module();
    if (!gpu.is_none()) {
      return gpu;
    }
    throw std::runtime_error("decode_backend=gpu requested but GPU backend module is unavailable");
  }

  auto cpu = import_cpu_module();
  if (!cpu.is_none()) {
    return cpu;
  }
  auto gpu = import_gpu_module();
  if (!gpu.is_none()) {
    return gpu;
  }
  throw std::runtime_error("decode_backend=cpu requested but no native backend modules are available");
}

py::object resolve_patch_stream_module(const std::string& resolved_decode_backend,
                                       const std::string& resolved_process_backend) {
  const bool need_gpu_module =
      (resolved_decode_backend == "gpu") || (resolved_process_backend == "gpu");
  if (need_gpu_module) {
    auto gpu = import_gpu_module();
    if (!gpu.is_none()) {
      return gpu;
    }
    throw std::runtime_error("patch_stream combo decode=" + resolved_decode_backend +
                             " process=" + resolved_process_backend +
                             " requires GPU backend module, but it is unavailable");
  }

  auto cpu = import_cpu_module();
  if (!cpu.is_none()) {
    return cpu;
  }
  auto gpu = import_gpu_module();
  if (!gpu.is_none()) {
    return gpu;
  }
  throw std::runtime_error("No available backend modules (gpu/cpu)");
}

}  // namespace

class CodecPatchStreamNativeDispatch {
 public:
  CodecPatchStreamNativeDispatch(const std::string& video_path,
                                 int64_t sequence_length,
                                 const std::string& decode_backend,
                                 const std::string& process_backend,
                                 int64_t decode_device_id,
                                 int64_t process_device_id,
                                 const std::string& uniform_strategy,
                                 int64_t input_size,
                                 int64_t min_pixels,
                                 int64_t max_pixels,
                                 int64_t patch_size,
                                 int64_t k_keep,
                                 const std::string& selection_unit,
                                 bool static_fallback,
                                 double static_abs_thresh,
                                 double static_rel_thresh,
                                 int64_t static_uniform_frames,
                                 double energy_pct,
                                 const std::string& output_dtype,
                                 int64_t prefetch_depth,
                                 int64_t nvdec_session_pool_size,
                                 int64_t uniform_auto_ratio,
                                 int64_t decode_threads,
                                 const std::string& decode_thread_type,
                                 int64_t reader_cache_size,
                                 int64_t nvdec_reuse_open_decoder) {
    const std::string resolved_decode_backend = resolve_decode_backend(decode_backend);
    const std::string resolved_process_backend =
        resolve_process_backend(process_backend, resolved_decode_backend);
    validate_device_id("decode_device_id", decode_device_id);
    validate_device_id("process_device_id", process_device_id);

    py::object mod =
        resolve_patch_stream_module(resolved_decode_backend, resolved_process_backend);
    impl_ = mod.attr("CodecPatchStreamNative")(video_path,
                                               sequence_length,
                                               resolved_decode_backend,
                                               resolved_process_backend,
                                               decode_device_id,
                                               process_device_id,
                                               uniform_strategy,
                                               input_size,
                                               min_pixels,
                                               max_pixels,
                                               patch_size,
                                               k_keep,
                                               selection_unit,
                                               static_fallback,
                                               static_abs_thresh,
                                               static_rel_thresh,
                                               static_uniform_frames,
                                               energy_pct,
                                               output_dtype,
                                               prefetch_depth,
                                               nvdec_session_pool_size,
                                               uniform_auto_ratio,
                                               decode_threads,
                                               decode_thread_type,
                                               reader_cache_size,
                                               nvdec_reuse_open_decoder);
  }

  int64_t size() const {
    return impl_.attr("__len__")().cast<int64_t>();
  }

  void reset() {
    impl_.attr("reset")();
  }

  void close() {
    impl_.attr("close")();
  }

  py::object patches() const {
    return impl_.attr("patches");
  }

  py::object metadata() const {
    return impl_.attr("metadata");
  }

  py::object metadata_tensors() const {
    return impl_.attr("metadata_tensors");
  }

  py::object sampled_frame_ids() const {
    return impl_.attr("sampled_frame_ids");
  }

  double fps() const {
    return impl_.attr("fps").cast<double>();
  }

  double duration_sec() const {
    return impl_.attr("duration_sec").cast<double>();
  }

  py::tuple next_n(int64_t n) {
    return impl_.attr("next_n")(n).cast<py::tuple>();
  }

  py::tuple next_n_tensors(int64_t n) {
    return impl_.attr("next_n_tensors")(n).cast<py::tuple>();
  }

  py::tuple next() {
    return impl_.attr("__next__")().cast<py::tuple>();
  }

 private:
  py::object impl_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "codec_patch_stream unified pybind entry";

  m.def("has_backend", [](const std::string& backend) {
    const std::string key = normalize_backend(backend);
    if (key == "gpu") {
      return !import_gpu_module().is_none();
    }
    if (key == "cpu") {
      return !import_cpu_module().is_none() || !import_gpu_module().is_none();
    }
    return !import_gpu_module().is_none() || !import_cpu_module().is_none();
  });

  m.def("version", []() {
    try {
      auto gpu = import_gpu_module();
      if (!gpu.is_none()) {
        return gpu.attr("version")().cast<std::string>();
      }
      auto cpu = import_cpu_module();
      if (!cpu.is_none()) {
        return cpu.attr("version")().cast<std::string>();
      }
      return std::string("unavailable");
    } catch (const std::exception&) {
      return std::string("unavailable");
    }
  });

  m.def("decode_only_native",
        [](const std::string& video_path,
           int64_t sequence_length,
           const std::string& decode_backend,
           int64_t decode_device_id,
           const std::string& uniform_strategy,
           int64_t nvdec_session_pool_size,
           int64_t uniform_auto_ratio,
           int64_t decode_threads,
           const std::string& decode_thread_type,
           int64_t reader_cache_size,
           int64_t nvdec_reuse_open_decoder) {
          const std::string resolved_decode_backend = resolve_decode_backend(decode_backend);
          validate_device_id("decode_device_id", decode_device_id);
          py::object mod = resolve_decode_module(resolved_decode_backend);
          if (py::hasattr(mod, "decode_only_native")) {
            return mod.attr("decode_only_native")(video_path,
                                                  sequence_length,
                                                  decode_device_id,
                                                  uniform_strategy,
                                                  nvdec_session_pool_size,
                                                  uniform_auto_ratio,
                                                  decode_threads,
                                                  decode_thread_type,
                                                  reader_cache_size,
                                                  nvdec_reuse_open_decoder)
                .cast<py::dict>();
          }
          if (py::hasattr(mod, "decode_uniform_frames")) {
            return mod.attr("decode_uniform_frames")(video_path,
                                                     sequence_length,
                                                     decode_device_id,
                                                     "throughput")
                .cast<py::dict>();
          }
          throw std::runtime_error("backend module does not expose decode functions");
        },
        py::arg("video_path"),
        py::arg("sequence_length") = 16,
        py::arg("decode_backend") = "auto",
        py::arg("decode_device_id") = 0,
        py::arg("uniform_strategy") = "auto",
        py::arg("nvdec_session_pool_size") = -1,
        py::arg("uniform_auto_ratio") = -1,
        py::arg("decode_threads") = -1,
        py::arg("decode_thread_type") = "",
        py::arg("reader_cache_size") = -1,
        py::arg("nvdec_reuse_open_decoder") = -1);

  py::class_<CodecPatchStreamNativeDispatch>(m, "CodecPatchStreamNative")
      .def(py::init<const std::string&,
                    int64_t,
                    const std::string&,
                    const std::string&,
                    int64_t,
                    int64_t,
                    const std::string&,
                    int64_t,
                    int64_t,
                    int64_t,
                    int64_t,
                    int64_t,
                    const std::string&,
                    bool,
                    double,
                    double,
                    int64_t,
                    double,
                    const std::string&,
                    int64_t,
                    int64_t,
                    int64_t,
                    int64_t,
                    const std::string&,
                    int64_t,
                    int64_t>(),
           py::arg("video_path"),
           py::arg("sequence_length") = 16,
           py::arg("decode_backend") = "auto",
           py::arg("process_backend") = "auto",
           py::arg("decode_device_id") = 0,
           py::arg("process_device_id") = 0,
           py::arg("uniform_strategy") = "auto",
           py::arg("input_size") = 224,
           py::arg("min_pixels") = -1,
           py::arg("max_pixels") = -1,
           py::arg("patch_size") = 14,
           py::arg("k_keep") = 2048,
           py::arg("selection_unit") = "patch",
           py::arg("static_fallback") = false,
           py::arg("static_abs_thresh") = 2.0,
           py::arg("static_rel_thresh") = 0.15,
           py::arg("static_uniform_frames") = 4,
           py::arg("energy_pct") = 95.0,
           py::arg("output_dtype") = "bfloat16",
           py::arg("prefetch_depth") = 3,
           py::arg("nvdec_session_pool_size") = -1,
           py::arg("uniform_auto_ratio") = -1,
           py::arg("decode_threads") = -1,
           py::arg("decode_thread_type") = "",
           py::arg("reader_cache_size") = -1,
           py::arg("nvdec_reuse_open_decoder") = -1)
      .def("__len__", &CodecPatchStreamNativeDispatch::size)
      .def("reset", &CodecPatchStreamNativeDispatch::reset)
      .def("close", &CodecPatchStreamNativeDispatch::close)
      .def_property_readonly("patches", &CodecPatchStreamNativeDispatch::patches)
      .def_property_readonly("metadata", &CodecPatchStreamNativeDispatch::metadata)
      .def_property_readonly("metadata_tensors", &CodecPatchStreamNativeDispatch::metadata_tensors)
      .def_property_readonly("sampled_frame_ids", &CodecPatchStreamNativeDispatch::sampled_frame_ids)
      .def_property_readonly("fps", &CodecPatchStreamNativeDispatch::fps)
      .def_property_readonly("duration_sec", &CodecPatchStreamNativeDispatch::duration_sec)
      .def("next_n", &CodecPatchStreamNativeDispatch::next_n)
      .def("next_n_tensors", &CodecPatchStreamNativeDispatch::next_n_tensors)
      .def("__iter__",
           [](CodecPatchStreamNativeDispatch& self) -> CodecPatchStreamNativeDispatch& {
             return self;
           },
           py::return_value_policy::reference_internal)
      .def("__next__",
           [](CodecPatchStreamNativeDispatch& self) {
             try {
               return self.next();
             } catch (const py::stop_iteration&) {
               throw;
             }
           });
}
