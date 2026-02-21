cmake_minimum_required(VERSION 3.20)

find_package(Torch REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale libswresample)

set(CPU_SOURCES
    ../../python/codec_patch_stream_pybind_cpu.cpp
    core/decode_postprocess.cpp
    core/decode_core_cpu.cpp
    backends/cpu/decode_executor_ffmpeg_cpu.cpp
    patch/energy_cpu.cpp
    patch/select_cpu.cpp
    patch/extract_cpu.cpp
    patch/patch_stream_engine_cpu.cpp
)

set(DISPATCH_SOURCES
    ../../python/codec_patch_stream_pybind.cpp
)

add_library(_codec_patch_stream_native MODULE ${DISPATCH_SOURCES})
target_include_directories(_codec_patch_stream_native PRIVATE include core ${FFMPEG_INCLUDE_DIRS})
target_link_libraries(_codec_patch_stream_native PRIVATE ${TORCH_LIBRARIES})
target_compile_features(_codec_patch_stream_native PRIVATE cxx_std_17)
set_target_properties(_codec_patch_stream_native PROPERTIES PREFIX "")

add_library(_codec_patch_stream_cpu MODULE ${CPU_SOURCES})
target_include_directories(_codec_patch_stream_cpu PRIVATE include core ${FFMPEG_INCLUDE_DIRS})
target_link_libraries(_codec_patch_stream_cpu PRIVATE ${TORCH_LIBRARIES} ${FFMPEG_LIBRARIES})
target_compile_features(_codec_patch_stream_cpu PRIVATE cxx_std_17)
set_target_properties(_codec_patch_stream_cpu PROPERTIES PREFIX "")

if(CODEC_ENABLE_GPU)
  set(GPU_SOURCES
      ../../python/codec_patch_stream_pybind_gpu.cpp
      core/decode_postprocess.cpp
      core/decode_core_gpu.cpp
      backends/gpu/decode_executor_nvdec.cpp
      backends/gpu/nv12_to_rgb_kernels.cu
      patch/energy_gpu.cu
      patch/select_gpu.cu
      patch/extract_gpu.cu
      patch/patch_stream_engine_gpu.cpp
  )

  add_library(_codec_patch_stream_gpu MODULE ${GPU_SOURCES})
  target_include_directories(_codec_patch_stream_gpu PRIVATE include core ${FFMPEG_INCLUDE_DIRS})
  target_link_libraries(_codec_patch_stream_gpu PRIVATE ${TORCH_LIBRARIES} ${FFMPEG_LIBRARIES})
  target_compile_features(_codec_patch_stream_gpu PRIVATE cxx_std_17)
  set_target_properties(_codec_patch_stream_gpu PROPERTIES PREFIX "")
endif()
