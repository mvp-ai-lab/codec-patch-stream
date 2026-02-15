cmake_minimum_required(VERSION 3.20)

find_package(Torch REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale libswresample)

set(CPU_SOURCES
    ../../python/codec_patch_stream_pybind_cpu.cpp
    src/demux_decode_ffmpeg_cpu.cpp
    src/motion_residual_proxy_cpu.cpp
    src/patch_select_cpu.cpp
    src/patch_extract_cpu.cpp
    src/stream_engine_cpu.cpp
)

add_library(_codec_patch_stream_cpu MODULE ${CPU_SOURCES})
target_include_directories(_codec_patch_stream_cpu PRIVATE include ${FFMPEG_INCLUDE_DIRS})
target_link_libraries(_codec_patch_stream_cpu PRIVATE ${TORCH_LIBRARIES} ${FFMPEG_LIBRARIES})
target_compile_features(_codec_patch_stream_cpu PRIVATE cxx_std_17)
set_target_properties(_codec_patch_stream_cpu PROPERTIES PREFIX "")

if(CODEC_ENABLE_GPU)
  set(GPU_SOURCES
      ../../python/codec_patch_stream_pybind_gpu.cpp
      src/demux_decode_nvdec.cpp
      src/nv12_to_rgb_kernels.cu
      src/motion_residual_proxy_kernels.cu
      src/patch_select_kernels.cu
      src/patch_extract_kernels.cu
      src/stream_engine.cpp
  )

  add_library(_codec_patch_stream_gpu MODULE ${GPU_SOURCES})
  target_include_directories(_codec_patch_stream_gpu PRIVATE include ${FFMPEG_INCLUDE_DIRS})
  target_link_libraries(_codec_patch_stream_gpu PRIVATE ${TORCH_LIBRARIES} ${FFMPEG_LIBRARIES})
  target_compile_features(_codec_patch_stream_gpu PRIVATE cxx_std_17)
  set_target_properties(_codec_patch_stream_gpu PROPERTIES PREFIX "")
endif()
