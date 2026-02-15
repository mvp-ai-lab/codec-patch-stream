cmake_minimum_required(VERSION 3.20)

find_package(Torch REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale libswresample)

set(SOURCES
    ../../python/codec_patch_stream_pybind.cpp
    src/demux_decode_nvdec.cpp
    src/nv12_to_rgb_kernels.cu
    src/motion_residual_proxy_kernels.cu
    src/patch_select_kernels.cu
    src/patch_extract_kernels.cu
    src/stream_engine.cpp
)

add_library(_codec_patch_stream_native MODULE ${SOURCES})
target_include_directories(_codec_patch_stream_native PRIVATE include ${FFMPEG_INCLUDE_DIRS})
target_link_libraries(_codec_patch_stream_native PRIVATE ${TORCH_LIBRARIES} ${FFMPEG_LIBRARIES})
target_compile_features(_codec_patch_stream_native PRIVATE cxx_std_17)
set_target_properties(_codec_patch_stream_native PROPERTIES PREFIX "")
