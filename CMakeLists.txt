cmake_minimum_required(VERSION 3.20)
project(codec_patch_stream LANGUAGES CXX)

option(CODEC_ENABLE_GPU "Build GPU backend module" ON)
if(CODEC_ENABLE_GPU)
  enable_language(CUDA)
endif()

add_subdirectory(cpp/codec_patch_stream)
