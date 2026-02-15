from __future__ import annotations

import os
import shutil
from pathlib import Path

from setuptools import find_packages, setup


def _get_extensions():
    build_native = os.getenv("CODEC_BUILD_NATIVE", "0") == "1"
    if not build_native:
        return []

    gpu_mode = os.getenv("CODEC_ENABLE_GPU", "auto").strip().lower()
    if gpu_mode not in {"auto", "0", "1", "false", "true", "off", "on"}:
        raise ValueError("CODEC_ENABLE_GPU must be one of: auto, 0, 1, false, true, off, on")

    try:
        from torch.utils.cpp_extension import CppExtension, CUDAExtension
    except Exception as exc:
        raise RuntimeError(
            "CODEC_BUILD_NATIVE=1 but torch with C++ extension tooling is unavailable"
        ) from exc

    root = Path(__file__).resolve().parent
    src_dir = root / "cpp" / "codec_patch_stream"
    include_dirs = [
        str((src_dir / "include").resolve()),
        "/usr/include/x86_64-linux-gnu",
        "/usr/local/include",
    ]

    common_link_args = [
        "-lavcodec",
        "-lavformat",
        "-lavutil",
        "-lswscale",
        "-lswresample",
    ]
    cxx_args = ["-O3", "-std=c++17"]
    ext_modules = [
        CppExtension(
            name="codec_patch_stream._codec_patch_stream_cpu",
            sources=[
                "python/codec_patch_stream_pybind_cpu.cpp",
                "cpp/codec_patch_stream/src/demux_decode_ffmpeg_cpu.cpp",
                "cpp/codec_patch_stream/src/motion_residual_proxy_cpu.cpp",
                "cpp/codec_patch_stream/src/patch_select_cpu.cpp",
                "cpp/codec_patch_stream/src/patch_extract_cpu.cpp",
                "cpp/codec_patch_stream/src/stream_engine_cpu.cpp",
            ],
            include_dirs=include_dirs,
            extra_compile_args=cxx_args,
            extra_link_args=common_link_args,
        )
    ]

    nvcc_path = shutil.which("nvcc")
    should_build_gpu = (
        gpu_mode in {"1", "true", "on"}
        or (gpu_mode == "auto" and nvcc_path is not None)
    )

    if gpu_mode in {"1", "true", "on"} and nvcc_path is None:
        raise RuntimeError("CODEC_ENABLE_GPU=1 but nvcc was not found in PATH")

    if should_build_gpu:
        ext_modules.append(
            CUDAExtension(
                name="codec_patch_stream._codec_patch_stream_gpu",
                sources=[
                    "python/codec_patch_stream_pybind_gpu.cpp",
                    "cpp/codec_patch_stream/src/demux_decode_nvdec.cpp",
                    "cpp/codec_patch_stream/src/nv12_to_rgb_kernels.cu",
                    "cpp/codec_patch_stream/src/motion_residual_proxy_kernels.cu",
                    "cpp/codec_patch_stream/src/patch_select_kernels.cu",
                    "cpp/codec_patch_stream/src/patch_extract_kernels.cu",
                    "cpp/codec_patch_stream/src/stream_engine.cpp",
                ],
                include_dirs=include_dirs,
                extra_compile_args={
                    "cxx": cxx_args,
                    "nvcc": ["-O3", "--use_fast_math", "-lineinfo", "-std=c++17"],
                },
                extra_link_args=common_link_args,
            )
        )

    return ext_modules


cmdclass = {}
ext_modules = _get_extensions()
if ext_modules:
    from torch.utils.cpp_extension import BuildExtension

    cmdclass = {"build_ext": BuildExtension}

setup(
    name="codec-patch-stream",
    version="0.3.0",
    description="Video patch stream loader with CPU/GPU native backends",
    packages=find_packages(include=["codec_patch_stream", "codec_patch_stream.*"]),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.10",
)
