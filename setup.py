from __future__ import annotations

import os
from pathlib import Path

from setuptools import find_packages, setup


def _get_extensions():
    build_native = os.getenv("CODEC_BUILD_NATIVE", "0") == "1"
    if not build_native:
        return []

    try:
        import torch
        from torch.utils.cpp_extension import CUDAExtension
    except Exception as exc:
        raise RuntimeError(
            "CODEC_BUILD_NATIVE=1 but torch with C++ extension tooling is unavailable"
        ) from exc

    root = Path(__file__).resolve().parent
    src_dir = root / "cpp" / "codec_patch_stream"
    sources = [
        "python/codec_patch_stream_pybind.cpp",
        "cpp/codec_patch_stream/src/demux_decode_nvdec.cpp",
        "cpp/codec_patch_stream/src/nv12_to_rgb_kernels.cu",
        "cpp/codec_patch_stream/src/patch_select_kernels.cu",
        "cpp/codec_patch_stream/src/patch_extract_kernels.cu",
        "cpp/codec_patch_stream/src/stream_engine.cpp",
    ]

    include_dirs = [
        str((src_dir / "include").resolve()),
        "/usr/include/x86_64-linux-gnu",
        "/usr/local/include",
    ]

    return [
        CUDAExtension(
            name="codec_patch_stream._codec_patch_stream_native",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo", "-std=c++17"],
            },
            extra_link_args=[
                "-lavcodec",
                "-lavformat",
                "-lavutil",
                "-lswscale",
                "-lswresample",
            ],
        )
    ]


cmdclass = {}
ext_modules = _get_extensions()
if ext_modules:
    from torch.utils.cpp_extension import BuildExtension

    cmdclass = {"build_ext": BuildExtension}

setup(
    name="codec-patch-stream",
    version="0.2.0",
    description="GPU-accelerated video patch stream loader",
    packages=find_packages(include=["codec_patch_stream", "codec_patch_stream.*"]),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.10",
)
