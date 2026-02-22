from __future__ import annotations

import os
import subprocess
import shutil
from pathlib import Path

from setuptools import find_packages, setup


def _detect_torch_cuda_arch_list() -> str | None:
    """Detect local GPU compute capabilities and convert to TORCH_CUDA_ARCH_LIST format."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None

    caps: list[str] = []
    for raw in out.splitlines():
        cap = raw.strip()
        if not cap:
            continue
        parts = cap.split(".", 1)
        if len(parts) != 2:
            continue
        major, minor = parts[0].strip(), parts[1].strip()
        if not (major.isdigit() and minor.isdigit()):
            continue
        normalized = f"{int(major)}.{int(minor)}"
        if normalized not in caps:
            caps.append(normalized)
    if not caps:
        return None
    return ";".join(caps)


def _maybe_configure_cuda_arch_list() -> str | None:
    # Priority: user explicit CODEC_CUDA_ARCH_LIST > existing TORCH_CUDA_ARCH_LIST > auto detect.
    codec_arch = os.getenv("CODEC_CUDA_ARCH_LIST", "").strip()
    if codec_arch:
        os.environ["TORCH_CUDA_ARCH_LIST"] = codec_arch
        return codec_arch

    torch_arch = os.getenv("TORCH_CUDA_ARCH_LIST", "").strip()
    if torch_arch:
        return torch_arch

    detected = _detect_torch_cuda_arch_list()
    if detected:
        os.environ["TORCH_CUDA_ARCH_LIST"] = detected
        return detected
    return None


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
        str((src_dir / "core").resolve()),
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
            name="codec_patch_stream._codec_patch_stream_native",
            sources=[
                "python/codec_patch_stream_pybind.cpp",
            ],
            include_dirs=include_dirs,
            extra_compile_args=cxx_args,
            extra_link_args=[],
        ),
        CppExtension(
            name="codec_patch_stream._codec_patch_stream_cpu",
            sources=[
                "python/codec_patch_stream_pybind_cpu.cpp",
                "cpp/codec_patch_stream/core/decode_postprocess.cpp",
                "cpp/codec_patch_stream/core/decode_core_cpu.cpp",
                "cpp/codec_patch_stream/backends/cpu/decode_executor_ffmpeg_cpu.cpp",
                "cpp/codec_patch_stream/patch/energy_cpu.cpp",
                "cpp/codec_patch_stream/patch/select_cpu.cpp",
                "cpp/codec_patch_stream/patch/extract_cpu.cpp",
                "cpp/codec_patch_stream/patch/patch_stream_engine_cpu.cpp",
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
        _maybe_configure_cuda_arch_list()
        nvcc_args = ["-O3", "--use_fast_math", "-std=c++17"]
        if os.getenv("CODEC_NVCC_LINEINFO", "0").strip() == "1":
            nvcc_args.append("-lineinfo")
        # decode_executor_nvdec.cpp uses CUDA Driver API (e.g. cuCtxGetCurrent),
        # which requires explicit linkage to libcuda.
        gpu_link_args = common_link_args + ["-lcuda"]
        ext_modules.append(
            CUDAExtension(
                name="codec_patch_stream._codec_patch_stream_gpu",
                sources=[
                    "python/codec_patch_stream_pybind_gpu.cpp",
                    "cpp/codec_patch_stream/core/decode_postprocess.cpp",
                    "cpp/codec_patch_stream/core/decode_core_gpu.cpp",
                    "cpp/codec_patch_stream/core/decode_core_cpu.cpp",
                    "cpp/codec_patch_stream/backends/gpu/decode_executor_nvdec.cpp",
                    "cpp/codec_patch_stream/backends/cpu/decode_executor_ffmpeg_cpu.cpp",
                    "cpp/codec_patch_stream/backends/gpu/nv12_to_rgb_kernels.cu",
                    "cpp/codec_patch_stream/patch/energy_cpu.cpp",
                    "cpp/codec_patch_stream/patch/energy_gpu.cu",
                    "cpp/codec_patch_stream/patch/select_cpu.cpp",
                    "cpp/codec_patch_stream/patch/select_gpu.cu",
                    "cpp/codec_patch_stream/patch/extract_cpu.cpp",
                    "cpp/codec_patch_stream/patch/extract_gpu.cu",
                    "cpp/codec_patch_stream/patch/patch_stream_engine_gpu.cpp",
                ],
                include_dirs=include_dirs,
                extra_compile_args={
                    "cxx": cxx_args,
                    "nvcc": nvcc_args,
                },
                extra_link_args=gpu_link_args,
            )
        )

    return ext_modules


cmdclass = {}
ext_modules = _get_extensions()
if ext_modules:
    from torch.utils.cpp_extension import BuildExtension

    cmdclass = {"build_ext": BuildExtension.with_options(use_ninja=True)}

setup(
    name="codec-patch-stream",
    version="0.4.0",
    description="Video patch stream loader with CPU/GPU native backends",
    packages=find_packages(include=["codec_patch_stream", "codec_patch_stream.*"]),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.10",
)
