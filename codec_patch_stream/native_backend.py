from __future__ import annotations

import importlib
from typing import Any

_GPU_NATIVE: Any | None = None
_CPU_NATIVE: Any | None = None
_GPU_IMPORT_ERROR: Exception | None = None
_CPU_IMPORT_ERROR: Exception | None = None


def _import_optional(module_name: str) -> tuple[Any | None, Exception | None]:
    try:
        return importlib.import_module(f".{module_name}", __package__), None
    except Exception as exc:  # pragma: no cover - optional extension import
        return None, exc


def _load_gpu_native() -> Any | None:
    global _GPU_NATIVE, _GPU_IMPORT_ERROR
    if _GPU_NATIVE is not None:
        return _GPU_NATIVE
    if _GPU_IMPORT_ERROR is not None:
        return None

    mod, err = _import_optional("_codec_patch_stream_gpu")
    if mod is not None:
        _GPU_NATIVE = mod
        return _GPU_NATIVE

    legacy_mod, legacy_err = _import_optional("_codec_patch_stream_native")
    if legacy_mod is not None:
        _GPU_NATIVE = legacy_mod
        return _GPU_NATIVE

    if err is not None and legacy_err is not None:
        _GPU_IMPORT_ERROR = RuntimeError(f"{err}; legacy module error: {legacy_err}")
    else:
        _GPU_IMPORT_ERROR = err or legacy_err
    return None


def _load_cpu_native() -> Any | None:
    global _CPU_NATIVE, _CPU_IMPORT_ERROR
    if _CPU_NATIVE is not None:
        return _CPU_NATIVE
    if _CPU_IMPORT_ERROR is not None:
        return None

    mod, err = _import_optional("_codec_patch_stream_cpu")
    if mod is not None:
        _CPU_NATIVE = mod
        return _CPU_NATIVE
    _CPU_IMPORT_ERROR = err
    return None


def _fmt_err(prefix: str, err: Exception | None) -> str:
    if err is None:
        return f"{prefix}: unavailable"
    return f"{prefix}: {err}"


def has_native_backend(backend: str = "auto") -> bool:
    key = str(backend).strip().lower()
    if key == "gpu":
        return _load_gpu_native() is not None
    if key == "cpu":
        return _load_cpu_native() is not None
    if key == "auto":
        if _load_gpu_native() is not None:
            return True
        return _load_cpu_native() is not None
    return False


def load_native_backend(backend: str = "auto"):
    key = str(backend).strip().lower()
    if key == "gpu":
        mod = _load_gpu_native()
        if mod is None:
            detail = _fmt_err("GPU backend import failed", _GPU_IMPORT_ERROR)
            raise RuntimeError(
                f"codec_patch_stream GPU backend is required but unavailable ({detail})."
            )
        return mod

    if key == "cpu":
        mod = _load_cpu_native()
        if mod is None:
            detail = _fmt_err("CPU backend import failed", _CPU_IMPORT_ERROR)
            raise RuntimeError(
                f"codec_patch_stream CPU backend is required but unavailable ({detail})."
            )
        return mod

    if key == "auto":
        mod = _load_gpu_native()
        if mod is not None:
            return mod
        mod = _load_cpu_native()
        if mod is not None:
            return mod
        detail = "; ".join(
            [
                _fmt_err("GPU", _GPU_IMPORT_ERROR),
                _fmt_err("CPU", _CPU_IMPORT_ERROR),
            ]
        )
        raise RuntimeError(
            "codec_patch_stream native backend is required but unavailable. "
            f"Build with CODEC_BUILD_NATIVE=1. ({detail})"
        )

    raise ValueError("backend must be one of: auto, gpu, cpu")


def require_native_backend(backend: str = "auto"):
    return load_native_backend(backend)


def native_version(backend: str = "auto") -> str:
    key = str(backend).strip().lower()
    if key == "gpu":
        mod = _load_gpu_native()
        return "unavailable" if mod is None else str(mod.version())
    if key == "cpu":
        mod = _load_cpu_native()
        return "unavailable" if mod is None else str(mod.version())

    if _GPU_NATIVE is not None:
        return f"gpu:{_GPU_NATIVE.version()}"
    if _CPU_NATIVE is not None:
        return f"cpu:{_CPU_NATIVE.version()}"

    mod = _load_gpu_native()
    if mod is not None:
        return f"gpu:{mod.version()}"
    mod = _load_cpu_native()
    if mod is not None:
        return f"cpu:{mod.version()}"
    return "unavailable"
