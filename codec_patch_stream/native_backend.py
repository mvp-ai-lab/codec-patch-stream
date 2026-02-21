from __future__ import annotations

import importlib
from typing import Any

_NATIVE: Any | None = None
_NATIVE_IMPORT_ERROR: Exception | None = None


def _import_optional(module_name: str) -> tuple[Any | None, Exception | None]:
    try:
        return importlib.import_module(f".{module_name}", __package__), None
    except Exception as exc:  # pragma: no cover - optional extension import
        return None, exc


def _load_dispatch_native() -> Any | None:
    global _NATIVE, _NATIVE_IMPORT_ERROR
    if _NATIVE is not None:
        return _NATIVE
    if _NATIVE_IMPORT_ERROR is not None:
        return None

    mod, err = _import_optional("_codec_patch_stream_native")
    if mod is not None:
        _NATIVE = mod
        return _NATIVE

    _NATIVE_IMPORT_ERROR = err
    return None


def _normalize_backend(backend: str) -> str:
    key = str(backend).strip().lower()
    if key not in {"auto", "gpu", "cpu"}:
        raise ValueError("backend must be one of: auto, gpu, cpu")
    return key


def has_native_backend(backend: str = "auto") -> bool:
    key = _normalize_backend(backend)
    mod = _load_dispatch_native()
    if mod is None:
        return False
    if not hasattr(mod, "has_backend"):
        return key == "auto"
    return bool(mod.has_backend(key))


def load_native_backend(backend: str = "auto"):
    key = _normalize_backend(backend)
    mod = _load_dispatch_native()
    if mod is None:
        detail = (
            "unavailable" if _NATIVE_IMPORT_ERROR is None else str(_NATIVE_IMPORT_ERROR)
        )
        raise RuntimeError(
            "codec_patch_stream native backend is required but unavailable. "
            f"Build with CODEC_BUILD_NATIVE=1. (dispatch import failed: {detail})"
        )

    if hasattr(mod, "has_backend") and not bool(mod.has_backend(key)):
        raise RuntimeError(f"codec_patch_stream backend '{key}' is unavailable")

    return mod


def require_native_backend(backend: str = "auto"):
    return load_native_backend(backend)


def native_version(backend: str = "auto") -> str:
    mod = _load_dispatch_native()
    if mod is None:
        return "unavailable"
    try:
        return str(mod.version())
    except Exception:
        return "unavailable"
