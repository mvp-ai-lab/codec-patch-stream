from __future__ import annotations

_NATIVE_IMPORT_ERROR: Exception | None = None

try:
    from . import _codec_patch_stream_native as native
except Exception as exc:  # pragma: no cover - optional extension import
    native = None
    _NATIVE_IMPORT_ERROR = exc


def has_native_backend() -> bool:
    return native is not None


def require_native_backend():
    if native is None:
        detail = f": {_NATIVE_IMPORT_ERROR}" if _NATIVE_IMPORT_ERROR is not None else ""
        raise RuntimeError(
            "codec_patch_stream native backend is required but unavailable"
            f"{detail}. Build with CODEC_BUILD_NATIVE=1."
        )
    return native


def native_version() -> str:
    if native is None:
        return "unavailable"
    return str(native.version())
