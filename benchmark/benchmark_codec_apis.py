from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from statistics import fmean, stdev
from time import perf_counter

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from codec_patch_stream import DecodeConfig, PatchStreamConfig, decode_only, patch_stream

VIDEO_EXTENSIONS = {
    ".avi",
    ".flv",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ts",
    ".webm",
}


def _sync_gpu_if_needed(backend: str, device_id: int) -> None:
    if backend != "gpu" or not torch.cuda.is_available():
        return
    torch.cuda.synchronize(device=device_id)


def _resolve_decode_backend(backend: str) -> str:
    key = str(backend).strip().lower()
    if key not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"invalid decode backend: {backend}")
    if key == "auto":
        return "gpu" if torch.cuda.is_available() else "cpu"
    return key


def _resolve_process_backend(process_backend: str, resolved_decode_backend: str) -> str:
    key = str(process_backend).strip().lower()
    if key not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"invalid process backend: {process_backend}")
    if key == "auto":
        return resolved_decode_backend
    return key


def _list_videos(video_dir: Path, limit: int | None) -> list[Path]:
    files = [p for p in sorted(video_dir.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    if limit is not None:
        return files[:limit]
    return files


def _backend_is_available(backend: str) -> bool:
    key = _resolve_decode_backend(backend)
    if key == "cpu":
        return True
    if not torch.cuda.is_available():
        return False
    try:
        from codec_patch_stream import _codec_patch_stream_native as _dispatch

        has_backend = getattr(_dispatch, "has_backend", None)
        if callable(has_backend):
            return bool(has_backend("gpu"))
    except Exception:
        return True
    return True


def _gpu_runtime_available() -> bool:
    return bool(torch.cuda.is_available())


def _decord_backend_is_available(backend: str, device_id: int) -> bool:
    key = _resolve_decode_backend(backend)
    try:
        import decord
    except Exception:
        return False

    if key == "cpu":
        return True
    if not torch.cuda.is_available():
        return False
    try:
        decord.gpu(device_id)
    except Exception:
        return False
    return True


def _uniform_indices(total_frames: int, sequence_length: int) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError("video has no frames")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be > 0")
    if total_frames == 1:
        return np.zeros((sequence_length,), dtype=np.int64)
    return np.linspace(0, total_frames - 1, sequence_length, dtype=np.int64)


def _run_decode(video_path: Path, args: argparse.Namespace, decode_backend: str) -> float:
    resolved_decode = _resolve_decode_backend(decode_backend)
    config = DecodeConfig(
        video_path=str(video_path),
        sequence_length=args.sequence_length,
        decode_backend=decode_backend,
        decode_device_id=args.decode_device_id,
        uniform_strategy=args.uniform_strategy,
    )
    _sync_gpu_if_needed(resolved_decode, args.decode_device_id)
    t0 = perf_counter()
    decoded = decode_only(config)
    _ = decoded.frames.shape
    _sync_gpu_if_needed(resolved_decode, args.decode_device_id)
    return perf_counter() - t0


def _run_patch(
    video_path: Path,
    args: argparse.Namespace,
    decode_backend: str,
    process_backend: str,
) -> float:
    resolved_decode = _resolve_decode_backend(decode_backend)
    resolved_process = _resolve_process_backend(process_backend, resolved_decode)

    config = PatchStreamConfig(
        video_path=str(video_path),
        sequence_length=args.sequence_length,
        uniform_strategy=args.uniform_strategy,
        input_size=args.input_size,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        patch_size=args.patch_size,
        k_keep=args.k_keep,
        output_dtype=args.output_dtype,
        decode_backend=decode_backend,
        process_backend=process_backend,
        decode_device_id=args.decode_device_id,
        process_device_id=args.process_device_id,
    )

    _sync_gpu_if_needed(resolved_decode, args.decode_device_id)
    if args.process_device_id != args.decode_device_id:
        _sync_gpu_if_needed(resolved_process, args.process_device_id)

    t0 = perf_counter()
    session = patch_stream(config)
    try:
        _ = session.patches.shape
        _ = len(session.metadata)
    finally:
        session.close()

    _sync_gpu_if_needed(resolved_decode, args.decode_device_id)
    if args.process_device_id != args.decode_device_id:
        _sync_gpu_if_needed(resolved_process, args.process_device_id)
    return perf_counter() - t0


def _run_decord(video_path: Path, args: argparse.Namespace, decode_backend: str) -> float:
    try:
        import decord
    except Exception as exc:
        raise RuntimeError("decord is not installed in current environment") from exc

    resolved_decode = _resolve_decode_backend(decode_backend)
    ctx = decord.cpu() if resolved_decode == "cpu" else decord.gpu(args.decode_device_id)
    _sync_gpu_if_needed(resolved_decode, args.decode_device_id)
    t0 = perf_counter()
    try:
        vr = decord.VideoReader(
            str(video_path),
            ctx=ctx,
            num_threads=0 if resolved_decode == "gpu" else int(args.decord_cpu_threads),
        )
    except Exception as exc:
        if resolved_decode == "gpu" and "CUDA not enabled" in str(exc):
            raise RuntimeError("decord GPU backend unavailable (built without CUDA)") from exc
        raise
    indices = _uniform_indices(len(vr), args.sequence_length)
    batch = vr.get_batch(indices)
    if resolved_decode == "cpu":
        _ = batch.asnumpy().shape
    else:
        _ = batch.shape
    _sync_gpu_if_needed(resolved_decode, args.decode_device_id)
    return perf_counter() - t0


def _fmt_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return stdev(values)


def _benchmark_decode_like(
    *,
    api: str,
    decode_backend: str,
    videos: list[Path],
    args: argparse.Namespace,
) -> None:
    latencies: list[float] = []
    speeds: list[float] = []
    runner = _run_decode if api == "decode_only" else _run_decord

    print(f"\n=== API={api} decode={decode_backend} ===")
    for idx, video_path in enumerate(videos, start=1):
        try:
            elapsed = runner(video_path, args, decode_backend)
        except Exception as exc:
            print(f"[{idx}/{len(videos)}] {video_path.name} | failed: {exc}")
            continue
        speed = 1.0 / elapsed if elapsed > 0 else math.inf
        latencies.append(elapsed)
        speeds.append(speed)
        print(
            f"[{idx}/{len(videos)}] {video_path.name} | "
            f"latency={elapsed:.4f}s | speed={speed:.3f} videos/s"
        )

    if not latencies:
        print("Summary: no successful runs for this API/backend combo.")
        return

    mean_latency = fmean(latencies)
    std_latency = _fmt_std(latencies)
    mean_speed = fmean(speeds)
    std_speed = _fmt_std(speeds)
    print(
        "Summary: "
        f"count={len(latencies)} | "
        f"latency_mean={mean_latency:.4f}s latency_std={std_latency:.4f}s | "
        f"speed_mean={mean_speed:.3f} videos/s speed_std={std_speed:.3f}"
    )


def _benchmark_patch_combo(
    *,
    decode_backend: str,
    process_backend: str,
    videos: list[Path],
    args: argparse.Namespace,
) -> None:
    latencies: list[float] = []
    speeds: list[float] = []

    print(f"\n=== API=patch_stream decode={decode_backend} process={process_backend} ===")
    for idx, video_path in enumerate(videos, start=1):
        try:
            elapsed = _run_patch(video_path, args, decode_backend, process_backend)
        except Exception as exc:
            print(f"[{idx}/{len(videos)}] {video_path.name} | failed: {exc}")
            continue
        speed = 1.0 / elapsed if elapsed > 0 else math.inf
        latencies.append(elapsed)
        speeds.append(speed)
        print(
            f"[{idx}/{len(videos)}] {video_path.name} | "
            f"latency={elapsed:.4f}s | speed={speed:.3f} videos/s"
        )

    if not latencies:
        print("Summary: no successful runs for this API/backend combo.")
        return

    mean_latency = fmean(latencies)
    std_latency = _fmt_std(latencies)
    mean_speed = fmean(speeds)
    std_speed = _fmt_std(speeds)
    print(
        "Summary: "
        f"count={len(latencies)} | "
        f"latency_mean={mean_latency:.4f}s latency_std={std_latency:.4f}s | "
        f"speed_mean={mean_speed:.3f} videos/s speed_std={std_speed:.3f}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark decode_only / patch_stream / decord on CPU/GPU backends."
    )
    parser.add_argument("video_dir", type=Path, help="Directory containing videos.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of videos to benchmark.")
    parser.add_argument(
        "--apis",
        nargs="+",
        default=["decode_only", "patch_stream"],
        choices=["decode_only", "patch_stream", "decord"],
        help="APIs to benchmark, executed in listed order.",
    )
    parser.add_argument(
        "--decode-backends",
        nargs="+",
        default=["cpu", "gpu"],
        choices=["auto", "cpu", "gpu"],
        help="Decode backends to benchmark, executed in listed order.",
    )
    parser.add_argument(
        "--process-backends",
        nargs="+",
        default=["cpu", "gpu"],
        choices=["auto", "cpu", "gpu"],
        help="Process backends for patch_stream, executed in listed order.",
    )
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--uniform-strategy", default="auto", choices=["auto", "seek", "stream"])
    parser.add_argument("--decode-device-id", type=int, default=0)
    parser.add_argument("--process-device-id", type=int, default=0)
    parser.add_argument(
        "--decord-cpu-threads",
        type=int,
        default=0,
        help="CPU thread count for decord when decode backend resolves to cpu, 0 means decord default.",
    )
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--k-keep", type=int, default=2048)
    parser.add_argument(
        "--output-dtype",
        default="bfloat16",
        choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.video_dir.is_dir():
        raise ValueError(f"video_dir is not a directory: {args.video_dir}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be > 0")

    videos = _list_videos(args.video_dir, args.limit)
    if not videos:
        raise ValueError(f"No videos found in {args.video_dir}")

    print(f"Found {len(videos)} videos in {args.video_dir.resolve()}")
    print("Execution order: finish one API/backend combo over all videos, then switch combo.")

    for api in args.apis:
        if api in {"decode_only", "decord"}:
            for decode_backend in args.decode_backends:
                combo_ok = (
                    _backend_is_available(decode_backend)
                    if api == "decode_only"
                    else _decord_backend_is_available(decode_backend, args.decode_device_id)
                )
                if not combo_ok:
                    print(f"\n=== API={api} decode={decode_backend} ===")
                    print("Skipped: backend unavailable in current runtime/build.")
                    continue
                _benchmark_decode_like(
                    api=api,
                    decode_backend=decode_backend,
                    videos=videos,
                    args=args,
                )
            continue

        for decode_backend in args.decode_backends:
            for process_backend in args.process_backends:
                resolved_decode = _resolve_decode_backend(decode_backend)
                resolved_process = _resolve_process_backend(process_backend, resolved_decode)
                combo_ok = True
                if resolved_decode == "gpu" or resolved_process == "gpu":
                    # For patch_stream mixed combos, backend probing can be conservative.
                    # If CUDA runtime is available, run and let per-video execution report
                    # precise backend/module errors instead of skipping early.
                    combo_ok = _gpu_runtime_available()
                if not combo_ok:
                    print(
                        f"\n=== API=patch_stream decode={decode_backend} process={process_backend} ==="
                    )
                    print("Skipped: backend unavailable in current runtime/build.")
                    continue
                _benchmark_patch_combo(
                    decode_backend=decode_backend,
                    process_backend=process_backend,
                    videos=videos,
                    args=args,
                )


if __name__ == "__main__":
    main()
