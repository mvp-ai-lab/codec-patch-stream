#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from codec_patch_stream import DecodeConfig, PatchStreamConfig, decode_only, patch_stream

DEFAULT_EXTS = ("mp4", "mkv", "avi", "mov", "webm", "flv", "m4v", "ts")


@dataclass
class DecodeRun:
    elapsed_s: float
    shape: tuple[int, ...]
    dtype: str
    sampled_frame_ids: list[int]


@dataclass
class PatchRun:
    elapsed_s: float
    patch_shape: tuple[int, ...]
    patch_dtype: str
    selected_patches: int
    sampled_frame_ids: list[int]


def normalize_dtype(dtype: object) -> str:
    s = str(dtype)
    if s.startswith("torch."):
        return s[len("torch.") :]
    return s


def normalize_exts(raw_exts: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for ext in raw_exts:
        e = ext.strip().lower()
        if not e:
            continue
        if e.startswith("."):
            e = e[1:]
        out.add(e)
    if not out:
        raise ValueError("No valid extensions from --ext")
    return out


def collect_videos(
    input_path: Path, exts: set[str], recursive: bool, limit: int
) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    pattern = "**/*" if recursive else "*"
    videos = [
        p
        for p in input_path.glob(pattern)
        if p.is_file() and p.suffix.lower().lstrip(".") in exts
    ]
    videos.sort()
    if limit > 0:
        videos = videos[:limit]
    if not videos:
        raise RuntimeError(
            f"No videos found in {input_path} with extensions: {sorted(exts)}"
        )
    return videos


def print_table(title: str, rows: list[tuple[str, str]]) -> None:
    key_header = "Field"
    val_header = "Value"
    key_w = max(len(key_header), *(len(k) for k, _ in rows))
    val_w = max(len(val_header), *(len(v) for _, v in rows))
    border = f"+-{'-' * key_w}-+-{'-' * val_w}-+"

    print(title)
    print(border)
    print(f"| {key_header.ljust(key_w)} | {val_header.ljust(val_w)} |")
    print(border)
    for k, v in rows:
        print(f"| {k.ljust(key_w)} | {v.ljust(val_w)} |")
    print(border)


def stat(values: list[float]) -> tuple[float, float, float, float]:
    mean_v = statistics.mean(values)
    std_v = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean_v, std_v, min(values), max(values)


def pick_order(order: str, round_idx: int) -> tuple[str, str]:
    if order == "decode-first":
        return ("decode", "patch")
    if order == "patch-first":
        return ("patch", "decode")
    if round_idx % 2 == 0:
        return ("decode", "patch")
    return ("patch", "decode")


def sync_if_needed(gpu: int) -> None:
    if gpu >= 0:
        torch.cuda.synchronize(torch.device("cuda", gpu))


def run_decode_once(
    video_path: Path, args: argparse.Namespace, backend: str, device_id: int
) -> DecodeRun:
    sync_if_needed(args.gpu)
    t0 = time.perf_counter()
    decoded = decode_only(
        DecodeConfig(
            video_path=video_path,
            sequence_length=int(args.num_frames),
            backend=backend,
            device_id=int(device_id),
            decode_mode=str(args.decode_mode),
            uniform_strategy=str(args.uniform_strategy),
            decode_threads=int(args.cpu_threads) if backend == "cpu" else None,
        )
    )
    frames = decoded.frames
    if int(frames.shape[0]) != args.num_frames:
        raise RuntimeError(
            f"decode_only returned {int(frames.shape[0])} frames, "
            f"expected {args.num_frames}"
        )
    sync_if_needed(args.gpu)
    elapsed_s = time.perf_counter() - t0
    return DecodeRun(
        elapsed_s=elapsed_s,
        shape=tuple(int(x) for x in frames.shape),
        dtype=normalize_dtype(frames.dtype),
        sampled_frame_ids=[int(x) for x in decoded.sampled_frame_ids],
    )


def run_patch_once(
    video_path: Path, args: argparse.Namespace, backend: str, device_id: int
) -> PatchRun:
    sync_if_needed(args.gpu)
    t0 = time.perf_counter()
    stream = patch_stream(
        PatchStreamConfig(
            video_path=str(video_path),
            sequence_length=int(args.num_frames),
            decode_mode=str(args.decode_mode),
            uniform_strategy=str(args.uniform_strategy),
            input_size=int(args.input_size),
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            patch_size=int(args.patch),
            k_keep=int(args.topk),
            selection_unit=str(args.selection_unit),
            static_fallback=bool(args.static_fallback),
            static_abs_thresh=float(args.static_abs_thresh),
            static_rel_thresh=float(args.static_rel_thresh),
            static_uniform_frames=int(args.static_uniform_frames),
            output_dtype=str(args.dtype),
            device_id=int(device_id),
            backend=str(backend),
            decode_threads=int(args.cpu_threads) if backend == "cpu" else None,
        )
    )
    try:
        selected_patches = int(len(stream))
        patches = stream.patches
        sampled_frame_ids = [int(x) for x in stream.sampled_frame_ids]
        patch_shape = tuple(int(x) for x in patches.shape)
        patch_dtype = normalize_dtype(patches.dtype)
    finally:
        stream.close()
    sync_if_needed(args.gpu)
    elapsed_s = time.perf_counter() - t0
    return PatchRun(
        elapsed_s=elapsed_s,
        patch_shape=patch_shape,
        patch_dtype=patch_dtype,
        selected_patches=selected_patches,
        sampled_frame_ids=sampled_frame_ids,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark codec-patch-stream decode-only vs patch-streaming latency "
            "with the same sampled frame count."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to input video or a folder containing videos",
    )
    parser.add_argument(
        "--ext",
        nargs="+",
        default=list(DEFAULT_EXTS),
        help="Video extensions for folder mode, e.g. mp4 mkv mov",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="In folder mode, only scan top-level files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of videos to use in folder mode, 0 means no limit",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Initial sampled frame count shared by both pipelines",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="GPU id, -1 means CPU backend",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup rounds per video")
    parser.add_argument("--runs", type=int, default=10, help="Measured rounds per video")
    parser.add_argument(
        "--order",
        type=str,
        default="alternate",
        choices=["alternate", "decode-first", "patch-first"],
        help="Execution order per round",
    )
    parser.add_argument(
        "--decode-mode",
        type=str,
        default="throughput",
        choices=["throughput", "latency", "auto"],
        help="Decode mode for decode_only",
    )
    parser.add_argument(
        "--uniform-strategy",
        type=str,
        default="auto",
        choices=["auto", "seek", "stream"],
        help="Uniform decode strategy (GPU planner path; CPU ignores and uses seek-index)",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="CPU decode threads for codec-patch-stream CPU backend",
    )
    parser.add_argument(
        "--no-strict-frame-id-check",
        action="store_true",
        help="Allow sampled frame id mismatch between pipelines",
    )

    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--patch", type=int, default=14)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument(
        "--selection-unit",
        type=str,
        default="patch",
        choices=["patch", "block2x2"],
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32", "bfloat16", "float16", "float32"],
    )
    parser.add_argument("--static-fallback", action="store_true")
    parser.add_argument("--static-abs-thresh", type=float, default=2.0)
    parser.add_argument("--static-rel-thresh", type=float, default=0.15)
    parser.add_argument("--static-uniform-frames", type=int, default=4)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a single-line JSON summary in addition to tables",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exts = normalize_exts(args.ext)
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    videos = collect_videos(
        args.input_path, exts, recursive=not args.no_recursive, limit=int(args.limit)
    )

    if args.num_frames <= 0:
        raise ValueError("--num-frames must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.runs <= 0:
        raise ValueError("--runs must be > 0")
    if args.cpu_threads < 0:
        raise ValueError("--cpu-threads must be >= 0")
    if args.gpu < -1:
        raise ValueError("--gpu must be -1 or >= 0")
    if args.selection_unit == "block2x2" and args.topk % 4 != 0:
        raise ValueError("--topk must be divisible by 4 when --selection-unit=block2x2")

    if args.gpu >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but --gpu >= 0 was requested")
        torch.cuda.set_device(args.gpu)
        backend = "gpu"
        device_id = args.gpu
    else:
        backend = "cpu"
        device_id = 0

    strict_check = not args.no_strict_frame_id_check
    total_rounds = args.warmup + args.runs

    decode_samples: list[float] = []
    patch_samples: list[float] = []
    frame_id_mismatch_rounds = 0
    decode_meta: DecodeRun | None = None
    patch_meta: PatchRun | None = None

    for vid_idx, video_path in enumerate(videos, start=1):
        if len(videos) > 1:
            print(f"[{vid_idx}/{len(videos)}] {video_path}")

        for i in range(total_rounds):
            first, second = pick_order(args.order, i)
            round_decode: DecodeRun | None = None
            round_patch: PatchRun | None = None

            for pipeline in (first, second):
                if pipeline == "decode":
                    round_decode = run_decode_once(video_path, args, backend, device_id)
                else:
                    round_patch = run_patch_once(video_path, args, backend, device_id)

            if round_decode is None or round_patch is None:
                raise RuntimeError("Internal error: round did not run both pipelines")

            if round_decode.sampled_frame_ids != round_patch.sampled_frame_ids:
                frame_id_mismatch_rounds += 1
                if strict_check:
                    raise RuntimeError(
                        "Sampled frame ids mismatch between decode-only and patch-streaming.\n"
                        f"video={video_path}\n"
                        f"decode ids (head): {round_decode.sampled_frame_ids[:8]}\n"
                        f"patch  ids (head): {round_patch.sampled_frame_ids[:8]}"
                    )

            if i >= args.warmup:
                decode_samples.append(round_decode.elapsed_s)
                patch_samples.append(round_patch.elapsed_s)
                if decode_meta is None:
                    decode_meta = round_decode
                if patch_meta is None:
                    patch_meta = round_patch

    if not decode_samples or not patch_samples or decode_meta is None or patch_meta is None:
        raise RuntimeError("No benchmark samples collected")

    dec_mean, dec_std, dec_min, dec_max = stat(decode_samples)
    pat_mean, pat_std, pat_min, pat_max = stat(patch_samples)

    slowdown = pat_mean / dec_mean
    if slowdown >= 1.0:
        verdict = f"patch streaming slower: {slowdown:.3f}x decode-only latency"
    else:
        verdict = f"patch streaming faster: {(1.0 / slowdown):.3f}x decode-only latency"

    cfg_rows = [
        ("input_path", str(args.input_path)),
        ("videos_total", str(len(videos))),
        ("video_limit", "none" if args.limit == 0 else str(args.limit)),
        ("context", "cpu" if args.gpu < 0 else f"gpu:{args.gpu}"),
        ("initial_sampled_frames", str(args.num_frames)),
        ("order", args.order),
        ("runs_per_video", f"{args.runs} (warmup={args.warmup})"),
        ("measured_samples", str(len(decode_samples))),
        ("strict_frame_id_check", str(strict_check)),
        ("frame_id_mismatch_rounds", str(frame_id_mismatch_rounds)),
        ("decode_mode", args.decode_mode),
        ("uniform_strategy", args.uniform_strategy),
        ("patch_input_size", str(args.input_size)),
        ("patch_min_pixels", str(args.min_pixels)),
        ("patch_max_pixels", str(args.max_pixels)),
        ("patch_size", str(args.patch)),
        ("k_keep", str(args.topk)),
        ("selection_unit", args.selection_unit),
        ("output_dtype", args.dtype),
    ]
    if args.gpu < 0:
        cfg_rows.append(("cpu_threads", str(args.cpu_threads)))

    out_rows = [
        ("decode_output_shape", str(decode_meta.shape)),
        ("decode_output_dtype", decode_meta.dtype),
        ("patch_output_shape", str(patch_meta.patch_shape)),
        ("patch_output_dtype", patch_meta.patch_dtype),
        ("patch_selected", str(patch_meta.selected_patches)),
        (
            "patch_selected_per_frame",
            f"{patch_meta.selected_patches / float(args.num_frames):.2f}",
        ),
    ]

    timing_rows = [
        (
            "decode_only_ms",
            (
                f"mean={dec_mean * 1000.0:.3f}, std={dec_std * 1000.0:.3f}, "
                f"min={dec_min * 1000.0:.3f}, max={dec_max * 1000.0:.3f}"
            ),
        ),
        (
            "patch_stream_ms",
            (
                f"mean={pat_mean * 1000.0:.3f}, std={pat_std * 1000.0:.3f}, "
                f"min={pat_min * 1000.0:.3f}, max={pat_max * 1000.0:.3f}"
            ),
        ),
        ("latency_ratio_patch_over_decode", f"{slowdown:.3f}"),
        ("verdict", verdict),
    ]

    print_table("Config", cfg_rows)
    print()
    print_table("Outputs", out_rows)
    print()
    print_table("Timing", timing_rows)
    if args.json:
        payload = {
            "config": {
                "input_path": str(args.input_path),
                "videos_total": len(videos),
                "video_limit": None if args.limit == 0 else int(args.limit),
                "context": "cpu" if args.gpu < 0 else f"gpu:{args.gpu}",
                "initial_sampled_frames": int(args.num_frames),
                "order": str(args.order),
                "runs_per_video": int(args.runs),
                "warmup_per_video": int(args.warmup),
                "measured_samples": int(len(decode_samples)),
                "strict_frame_id_check": bool(strict_check),
                "frame_id_mismatch_rounds": int(frame_id_mismatch_rounds),
                "decode_mode": str(args.decode_mode),
                "uniform_strategy": str(args.uniform_strategy),
                "patch_input_size": int(args.input_size),
                "patch_min_pixels": args.min_pixels,
                "patch_max_pixels": args.max_pixels,
                "patch_size": int(args.patch),
                "k_keep": int(args.topk),
                "selection_unit": str(args.selection_unit),
                "output_dtype": str(args.dtype),
                "cpu_threads": int(args.cpu_threads) if args.gpu < 0 else None,
            },
            "outputs": {
                "decode_output_shape": list(decode_meta.shape),
                "decode_output_dtype": str(decode_meta.dtype),
                "patch_output_shape": list(patch_meta.patch_shape),
                "patch_output_dtype": str(patch_meta.patch_dtype),
                "patch_selected": int(patch_meta.selected_patches),
                "patch_selected_per_frame": float(
                    patch_meta.selected_patches / float(args.num_frames)
                ),
            },
            "timing": {
                "decode_only_ms": {
                    "mean": float(dec_mean * 1000.0),
                    "std": float(dec_std * 1000.0),
                    "min": float(dec_min * 1000.0),
                    "max": float(dec_max * 1000.0),
                },
                "patch_stream_ms": {
                    "mean": float(pat_mean * 1000.0),
                    "std": float(pat_std * 1000.0),
                    "min": float(pat_min * 1000.0),
                    "max": float(pat_max * 1000.0),
                },
                "latency_ratio_patch_over_decode": float(slowdown),
                "verdict": str(verdict),
            },
        }
        print(json.dumps(payload, ensure_ascii=True))


if __name__ == "__main__":
    main()
