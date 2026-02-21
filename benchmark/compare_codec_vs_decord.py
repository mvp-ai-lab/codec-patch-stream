#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

DEFAULT_EXTS = ("mp4", "mkv", "avi", "mov", "webm", "flv", "m4v", "ts")


def print_table(headers: list[str], rows: list[list[str]], title: str | None = None) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    border = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    if title:
        print(title)
    print(border)
    print("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |")
    print(border)
    for row in rows:
        print("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    print(border)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark decord vs codec-patch-stream in the same environment "
            "(uniformly sample frames; GPU mode avoids host copy, CPU mode uses numpy output)"
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
    parser.add_argument("--num-frames", type=int, default=16, help="Uniform sampled frame count")
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Deprecated alias for --codec-gpu when --codec-gpu is omitted",
    )
    parser.add_argument(
        "--decord-gpu",
        type=int,
        default=-1,
        help="decord context GPU id, -1 for CPU (default: -1)",
    )
    parser.add_argument(
        "--codec-gpu",
        type=int,
        default=None,
        help="codec-patch-stream context GPU id, -1 for CPU (default: value of --gpu, else -1)",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="CPU decode threads for both libs, 0 means auto",
    )
    parser.add_argument(
        "--cpu-affinity",
        type=str,
        default="",
        help="Optional CPU affinity, e.g. '0-7' or '0,2,4,6'",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=-1,
        help="Resize width, must be -1 for fair decode-only compare",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=-1,
        help="Resize height, must be -1 for fair decode-only compare",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup rounds per video")
    parser.add_argument("--runs", type=int, default=20, help="Measured rounds per video")
    parser.add_argument(
        "--order",
        type=str,
        default="alternate",
        choices=["alternate", "decord-first", "codec-first"],
        help="Execution order per round",
    )
    parser.add_argument(
        "--no-strict-shape-check",
        action="store_true",
        help="Disable strict check that both libs output same shape/dtype",
    )
    parser.add_argument(
        "--codec-mode",
        type=str,
        default="throughput",
        choices=["throughput", "latency", "auto"],
        help="codec-patch-stream decode mode (default: throughput)",
    )
    parser.add_argument(
        "--no-isolate-process",
        action="store_true",
        help="Disable process isolation and run both libraries in one process",
    )
    parser.add_argument(
        "--worker-lib",
        type=str,
        default="",
        choices=["", "decord", "codec"],
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def build_uniform_indices(total_frames: int, num_frames: int) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError("Video has no frame")
    if num_frames <= 0:
        raise ValueError("--num-frames must be > 0")
    if total_frames == 1:
        return np.zeros((num_frames,), dtype=np.int64)
    return np.linspace(0, total_frames - 1, num_frames, dtype=np.int64)


def parse_cpu_affinity(spec: str) -> set[int]:
    out: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            lo = int(a)
            hi = int(b)
            if lo > hi:
                raise ValueError(f"Invalid affinity range: {token}")
            for x in range(lo, hi + 1):
                out.add(x)
        else:
            out.add(int(token))
    if not out:
        raise ValueError("Empty --cpu-affinity")
    return out


def maybe_sync(gpu: int) -> None:
    if gpu >= 0:
        torch.cuda.synchronize(torch.device("cuda", gpu))


def resolve_context_gpus(args: argparse.Namespace) -> tuple[int, int]:
    codec_gpu = args.codec_gpu if args.codec_gpu is not None else args.gpu
    if codec_gpu is None:
        codec_gpu = -1
    return int(args.decord_gpu), int(codec_gpu)


def run_decord_once(
    video_path: Path,
    num_frames: int,
    gpu: int,
    cpu_threads: int,
    width: int,
    height: int,
) -> tuple[float, tuple[int, ...], str]:
    import decord

    ctx = decord.cpu() if gpu < 0 else decord.gpu(gpu)
    maybe_sync(gpu)
    t0 = time.perf_counter()
    try:
        vr = decord.VideoReader(
            str(video_path),
            ctx=ctx,
            width=width,
            height=height,
            num_threads=cpu_threads if gpu < 0 else 0,
        )
    except Exception as exc:
        if gpu >= 0 and "CUDA not enabled" in str(exc):
            raise RuntimeError(
                "decord GPU backend is unavailable in this environment "
                "(built without CUDA). Rebuild decord with CUDA or use --gpu -1."
            ) from exc
        raise
    indices = build_uniform_indices(len(vr), num_frames)
    batch = vr.get_batch(indices)
    if gpu >= 0:
        shape = tuple(int(x) for x in batch.shape)
        dtype = normalize_dtype(batch.dtype)
    else:
        frames_np = batch.asnumpy()
        shape = tuple(frames_np.shape)
        dtype = normalize_dtype(frames_np.dtype)
    maybe_sync(gpu)
    elapsed = time.perf_counter() - t0
    return elapsed, shape, dtype


def run_codec_once(
    video_path: Path,
    num_frames: int,
    gpu: int,
    cpu_threads: int,
    mode: str,
) -> tuple[float, tuple[int, ...], str]:
    from codec_patch_stream import DecodeConfig, decode_only

    backend = "cpu" if gpu < 0 else "gpu"
    device_id = 0 if gpu < 0 else gpu
    maybe_sync(gpu)
    t0 = time.perf_counter()
    decoded = decode_only(
        DecodeConfig(
            video_path=video_path,
            sequence_length=int(num_frames),
            backend=backend,
            device_id=int(device_id),
            decode_mode=str(mode),
            decode_threads=int(cpu_threads) if gpu < 0 else None,
        )
    )
    if gpu >= 0:
        frames = decoded.frames
        shape = tuple(frames.shape)
        dtype = normalize_dtype(frames.dtype)
    else:
        frames_np = decoded.asnumpy()
        shape = tuple(frames_np.shape)
        dtype = normalize_dtype(frames_np.dtype)
    maybe_sync(gpu)
    elapsed = time.perf_counter() - t0
    return elapsed, shape, dtype


def pick_order(order: str, round_idx: int) -> tuple[str, str]:
    if order == "decord-first":
        return ("decord", "codec")
    if order == "codec-first":
        return ("codec", "decord")
    if round_idx % 2 == 0:
        return ("decord", "codec")
    return ("codec", "decord")


def stat(values: list[float]) -> tuple[float, float, float, float]:
    mean_v = statistics.mean(values)
    std_v = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean_v, std_v, min(values), max(values)


class DecodeWorker:
    def __init__(
        self,
        lib: str,
        args: argparse.Namespace,
        decord_gpu: int,
        codec_gpu: int,
    ) -> None:
        script = Path(__file__).resolve()
        cmd = [
            sys.executable,
            str(script),
            str(args.input_path),
            "--num-frames",
            str(args.num_frames),
            "--decord-gpu",
            str(decord_gpu),
            "--codec-gpu",
            str(codec_gpu),
            "--cpu-threads",
            str(args.cpu_threads),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--codec-mode",
            str(args.codec_mode),
            "--worker-lib",
            lib,
            "--no-isolate-process",
        ]
        self.lib = lib
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )

    def run_once(self, video_path: Path | None = None) -> tuple[float, tuple[int, ...], str]:
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError(f"{self.lib} worker has no stdio pipes")
        if self.proc.poll() is not None:
            raise RuntimeError(f"{self.lib} worker exited early with code {self.proc.returncode}")

        payload: dict[str, object] = {"cmd": "run"}
        if video_path is not None:
            payload["video"] = str(video_path)
        self.proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()

        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.lib} worker closed stdout unexpectedly")
        reply = json.loads(line)
        if not reply.get("ok", False):
            msg = reply.get("error", "unknown worker error")
            raise RuntimeError(f"{self.lib} worker failed: {msg}")

        elapsed = float(reply["elapsed"])
        shape = tuple(int(x) for x in reply["shape"])
        dtype = str(reply["dtype"])
        return elapsed, shape, dtype

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
                self.proc.stdin.flush()
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5.0)
        except Exception:
            self.proc.kill()
            self.proc.wait(timeout=5.0)


def worker_loop(args: argparse.Namespace, decord_gpu: int, codec_gpu: int) -> int:
    lib = args.worker_lib
    if lib not in {"decord", "codec"}:
        return 2

    default_video: Path | None = args.input_path if args.input_path.is_file() else None

    while True:
        line = sys.stdin.readline()
        if not line:
            return 0
        raw = line.strip()
        if not raw:
            continue

        cmd = ""
        payload: dict[str, object] = {}
        if raw.startswith("{"):
            try:
                payload = json.loads(raw)
            except Exception as exc:
                print(
                    json.dumps({"ok": False, "error": f"invalid json command: {exc}"}),
                    flush=True,
                )
                continue
            cmd = str(payload.get("cmd", "")).strip().lower()
        else:
            cmd = raw.lower()

        if cmd == "exit":
            return 0
        if cmd != "run":
            print(json.dumps({"ok": False, "error": f"unknown command: {cmd}"}), flush=True)
            continue

        video_raw = payload.get("video")
        if video_raw is None:
            video_path = default_video
        else:
            video_path = Path(str(video_raw))
        if video_path is None or not video_path.is_file():
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": "run command requires a valid video path in worker mode",
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            continue

        try:
            if lib == "decord":
                elapsed, shape, dtype = run_decord_once(
                    video_path=video_path,
                    num_frames=args.num_frames,
                    gpu=decord_gpu,
                    cpu_threads=args.cpu_threads,
                    width=args.width,
                    height=args.height,
                )
            else:
                elapsed, shape, dtype = run_codec_once(
                    video_path=video_path,
                    num_frames=args.num_frames,
                    gpu=codec_gpu,
                    cpu_threads=args.cpu_threads,
                    mode=args.codec_mode,
                )
            print(
                json.dumps(
                    {
                        "ok": True,
                        "elapsed": float(elapsed),
                        "shape": [int(x) for x in shape],
                        "dtype": str(dtype),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
        except Exception as exc:
            print(
                json.dumps(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                    separators=(",", ":"),
                ),
                flush=True,
            )


def main() -> None:
    args = parse_args()
    decord_gpu, codec_gpu = resolve_context_gpus(args)

    if args.worker_lib:
        raise SystemExit(worker_loop(args, decord_gpu, codec_gpu))

    exts = normalize_exts(args.ext)
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    videos = collect_videos(
        args.input_path, exts, recursive=not args.no_recursive, limit=int(args.limit)
    )

    if args.num_frames <= 0:
        raise ValueError("--num-frames must be > 0")
    if args.cpu_threads < 0:
        raise ValueError("--cpu-threads must be >= 0")
    if args.width != -1 or args.height != -1:
        raise ValueError(
            "codec-patch-stream pure decode API does not include resize; "
            "use --width -1 --height -1 for fair comparison"
        )
    if args.cpu_affinity:
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("Current platform does not support sched_setaffinity")
        cpu_set = parse_cpu_affinity(args.cpu_affinity)
        os.sched_setaffinity(0, cpu_set)

    strict_check = not args.no_strict_shape_check
    total_rounds = args.warmup + args.runs
    use_isolation = not args.no_isolate_process

    samples = {"decord": [], "codec": []}
    out_shape: dict[str, tuple[int, ...] | None] = {"decord": None, "codec": None}
    out_dtype: dict[str, str | None] = {"decord": None, "codec": None}

    workers: dict[str, DecodeWorker] = {}
    try:
        if use_isolation:
            workers["decord"] = DecodeWorker("decord", args, decord_gpu, codec_gpu)
            workers["codec"] = DecodeWorker("codec", args, decord_gpu, codec_gpu)

        for vid_idx, video_path in enumerate(videos, start=1):
            if len(videos) > 1:
                print(f"[{vid_idx}/{len(videos)}] {video_path}")
            for i in range(total_rounds):
                first, second = pick_order(args.order, i)
                round_results: dict[str, tuple[float, tuple[int, ...], str]] = {}

                for lib in (first, second):
                    if use_isolation:
                        elapsed, shape, dtype = workers[lib].run_once(video_path)
                    elif lib == "decord":
                        elapsed, shape, dtype = run_decord_once(
                            video_path=video_path,
                            num_frames=args.num_frames,
                            gpu=decord_gpu,
                            cpu_threads=args.cpu_threads,
                            width=args.width,
                            height=args.height,
                        )
                    else:
                        elapsed, shape, dtype = run_codec_once(
                            video_path=video_path,
                            num_frames=args.num_frames,
                            gpu=codec_gpu,
                            cpu_threads=args.cpu_threads,
                            mode=args.codec_mode,
                        )

                    round_results[lib] = (elapsed, shape, dtype)
                    if i >= args.warmup:
                        samples[lib].append(elapsed)
                        if out_shape[lib] is None:
                            out_shape[lib] = shape
                            out_dtype[lib] = dtype

                if strict_check:
                    d_shape = round_results["decord"][1]
                    c_shape = round_results["codec"][1]
                    d_dtype = round_results["decord"][2]
                    c_dtype = round_results["codec"][2]
                    if d_shape != c_shape:
                        raise RuntimeError(
                            f"Output shape mismatch at video={video_path}: "
                            f"decord={d_shape}, codec-patch-stream={c_shape}"
                        )
                    if d_dtype != c_dtype:
                        raise RuntimeError(
                            f"Output dtype mismatch at video={video_path}: "
                            f"decord={d_dtype}, codec-patch-stream={c_dtype}"
                        )
    finally:
        for w in workers.values():
            w.close()

    if not samples["decord"] or not samples["codec"]:
        raise RuntimeError("No benchmark samples collected")

    dec_mean, dec_std, dec_min, dec_max = stat(samples["decord"])
    cod_mean, cod_std, cod_min, cod_max = stat(samples["codec"])

    speedup = dec_mean / cod_mean
    if speedup >= 1.0:
        verdict = f"codec-patch-stream faster ({speedup:.3f}x vs decord)"
    else:
        verdict = f"decord faster ({(1.0 / speedup):.3f}x vs codec-patch-stream)"

    print_table(
        headers=["Field", "Value"],
        rows=[
            ["input_path", str(args.input_path)],
            ["videos_total", str(len(videos))],
            ["video_limit", "none" if args.limit == 0 else str(args.limit)],
            ["decord_context", "cpu" if decord_gpu < 0 else f"gpu:{decord_gpu}"],
            ["codec_context", "cpu" if codec_gpu < 0 else f"gpu:{codec_gpu}"],
            ["process_isolation", str(use_isolation)],
            ["cpu_threads", str(args.cpu_threads)],
            ["cpu_affinity", args.cpu_affinity if args.cpu_affinity else "none"],
            ["sampled_frames", str(args.num_frames)],
            ["codec_mode", args.codec_mode],
            ["width", str(args.width)],
            ["height", str(args.height)],
            ["runs_per_video", f"{args.runs} (warmup={args.warmup})"],
            ["measured_samples", str(len(samples["decord"]))],
            ["order", args.order],
            ["strict_shape_check", str(strict_check)],
            [
                "decord_materialization",
                "device tensor (no asnumpy)" if decord_gpu >= 0 else "numpy",
            ],
            [
                "codec_materialization",
                "device tensor (no asnumpy)" if codec_gpu >= 0 else "numpy",
            ],
        ],
        title="Config",
    )
    print()
    print_table(
        headers=["Library", "Output Shape", "Output Dtype"],
        rows=[
            ["decord", str(out_shape["decord"]), str(out_dtype["decord"])],
            ["codec-patch-stream", str(out_shape["codec"]), str(out_dtype["codec"])],
        ],
        title="Output",
    )
    print()
    print_table(
        headers=["Library", "mean_ms", "std_ms", "min_ms", "max_ms"],
        rows=[
            [
                "decord",
                f"{dec_mean * 1000.0:.3f}",
                f"{dec_std * 1000.0:.3f}",
                f"{dec_min * 1000.0:.3f}",
                f"{dec_max * 1000.0:.3f}",
            ],
            [
                "codec-patch-stream",
                f"{cod_mean * 1000.0:.3f}",
                f"{cod_std * 1000.0:.3f}",
                f"{cod_min * 1000.0:.3f}",
                f"{cod_max * 1000.0:.3f}",
            ],
        ],
        title="Timing",
    )
    print()
    print_table(
        headers=["Metric", "Value"],
        rows=[
            ["speedup(decord/codec)", f"{speedup:.6f}"],
            ["conclusion", verdict],
        ],
        title="Comparison",
    )


if __name__ == "__main__":
    main()
