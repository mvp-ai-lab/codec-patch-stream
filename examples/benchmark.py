from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from codec_patch_stream import CodecPatchStream


@dataclass
class BenchResult:
    samples_s: list[float]
    selected_patches: int

    @property
    def mean_s(self) -> float:
        return statistics.mean(self.samples_s)

    @property
    def std_s(self) -> float:
        if len(self.samples_s) < 2:
            return 0.0
        return statistics.stdev(self.samples_s)

    @property
    def min_s(self) -> float:
        return min(self.samples_s)

    @property
    def max_s(self) -> float:
        return max(self.samples_s)


def _sync(device_id: int) -> None:
    torch.cuda.synchronize(torch.device("cuda", device_id))


def _measure_prepare(
    fn: Callable[[], int],
    device_id: int,
    cuda_sync: bool,
    repeats: int,
    warmup: int,
) -> BenchResult:
    samples_s: list[float] = []
    selected_patches: int | None = None

    for i in range(warmup + repeats):
        if cuda_sync:
            _sync(device_id)
        t0 = time.perf_counter()
        cur_patches = fn()
        if cuda_sync:
            _sync(device_id)
        dt = time.perf_counter() - t0

        if i >= warmup:
            samples_s.append(dt)
            if selected_patches is None:
                selected_patches = cur_patches

    if selected_patches is None:
        raise RuntimeError("No benchmark samples collected")
    return BenchResult(samples_s=samples_s, selected_patches=selected_patches)


def _build_stream(args: argparse.Namespace) -> CodecPatchStream:
    return CodecPatchStream(
        video_path=args.video,
        sequence_length=args.frames,
        input_size=args.input_size,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        patch_size=args.patch,
        k_keep=args.topk,
        static_fallback=args.static_fallback,
        static_abs_thresh=args.static_abs_thresh,
        static_rel_thresh=args.static_rel_thresh,
        static_uniform_frames=args.static_uniform_frames,
        output_dtype=args.dtype,
        device_id=args.device_id,
        backend=args.backend,
    )


def _fmt_ms(x: float) -> str:
    return f"{x * 1000.0:.2f}"


def _fmt_int(x: int) -> str:
    return f"{x:,}"


def _section(title: str, width: int = 84) -> None:
    line = "=" * width
    print()
    print(line)
    print(title)
    print(line)


def _print_table(rows: list[tuple[str, str]], key_header: str = "Field", value_header: str = "Value") -> None:
    if not rows:
        return
    key_w = max(len(key_header), *(len(k) for k, _ in rows))
    val_w = max(len(value_header), *(len(v) for _, v in rows))

    border = f"+-{'-' * key_w}-+-{'-' * val_w}-+"
    print(border)
    print(f"| {key_header.ljust(key_w)} | {value_header.ljust(val_w)} |")
    print(border)
    for k, v in rows:
        print(f"| {k.ljust(key_w)} | {v.ljust(val_w)} |")
    print(border)


def _to_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _to_int(x: Any) -> int | None:
    try:
        if x is None:
            return None
        return int(x)
    except (TypeError, ValueError):
        return None


def _parse_fraction(v: str | None) -> float | None:
    if not v:
        return None
    if "/" not in v:
        return _to_float(v)
    num_s, den_s = v.split("/", 1)
    num = _to_float(num_s)
    den = _to_float(den_s)
    if num is None or den is None or den == 0.0:
        return None
    return num / den


def _probe_video_info(video_path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(video_path.resolve()),
        "size_bytes": video_path.stat().st_size,
        "size_mb": video_path.stat().st_size / (1024.0 * 1024.0),
    }
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        info["ffprobe_error"] = "ffprobe not found in PATH"
        return info

    if proc.returncode != 0:
        info["ffprobe_error"] = proc.stderr.strip() or f"ffprobe exited with code {proc.returncode}"
        return info

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        info["ffprobe_error"] = "ffprobe output is not valid JSON"
        return info

    fmt = payload.get("format", {})
    streams = payload.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)

    info["container"] = fmt.get("format_name")
    info["duration_sec"] = _to_float(fmt.get("duration"))
    info["bitrate_bps"] = _to_int(fmt.get("bit_rate"))

    if video_stream is not None:
        info["codec"] = video_stream.get("codec_name")
        info["profile"] = video_stream.get("profile")
        info["width"] = _to_int(video_stream.get("width"))
        info["height"] = _to_int(video_stream.get("height"))
        info["pix_fmt"] = video_stream.get("pix_fmt")
        info["nb_frames"] = _to_int(video_stream.get("nb_frames"))
        info["avg_fps"] = _parse_fraction(video_stream.get("avg_frame_rate"))
        info["r_fps"] = _parse_fraction(video_stream.get("r_frame_rate"))

    return info


def _print_video_info(info: dict[str, Any]) -> None:
    _section("Video Info")
    rows: list[tuple[str, str]] = [
        ("path", info["path"]),
        ("size", f"{info['size_mb']:.2f} MB ({_fmt_int(info['size_bytes'])} bytes)"),
    ]

    ffprobe_error = info.get("ffprobe_error")
    if ffprobe_error:
        rows.append(("ffprobe", ffprobe_error))
        _print_table(rows)
        return

    rows.append(("container", str(info.get("container"))))
    if info.get("duration_sec") is not None:
        rows.append(("duration", f"{info['duration_sec']:.3f} s"))
    if info.get("bitrate_bps") is not None:
        rows.append(("bitrate", f"{_fmt_int(info['bitrate_bps'])} bps"))
    if info.get("codec") is not None:
        rows.append(("codec", f"{info.get('codec')} ({info.get('profile')})"))
    if info.get("width") is not None and info.get("height") is not None:
        rows.append(("resolution", f"{info['width']}x{info['height']}"))
    if info.get("pix_fmt") is not None:
        rows.append(("pixel_format", str(info["pix_fmt"])))
    if info.get("avg_fps") is not None:
        rows.append(("avg_fps", f"{info['avg_fps']:.3f}"))
    elif info.get("r_fps") is not None:
        rows.append(("fps", f"{info['r_fps']:.3f}"))
    if info.get("nb_frames") is not None:
        rows.append(("frames", _fmt_int(info["nb_frames"])))

    _print_table(rows)


def _print_selected_params(args: argparse.Namespace) -> None:
    _section("Selected Params")
    rows = [
        ("device_id", str(args.device_id)),
        ("backend", str(args.backend)),
        ("sequence_length", str(args.frames)),
        ("input_size", str(args.input_size)),
        ("min_pixels", str(args.min_pixels)),
        ("max_pixels", str(args.max_pixels)),
        ("patch_size", str(args.patch)),
        ("k_keep", str(args.topk)),
        ("output_dtype", args.dtype),
        ("static_fallback", str(args.static_fallback)),
        ("static_abs_thresh", str(args.static_abs_thresh)),
        ("static_rel_thresh", str(args.static_rel_thresh)),
        ("static_uniform_frames", str(args.static_uniform_frames)),
        ("prepare_warmup", str(args.prepare_warmup)),
        ("prepare_repeats", str(args.prepare_repeats)),
    ]
    _print_table(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark CodecPatchStream latency")
    p.add_argument("video", type=str)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--min-pixels", type=int, default=None)
    p.add_argument("--max-pixels", type=int, default=None)
    p.add_argument("--patch", type=int, default=14)
    p.add_argument("--topk", type=int, default=1024)
    p.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32", "bfloat16", "float16", "float32"],
    )
    p.add_argument("--prepare-repeats", type=int, default=5)
    p.add_argument("--prepare-warmup", type=int, default=1)
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--backend", choices=["auto", "gpu", "cpu"], default="auto")

    p.add_argument("--static-fallback", action="store_true")
    p.add_argument("--static-abs-thresh", type=float, default=2.0)
    p.add_argument("--static-rel-thresh", type=float, default=0.15)
    p.add_argument("--static-uniform-frames", type=int, default=4)
    args = p.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if args.prepare_repeats <= 0:
        raise ValueError("prepare-repeats must be > 0")
    if args.prepare_warmup < 0:
        raise ValueError("prepare-warmup must be >= 0")

    if args.backend == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but backend=gpu was requested.")
        cuda_sync = True
        torch.cuda.set_device(args.device_id)
        device_label = f"cuda:{args.device_id}"
    elif args.backend == "cpu":
        cuda_sync = False
        device_label = "cpu"
    else:
        cuda_sync = False
        device_label = "auto (runtime selected)"

    _section("CodecPatchStream Benchmark")
    print(f"Device: {device_label}")
    _print_video_info(_probe_video_info(video_path))
    _print_selected_params(args)

    def bench_prepare_once() -> int:
        stream = _build_stream(args)
        n = len(stream)
        stream.close()
        return int(n)

    _section("Benchmark Results")
    result = _measure_prepare(
        fn=bench_prepare_once,
        device_id=args.device_id,
        cuda_sync=cuda_sync,
        repeats=args.prepare_repeats,
        warmup=args.prepare_warmup,
    )
    _print_table(
        [
            ("selected_patches", _fmt_int(result.selected_patches)),
            ("mean_ms", _fmt_ms(result.mean_s)),
            ("std_ms", _fmt_ms(result.std_s)),
            ("min_ms", _fmt_ms(result.min_s)),
            ("max_ms", _fmt_ms(result.max_s)),
        ],
        key_header="Metric",
        value_header="Value",
    )


if __name__ == "__main__":
    main()
