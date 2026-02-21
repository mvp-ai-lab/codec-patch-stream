# Codec Patch Stream

`codec-patch-stream` provides two native APIs on top of shared video decode core logic:

- `decode_only(DecodeConfig)`: uniform sampled decode output.
- `patch_stream(PatchStreamConfig)`: decode + energy-based patch selection.

Pipelines:

- GPU: NVDEC decode + GPU proxy energy (block matching + residual proxy) + top-k patch extraction.
- CPU: FFmpeg software decode + CPU energy path + top-k patch extraction.

> Note: NVDEC does not expose codec motion vectors/residuals directly, so GPU path uses fast estimated proxies from decoded frames.

## Preview

<div style="display: flex; gap: 10px;">
    <img src="./assets/demo.gif" style="height: 300px; width: 300px;" />
    <img src="./assets/output.gif" style="height: 300px; width: 300px;" />
</div>

## Build Prerequisites

- FFmpeg development libs: `libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev`
- GPU build (optional):
  - NVIDIA GPU
  - CUDA toolkit (`nvcc`)
  - NVCodec headers (e.g. `/usr/local/include/ffnvcodec`)

## Install

```bash
uv venv .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python torch
```

CPU backend only:

```bash
CODEC_BUILD_NATIVE=1 CODEC_ENABLE_GPU=0 \
uv pip install -e . --python ./.venv/bin/python --no-build-isolation
```

CPU + GPU backend:

```bash
CODEC_BUILD_NATIVE=1 CODEC_ENABLE_GPU=1 \
uv pip install -e . --python ./.venv/bin/python --no-build-isolation
```

Recommended GPU build env:

```bash
export CODEC_CUDA_ARCH_LIST=9.0   # H100/H200, adjust for your GPU
export MAX_JOBS=16
```

## API Usage

### Patch streaming

```python
from codec_patch_stream import PatchStreamConfig, patch_stream

stream = patch_stream(
    PatchStreamConfig(
        video_path="/path/to/video.mp4",
        sequence_length=16,
        decode_mode="throughput",
        uniform_strategy="auto",
        input_size=224,
        min_pixels=None,
        max_pixels=None,
        patch_size=14,
        k_keep=2048,
        output_dtype="bf16",
        backend="auto",
        device_id=0,
    )
)
patches = stream.patches
meta_fields, meta_scores = stream.metadata_tensors
stream.close()
```

### Frame decoding only

```python
from codec_patch_stream import DecodeConfig, decode_only

decoded = decode_only(
    DecodeConfig(
        video_path="/path/to/video.mp4",
        sequence_length=16,
        backend="auto",
        device_id=0,
        decode_mode="throughput",
        uniform_strategy="auto",
    )
)
frames = decoded.frames
frames_np = decoded.asnumpy()
```

## Runtime Knobs

### DecodeConfig (`decode_only`)

- `video_path` (required): input video path.
- `sequence_length` (default `16`): uniform sampled frame count; must be `> 0`.
Impact: higher value increases decode cost and memory roughly linearly.
- `backend` (default `auto`): `auto | gpu | cpu`.
Impact: `gpu` gives highest throughput when NVDEC is available; `cpu` is deterministic fallback.
- `device_id` (default `0`): CUDA device id for GPU path.
Impact: chooses which GPU owns decode/output tensors.
- `decode_mode` (default `throughput`): `throughput | latency | auto`.
Impact: `throughput` favors sustained QPS, `latency` favors single-request response; `auto` currently resolves to throughput behavior.
- `uniform_strategy` (default `auto`): `auto | seek | stream` (GPU planner path).
Impact: `seek` reduces unnecessary decode on sparse sampling; `stream` can be better when sampling is dense.
- `nvdec_session_pool_size` (default `None` -> internal default): pool size used by throughput mode.
Impact: larger value can improve concurrent throughput but increases VRAM/context pressure.
- `uniform_auto_ratio` (default `None` -> internal default): threshold for `uniform_strategy=auto`.
Impact: lower ratio biases to `stream`; higher ratio biases to `seek`.
- `decode_threads` (default `None` -> internal default): CPU decode thread count.
Impact: too low underutilizes CPU; too high can hurt due to contention.
- `decode_thread_type` (default `None` -> internal default): `auto | frame | slice`.
Impact: codec/content dependent; `auto` is generally safest.
- `reader_cache_size` (default `None` -> internal default): frame-index cache size (`<=0` disables).
Impact: larger cache helps repeated reads of same videos; disabling reduces memory but increases repeated index-build overhead.
- `nvdec_reuse_open_decoder` (default `None` -> internal policy): force NVDEC decoder reuse on/off.
Impact: `1` usually better throughput, `0` can help debug decoder lifecycle issues.

### PatchStreamConfig (`patch_stream`)

- `video_path` (required): input video path.
- `sequence_length` (default `16`): same sampled frame count basis as decode-only.
- `decode_mode` / `uniform_strategy` / `backend` / `device_id`: same semantics as `DecodeConfig`.
- `input_size` (default `224`): fallback area budget via `input_size^2`.
Impact: controls default resize target when `min_pixels/max_pixels` are unset.
- `min_pixels` / `max_pixels` (default `None`): smart-resize area bounds (`H*W`).
Impact: set both to same value for stable compute and stable patch-grid size.
- `patch_size` (default `14`): patch edge length.
Impact: smaller patch gives finer spatial granularity but more candidate patches.
- `k_keep` (default `2048`): number of selected patches.
Impact: larger `k_keep` increases output size and downstream compute.
- `selection_unit` (default `patch`): `patch | block2x2`.
Impact: `block2x2` improves local spatial continuity but requires `k_keep % 4 == 0` and even patch-grid dimensions.
- `static_fallback` / `static_abs_thresh` / `static_rel_thresh` / `static_uniform_frames`: static-scene handling knobs.
Impact: improves coverage diversity when motion is weak; aggressive thresholds may include less informative patches.
- `energy_pct` (default `95.0`): percentile normalization for motion/residual fusion, clamped to `[1,100]`.
Impact: higher values emphasize strong motion; lower values flatten energy distribution.
- `output_dtype` (default `bfloat16`): `bf16|bfloat16|fp16|float16|fp32|float32`.
Impact: lower precision saves memory/bandwidth; `fp32` is best for debugging numeric behavior.
- `prefetch_depth` (default `3`): currently reserved (not hot-path effective in this implementation).
- `nvdec_session_pool_size` / `uniform_auto_ratio` / `decode_threads` / `decode_thread_type` / `reader_cache_size` / `nvdec_reuse_open_decoder`: same semantics as `DecodeConfig`.

### Environment Knobs

Only profiling remains environment-controlled:

- `CODEC_DECODE_PROFILE`: `1/true/on` enables decode stage timing logs.
- `CODEC_DECODE_PROFILE_VERBOSE`: `1/true/on` enables verbose per-stage profiling details.

All non-profile runtime controls are configured via `DecodeConfig` / `PatchStreamConfig`.

### Recommended Modes

- Lowest end-to-end latency (single request):
    1. `backend=gpu`, `decode_mode=latency`, modest `sequence_length`, modest `k_keep`.

- Highest sustained throughput:
    1. `backend=gpu`, `decode_mode=throughput`, tune `nvdec_session_pool_size` from `2` upward while monitoring memory.

- Static-scene video robustness:
    1. enable `static_fallback=True`, start from defaults and tune `static_abs_thresh`/`static_rel_thresh` conservatively.

## Benchmarks

From `codec-patch-stream/`:

```bash
# decode-only vs patch-stream (same initial sampled frames)
python benchmark/decode_vs_patch_stream.py ./assets/demo.mp4 \
  --gpu 0 --num-frames 16 --warmup 3 --runs 10 \
  --decode-mode throughput --uniform-strategy auto \
  --input-size 224 --patch 14 --topk 2048 --dtype bf16

# codec-patch-stream decode vs decord decode
python benchmark/compare_codec_vs_decord.py ./assets/demo.mp4 \
  --decord-gpu -1 --codec-gpu 0 --num-frames 32 \
  --warmup 5 --runs 20 --codec-mode throughput
```

See `benchmark/README.md` for details.

## Demo

```bash
python examples/demo_patch_stream.py ./assets/demo.mp4 \
  --frames 16 --input-size 224 --patch 14 --topk 1024 \
  --dtype bf16 --backend auto --out-dir patch_viz
```
