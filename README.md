# Video Patch Streaming

`CodecPatchStream` exposes a native C++ video-to-patch iterator with CPU/GPU backends:

- input: H.264/H.265 video path
- output: iterable stream of selected ViT patches on the selected device
- GPU pipeline: NVDEC decode -> GPU block-matching MV proxy + compensation residual proxy (estimated) -> fused energy -> static fallback (optional) -> global Top-K (I-frame first) -> patch extract
- CPU pipeline: FFmpeg software decode -> CPU block-matching MV proxy + compensation residual proxy -> fused energy -> static fallback (optional) -> global Top-K (I-frame first) -> patch extract

> NOTE: As NVDEC doesn't expose motion vectors or residuals, we use block-matching on the decoded frames as a estimated proxy, which is much faster than optical flow and can be done on the GPU.

## Preview

<div style="display: flex; gap: 10px;">
    <img src="./assets/demo.gif" style="height: 300px; width: 300px;" />
    <img src="./assets/output.gif" style="height: 300px; width: 300px;" />
</div>

## Build prerequisites

- FFmpeg dev headers/libraries:
  - `libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev`
- Optional GPU prerequisites:
  - NVIDIA GPU + CUDA toolkit (`nvcc`)
  - NVCodec headers (e.g. `nv-codec-headers` -> `/usr/local/include/ffnvcodec`)

## Install with uv

```bash
uv venv .venv

# torch runtime
uv pip install --python .venv/bin/python torch

# build CPU backend only (works without nvcc/GPU)
CODEC_BUILD_NATIVE=1 CODEC_ENABLE_GPU=0 uv pip install -e . --python .venv/bin/python --no-build-isolation

# build CPU + GPU backend (if nvcc is available; default CODEC_ENABLE_GPU=auto)
CODEC_BUILD_NATIVE=1 uv pip install -e . --python .venv/bin/python --no-build-isolation
```

## Quick usage

```python
from codec_patch_stream import CodecPatchStream

stream = CodecPatchStream(
    video_path="/path/to/video.mp4",
    sequence_length=16,
    input_size=224,
    patch_size=14,
    k_keep=512,
    static_fallback=True,   # optional static-scene fallback
    static_abs_thresh=2.0,  # auto-compatible: <=1 means [0,1], >1 means [0,255]
    static_rel_thresh=0.15,
    static_uniform_frames=4,
    # global top-k with I-frame priority
    output_dtype="bf16",
    backend="auto",         # auto | gpu | cpu
)

for patch, meta in stream:
    # patch: (3, 14, 14) bf16 tensor on selected backend device
    # meta: seq_pos/frame_id/is_i/patch idx/score
    pass

# Metadata tensor fast path
patches, meta_fields, meta_scores = stream.next_n_tensors(256)
# meta_fields: int64 tensor (N, 6) on selected backend device
# columns: [seq_pos, frame_id, is_i, patch_linear_idx, patch_h_idx, patch_w_idx]
# meta_scores: float32 tensor (N,)

# full metadata tensors for all selected patches
all_meta_fields, all_meta_scores = stream.metadata_tensors
```

`static_fallback` only adjusts P-frame selection under the remaining global budget after I-frame priority.

```bash
python examples/demo_patch_stream.py ./assets/demo.mp4 \
    --frames 16 --input-size 224 --patch 14 --topk 1024 --dtype bf16 \
    --backend auto --out-dir patch_viz
```

## Benchmark

```bash
python examples/benchmark.py ./assets/demo.mp4 \
    --frames 16 --input-size 224 --patch 14 --topk 1024 --dtype bf16 \
    --backend auto --prepare-repeats 5 --prepare-warmup 1
```
