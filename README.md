# Codec Patch Stream

## Breaking Change (v0.4.x)

`backend/device_id` has been split into two stages:

- decode stage: `decode_backend`, `decode_device_id`
- process stage (patch pipeline only): `process_backend`, `process_device_id`

This is a breaking API upgrade. Old fields are removed (no compatibility shim).

`auto` resolution rules:

- `decode_backend=auto`: uses `gpu` if CUDA is available, else `cpu`
- `process_backend=auto`: follows resolved `decode_backend`

Output device rule:

- `process_backend=cpu`: `patches` and `metadata_tensors` are CPU tensors
- `process_backend=gpu`: `patches` and `metadata_tensors` are on `cuda:process_device_id`

---

`codec-patch-stream` provides two native APIs:

- `decode_only(DecodeConfig)`: uniform sampled decode output
- `patch_stream(PatchStreamConfig)`: decode + energy-based patch selection

## Install

```bash
uv venv .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python torch
```

CPU-only build:

```bash
CODEC_BUILD_NATIVE=1 CODEC_ENABLE_GPU=0 \
uv pip install -e . --python ./.venv/bin/python --no-build-isolation
```

CPU+GPU build:

```bash
CODEC_BUILD_NATIVE=1 CODEC_ENABLE_GPU=1 \
uv pip install -e . --python ./.venv/bin/python --no-build-isolation
```

## API Usage

### `patch_stream`

```python
from codec_patch_stream import PatchStreamConfig, patch_stream

session = patch_stream(
    PatchStreamConfig(
        video_path="/path/to/video.mp4",
        sequence_length=16,
        decode_backend="cpu",   # auto|cpu|gpu
        process_backend="gpu",  # auto|cpu|gpu
        decode_device_id=0,
        process_device_id=0,
        uniform_strategy="auto",
        input_size=224,
        patch_size=14,
        k_keep=2048,
        output_dtype="bf16",
    )
)
patches = session.patches
meta_fields, meta_scores = session.metadata_tensors
session.close()
```

Supported combinations (GPU build):

1. `cpu -> cpu`
2. `cpu -> gpu`
3. `gpu -> cpu`
4. `gpu -> gpu`

### `decode_only`

```python
from codec_patch_stream import DecodeConfig, decode_only

decoded = decode_only(
    DecodeConfig(
        video_path="/path/to/video.mp4",
        sequence_length=16,
        decode_backend="auto",  # auto|cpu|gpu
        decode_device_id=0,
        uniform_strategy="auto",
    )
)
frames = decoded.frames
```

## Migration

1. Old:
`DecodeConfig(backend=..., device_id=...)`

New:
`DecodeConfig(decode_backend=..., decode_device_id=...)`

2. Old:
`PatchStreamConfig(backend=..., device_id=...)`

New:
`PatchStreamConfig(decode_backend=..., process_backend=..., decode_device_id=..., process_device_id=...)`

## Combination Suggestions

| Decode -> Process | Recommended when |
| --- | --- |
| `cpu -> cpu` | no GPU runtime or lightweight workloads |
| `cpu -> gpu` | CPU decode stronger but patch compute is heavy |
| `gpu -> cpu` | GPU decode preferred but CPU-side downstream expected |
| `gpu -> gpu` | high-throughput fully GPU pipeline |

## Benchmark

```bash
python benchmark/benchmark_codec_apis.py /path/to/videos \
  --apis decode_only patch_stream decord \
  --decode-backends cpu gpu \
  --process-backends cpu gpu \
  --limit 10
```
