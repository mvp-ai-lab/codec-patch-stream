from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from codec_patch_stream import CodecPatchStream
from torchvision.utils import save_image


def main() -> None:
    p = argparse.ArgumentParser(description="CodecPatchStream visualization demo")
    p.add_argument("video")
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--min-pixels", type=int, default=None)
    p.add_argument("--max-pixels", type=int, default=None)
    p.add_argument("--patch", type=int, default=14)
    p.add_argument("--topk", type=int, default=512)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32", "bfloat16", "float16", "float32"])
    p.add_argument("--backend", choices=["auto", "gpu", "cpu"], default="auto")
    p.add_argument("--out-dir", default="patch_viz")
    args = p.parse_args()

    s = CodecPatchStream(
        video_path=args.video,
        sequence_length=args.frames,
        input_size=args.input_size,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        patch_size=args.patch,
        k_keep=args.topk,
        output_dtype=args.dtype,
        backend=args.backend,
        selection_unit="block2x2",
        static_fallback=False,
        energy_pct=95.0,
        prefetch_depth=3,
    )

    print(f"Total selected patches: {len(s)}")
    patches = s.patches
    metas = s.metadata
    print(f"Patch bank shape: {tuple(patches.shape)} dtype={patches.dtype} device={patches.device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if metas:
        canvas_h = (max(int(meta["patch_h_idx"]) for meta in metas) + 1) * args.patch
        canvas_w = (max(int(meta["patch_w_idx"]) for meta in metas) + 1) * args.patch
    else:
        canvas_h = args.patch
        canvas_w = args.patch
    print(f"Smart resize target: {canvas_h}x{canvas_w}")

    # Prepare black canvases for all sampled frames.
    canvases = torch.zeros((args.frames, 3, canvas_h, canvas_w), dtype=torch.float32)
    frame_id_by_seq_pos: dict[int, int] = {}
    selected_count_by_seq_pos: dict[int, int] = defaultdict(int)

    for i, meta in enumerate(metas):
        seq_pos = int(meta["seq_pos"])
        ph_idx = int(meta["patch_h_idx"])
        pw_idx = int(meta["patch_w_idx"])
        frame_id = int(meta["frame_id"])

        frame_id_by_seq_pos[seq_pos] = frame_id
        selected_count_by_seq_pos[seq_pos] += 1

        y0 = ph_idx * args.patch
        x0 = pw_idx * args.patch
        y1 = y0 + args.patch
        x1 = x0 + args.patch

        # Stream patch values are in 0~255 float/bfloat16; normalize to 0~1 for save_image.
        patch = patches[i].to(torch.float32).detach().cpu() / 255.0
        patch = patch.clamp_(0.0, 1.0)
        canvases[seq_pos, :, y0:y1, x0:x1] = patch

    for seq_pos in range(args.frames):
        frame_id = frame_id_by_seq_pos.get(seq_pos, -1)
        out_path = out_dir / f"seq_{seq_pos:03d}.png"
        save_image(canvases[seq_pos], str(out_path))
        print(
            f"Saved {out_path} | selected_patches={selected_count_by_seq_pos.get(seq_pos, 0)}"
        )

    print(f"Visualization completed: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
