"""Generate visualization videos for the scene-augmentation reproduction.

A: out/scene_aug_demo.mp4   original | scene-aug, sweeping 12 random seeds (each held 6 frames)
B: out/branch_cycle.mp4     original -> clutter -> texture -> scene, 4 branches cycling
Frames are rendered with PIL, encoded with system ffmpeg (H.264 + gif preview).
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import demo as D  # noqa: E402
from scene_aug_numpy import HEAD_CAMERA, SceneLibrary, ScenePlan  # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
FFMPEG = shutil.which("ffmpeg")


def encode(frames_dir, pattern, out_mp4, fps=15):
    if not FFMPEG:
        print("ffmpeg not found, skip", out_mp4)
        return
    subprocess.run([FFMPEG, "-y", "-framerate", str(fps), "-i", str(frames_dir / pattern),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out_mp4)],
                   check=True, capture_output=True)
    print("encoded", out_mp4, f"{out_mp4.stat().st_size/1024:.1f} KB")


def main():
    lib = SceneLibrary(HERE / "synthetic_library.npz")
    task = "microduck_place_a"
    baseline = D.paste_object(lib.median_head.astype(np.uint8), lib, 0, 150, 110)
    cand = lib.candidates(task)[:2].tolist()
    fg = lib.foreground(task, 0)

    # --- A: seed sweep, original | scene ---
    fa = HERE / "frames_a"
    fa.mkdir(exist_ok=True)
    n = 0
    for i in range(12):
        seed = 1000 + i * 37
        aug = ScenePlan(seed, True, True, cand, fg, kinds=7, feather=2.0).apply(baseline, HEAD_CAMERA, lib)
        left = D.label(Image.fromarray(baseline.transpose(1, 2, 0)), "original")
        right = D.label(Image.fromarray(aug.transpose(1, 2, 0)), f"scene aug | seed={seed}")
        canvas = Image.new("RGB", (512, 256), (255, 255, 255))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (256, 0))
        for _ in range(6):
            canvas.save(fa / f"frame_{n:04d}.png")
            n += 1
    encode(fa, "frame_%04d.png", OUT / "scene_aug_demo.mp4")

    # --- B: 4-branch cycle ---
    fb = HERE / "frames_b"
    fb.mkdir(exist_ok=True)
    branches = [("original", False, False), ("clutter", True, False),
                ("texture", False, True), ("scene", True, True)]
    n = 0
    for name, clutter, texture in branches:
        img = ScenePlan(2026, clutter, texture, cand, fg, kinds=7, feather=2.0).apply(baseline, HEAD_CAMERA, lib)
        pane = D.label(Image.fromarray(img.transpose(1, 2, 0)), name)
        for _ in range(20):
            pane.save(fb / f"frame_{n:04d}.png")
            n += 1
    encode(fb, "frame_%04d.png", OUT / "branch_cycle.mp4")

    print("done. frames:", len(list(fa.glob('*.png'))), "/", len(list(fb.glob('*.png'))))


if __name__ == "__main__":
    main()
