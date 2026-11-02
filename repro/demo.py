"""Reproduce + optimize LingBot-VLA2 scene augmentation on a self-contained synthetic scene,
then apply it to the microduck ('鸭子') render as a real-asset demo.

Run:  python demo.py
Out:  out/synth_compare.png  original / clutter / texture / scene
      out/synth_opt.png      scene: baseline vs optimized (kinds=7, feather=2)
      out/duck_compare.png   microduck render: baseline vs scene
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from scene_aug_numpy import (  # noqa: E402
    HEAD_CAMERA, SceneLibrary, ScenePlan, generate_texture,
)

W = H = 256
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
FONT_PATH = "C:/Windows/Fonts/arial.ttf"


def _font(size=16):
    for p in ("C:/Windows/Fonts/arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def build_library(path):
    """Synthetic 3-task library: grey wall + white table + 3 coloured object crops."""
    median = np.full((3, H, W), 247, np.uint8)
    median[:, : H // 4] = np.array([224, 208, 208], np.uint8)[:, None, None]  # wall strip

    specs = [  # (y, x, h, w, rgb, shape)
        (40, 36, 40, 40, (198, 62, 58), "box"),
        (74, 64, 44, 40, (58, 132, 198), "disc"),
        (64, 92, 50, 36, (72, 168, 92), "box"),
    ]
    meta, offsets, rgbs, bodies, shades = [], [0], [], [], []
    for task_id in range(3):
        for episode in range(2):
            for (y, x, h, w, color, shape) in specs:
                body = np.zeros((h, w), bool)
                yy, xx = np.mgrid[0:h, 0:w]
                if shape == "disc":
                    r = min(h, w) / 2 - 3
                    body = ((yy - h / 2) ** 2 + (xx - w / 2) ** 2) < r * r
                else:
                    body[3:-3, 3:-3] = True
                shade = np.full((h, w), 255, np.uint8)
                shade = np.where(~body & (np.abs(yy - h / 2) < h / 2) & (np.abs(xx - w / 2) < w / 2),
                                 (150 * np.ones_like(shade)).astype(np.uint8), shade)
                rgb = np.zeros((3, h, w), np.uint8)
                rgb[:, body] = np.array(color, np.uint8)[:, None]
                rgb[:, ~body] = 247
                meta.append([task_id, episode, y, x, h, w])
                rgbs.append(rgb.reshape(3, -1).T)
                bodies.append(body.reshape(-1))
                shades.append(shade.reshape(-1))
                offsets.append(offsets[-1] + h * w)

    tasks = np.array(["microduck_place_a", "bottle_pick_b", "blocks_stack_c"])
    fg = np.zeros((3, 2, H * W), bool)
    fg_packed = np.packbits(fg.reshape(3, 2, -1), axis=2)
    np.savez(path, tasks=tasks, meta=np.array(meta, np.int32), offsets=np.array(offsets, np.int64),
             rgb=np.concatenate(rgbs).astype(np.uint8), body=np.concatenate(bodies).astype(bool),
             shade=np.concatenate(shades).astype(np.uint8), episode_foreground=fg_packed,
             median_cam_high=median, median_cam_left_wrist=median, median_cam_right_wrist=median)
    return path


def paste_object(frame, library, index, y, x):
    """Put one existing object into the baseline scene (simulates a scene already containing it)."""
    _, _, rgb, body, shade = library.crop(index)
    h, w = body.shape
    out = frame.astype(np.float32)
    win = (slice(y, y + h), slice(x, x + w))
    out[:, win[0], win[1]][:, body] = rgb[:, body]
    out[:, win[0], win[1]][:, body] *= shade[body]
    return np.clip(out.round(), 0, 255).astype(np.uint8)


def label(img, text):
    img = img.copy()
    d = ImageDraw.Draw(img)
    f = _font(16)
    tw = d.textlength(text, font=f)
    d.rectangle([0, 0, tw + 12, 24], fill=(0, 0, 0))
    d.text((6, 3), text, fill=(255, 255, 255), font=f)
    return img


def hstack(images, path):
    w = sum(i.width for i in images)
    h = max(i.height for i in images)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.width
    canvas.save(path)
    print("saved", path)


def run_branches(lib, frame, task, tag, kinds=5, feather=0.0):
    cand = lib.candidates(task)
    picks = cand[:2].tolist() if len(cand) >= 2 else list(range(2))
    fg = lib.foreground(task, 0)
    results = {}
    for name, clutter, texture in [("original", False, False), ("clutter", True, False),
                                   ("texture", False, True), ("scene", True, True)]:
        plan = ScenePlan(20261102, clutter, texture, picks, fg, kinds=kinds, feather=feather)
        t0 = time.time()
        out = plan.apply(frame, HEAD_CAMERA, lib)
        results[name] = (out, time.time() - t0, plan.placements)
    return results, picks


def main():
    lib_path = HERE / "synthetic_library.npz"
    build_library(lib_path)
    lib = SceneLibrary(lib_path)
    task = "microduck_place_a"

    # baseline scene: table + one existing object
    baseline = paste_object(lib.median_head.astype(np.uint8), lib, 0, 150, 110)

    # --- reproduction (original settings) ---
    res, picks = run_branches(lib, baseline, task, "orig")
    hstack([label(Image.fromarray(np.transpose(res[k][0], (1, 2, 0))), k) for k in
            ("original", "clutter", "texture", "scene")], OUT / "synth_compare.png")
    print("clutter placements:", res["scene"][2])
    for k, (img, dt, _) in res.items():
        changed = np.any(img != baseline, axis=0).mean()
        print(f"  {k:9s} {dt*1000:7.1f} ms  changed_px={changed*100:5.1f}%")

    # --- optimized settings: 7 texture kinds + 2px feather ---
    res_opt, _ = run_branches(lib, baseline, task, "opt", kinds=7, feather=2.0)
    hstack([label(Image.fromarray(np.transpose(res["scene"][0], (1, 2, 0))), "scene (repro)"),
            label(Image.fromarray(np.transpose(res_opt["scene"][0], (1, 2, 0))), "scene (opt: 7k+feather)")],
           OUT / "synth_opt.png")

    # --- microduck real-asset demo ---
    duck_src = Path(os.environ.get("DUCK_IMG", "E:/师兄的项目/_microduck/out/view_45.png"))
    if duck_src.exists():
        duck = Image.open(duck_src).convert("RGB").resize((110, 110), Image.LANCZOS)
        canvas = lib.median_head.astype(np.uint8).copy()
        c = np.array(duck)
        yy, xx = 120, 80
        white = c.min(axis=2) > 238  # drop the render's white backdrop so it sits on the table
        dst = canvas[:, yy:yy + 110, xx:xx + 110]
        dst[:, ~white] = c.transpose(2, 0, 1)[:, ~white]
        duck_base = canvas
        r_duck, _ = run_branches(lib, duck_base, task, "duck", kinds=7, feather=2.0)
        hstack([label(Image.fromarray(np.transpose(duck_base, (1, 2, 0))), "microduck (baseline)"),
                label(Image.fromarray(np.transpose(r_duck["scene"][0], (1, 2, 0))), "microduck + scene aug")],
               OUT / "duck_compare.png")

    # determinism check
    a = ScenePlan(7, True, True, picks, None).apply(baseline, HEAD_CAMERA, lib)
    b = ScenePlan(7, True, True, picks, None).apply(baseline, HEAD_CAMERA, lib)
    print("deterministic:", np.array_equal(a, b), "| dtype:", a.dtype, "| shape:", a.shape)


if __name__ == "__main__":
    main()
