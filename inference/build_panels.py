#!/usr/bin/env python3
"""Process 1: Build forensic panels from images + cached boxes → /dev/shm/panels/

Runs independently from GPU scoring. Writes uint8 panels as .npy files.
GPU scorer (run_inference.py --steps text-score) reads and deletes them.

Usage:
    python build_panels.py --image-dir /path/to/images --box-cache ./paddle_cache --out /dev/shm/panels --workers 16
"""
import argparse, os, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

# Add parent paths for imports
_ROOT = str(Path(__file__).resolve().parent.parent)
_TRAIN = str(Path(__file__).resolve().parent.parent / "train")
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
if _TRAIN not in sys.path: sys.path.insert(0, _TRAIN)

from scripts.forensic_panels import forensic_panel_from_crop, PANEL_H, PANEL_W

# Must match inference constants
TOP_OCR_IGNORE_FRAC = 0.17
PAD_X, PAD_Y = 15, 5
MIN_W, MIN_H = 10, 8
MAX_FIELDS = 40


def filter_boxes(boxes, H, W, min_aspect=0.5, max_w_frac=0.85, max_h_frac=0.09):
    kept = []
    for x0, y0, x1, y1 in boxes:
        bw, bh = x1 - x0, y1 - y0
        if y0 < TOP_OCR_IGNORE_FRAC * H: continue
        if bw < MIN_W or bh < MIN_H: continue
        if bh > 0 and bw / bh < min_aspect: continue
        if bw > max_w_frac * W or bh > max_h_frac * H: continue
        kept.append([x0, y0, x1, y1])
    return np.array(kept[:MAX_FIELDS], np.float32) if kept else np.zeros((0, 4), np.float32)


def build_one(args):
    image_id, image_path, cache_dir, out_dir = args
    cv2.setNumThreads(1)

    # Load boxes from cache
    npy = cache_dir / f"{image_id}.npy"
    sz = cache_dir / f"{image_id}_size.npy"
    if not npy.exists():
        return 0
    raw = np.load(npy, allow_pickle=False)
    if sz.exists():
        s = np.load(sz)
        boxes = filter_boxes(raw, int(s[0]), int(s[1]))
    else:
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            return 0
        boxes = filter_boxes(raw, bgr.shape[0], bgr.shape[1])

    if len(boxes) == 0:
        return 0

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return 0
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    count = 0
    for idx, (x0, y0, x1, y1) in enumerate(boxes):
        cx0, cy0 = max(0, int(x0) - PAD_X), max(0, int(y0) - PAD_Y)
        cx1, cy1 = min(W, int(x1) + PAD_X), min(H, int(y1) + PAD_Y)
        crop = rgb[cy0:cy1, cx0:cx1]
        if crop.shape[0] >= 6 and crop.shape[1] >= 6:
            panel = forensic_panel_from_crop(crop)
            if panel.shape[:2] != (PANEL_H, PANEL_W):
                panel = cv2.resize(panel, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
            np.save(out_dir / f"{image_id}_{idx}.npy", panel)
            count += 1
    del bgr, rgb
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--box-cache", required=True)
    ap.add_argument("--out", default="/dev/shm/panels")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    image_dir = Path(args.image_dir)
    cache_dir = Path(args.box_cache)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write sentinel so scorer knows we're running
    (out_dir / "_BUILDING").touch()

    # Discover images
    exts = {'.jpeg', '.jpg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
    image_rows = [(f.stem, f) for f in sorted(image_dir.iterdir()) if f.suffix.lower() in exts]
    print(f"Found {len(image_rows)} images", flush=True)

    work = [(iid, str(ip), cache_dir, out_dir) for iid, ip in image_rows]

    t0 = time.time()
    total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for count in tqdm(pool.map(build_one, work), total=len(work), desc="build panels"):
            total += count

    elapsed = time.time() - t0
    print(f"Built {total} panels in {elapsed:.0f}s ({total/max(elapsed,1):.0f} panels/s)")

    # Remove sentinel — scorer knows we're done
    (out_dir / "_BUILDING").unlink(missing_ok=True)
    (out_dir / "_DONE").touch()


if __name__ == "__main__":
    main()
