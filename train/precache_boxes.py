#!/usr/bin/env python
"""Pre-run PaddleOCR over training images and cache field boxes as .npy files.

Resumable (skips already-cached ids), multi-worker.

Usage:
    python precache_boxes.py --data-root /path/to/data --ann-dir annotations
    python precache_boxes.py --data-root /path/to/data --ann-dir annotations --include-clean --workers 4
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_OCR = None


def _init_worker():
    global _OCR
    os.environ.setdefault("OMP_NUM_THREADS", "3")
    from paddleocr import TextDetection
    _OCR = TextDetection()


def _result_to_boxes(r) -> np.ndarray:
    get = r.get if hasattr(r, "get") else (lambda k, d=None: r[k] if k in r else d)
    polys = get("dt_polys", None)
    out = []
    for b in (polys if polys is not None else []):
        b = np.asarray(b, np.float32)
        out.append([b[:, 0].min(), b[:, 1].min(), b[:, 0].max(), b[:, 1].max()])
    return np.asarray(out, np.float32) if out else np.zeros((0, 4), np.float32)


def _task(item):
    image_id, path, cache_dir = item
    cache = Path(cache_dir) / f"{image_id}.npy"
    if cache.exists():
        return "cached"
    try:
        img = cv2.imread(path)
        if img is None:
            np.save(cache, np.zeros((0, 4), np.float32))
            return "err"
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = _OCR.predict(rgb)
        r = res[0] if isinstance(res, (list, tuple)) else res
        np.save(cache, _result_to_boxes(r))
        return "ok"
    except Exception as exc:
        np.save(cache, np.zeros((0, 4), np.float32))
        return f"err:{type(exc).__name__}"


def build_items(data_root, ann_dir, include_clean):
    """Build list of (image_id, image_path) for training images."""
    data_root = Path(data_root)
    ann_dir = Path(ann_dir)
    train_dir = data_root / "train" / "train"
    if not train_dir.exists():
        train_dir = data_root / "train"

    rows = []
    sub_csv = ann_dir / "subtask_annotations.csv"
    if sub_csv.exists():
        sub = pd.read_csv(sub_csv).rename(columns={"details_tamper": "text_label"})
        sub = sub[sub["text_label"].notna()].copy()
        sub["text_label"] = sub["text_label"].astype(int)
        sub = sub.sort_values("text_label", ascending=False)  # fakes first
        for _, r in sub.iterrows():
            if include_clean or int(r["text_label"]) == 1:
                rows.append((r["id"], str(train_dir / f"{r['id']}.jpeg")))
    return [(i, p) for i, p in rows if Path(p).exists()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=str, required=True, help="Competition data root (contains train_labels.csv)")
    ap.add_argument("--ann-dir", type=str, default="annotations", help="Annotations dir (contains subtask_annotations.csv)")
    ap.add_argument("--cache-dir", type=str, default=None, help="Output cache dir (default: ann-dir/paddle_cache)")
    ap.add_argument("--include-clean", action="store_true", help="Also cache clean (label=0) documents")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.ann_dir) / "paddle_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    items = build_items(args.data_root, args.ann_dir, args.include_clean)
    todo = [(i, p, str(cache_dir)) for i, p in items if not (cache_dir / f"{i}.npy").exists()]
    print(f"queue={len(items)} cached={len(items)-len(todo)} todo={len(todo)} "
          f"workers={args.workers} cache={cache_dir}", flush=True)
    if not todo:
        print("nothing to do — all cached.", flush=True)
        return

    t0 = time.time()
    counts = {"ok": 0, "cached": 0, "err": 0}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        for n, status in enumerate(ex.map(_task, todo, chunksize=4), 1):
            key = "err" if status.startswith("err") else status
            counts[key] = counts.get(key, 0) + 1
            if n % 50 == 0 or n == len(todo):
                rate = n / max(time.time() - t0, 1e-6)
                eta = (len(todo) - n) / max(rate, 1e-6)
                print(f"[{n}/{len(todo)}] ok={counts['ok']} err={counts['err']} "
                      f"{rate:.1f} img/s eta={eta/60:.1f}min", flush=True)
    print(f"done in {(time.time() - t0)/60:.1f}min | {counts} | cache: {cache_dir}", flush=True)


if __name__ == "__main__":
    main()
