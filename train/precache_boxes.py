#!/usr/bin/env python
"""Pre-run PaddleOCR over the annotation/eval image queue and cache field boxes.

Fills artifacts/paddle_box_cache/<id>.npy (same cache the annotation server and the
notebook's Block 14 read), so the annotator never waits on OCR. Resumable (skips
already-cached ids), multi-worker, public-first ordering.

    python precache_paddle_boxes.py                 # all text-fakes (public first), 4 workers
    python precache_paddle_boxes.py --split public  # just public fakes
    python precache_paddle_boxes.py --include-clean # also all clean docs (for training negatives)
    python precache_paddle_boxes.py --workers 6
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DATA_ROOT = Path("the-freuid-challenge-2026-ijcai-ecai (1)")
TRAIN_DIR = DATA_ROOT / "train" / "train"
PUBLIC_DIR = DATA_ROOT / "public_test" / "public_test"
SUBTASK_CSV = Path("subtask_annotations.csv")
PUBLIC_MANUAL_CSV = Path("public_test_manual_labels.csv")
BOX_CACHE = Path("artifacts") / "paddle_box_cache"

_OCR = None


def _init_worker():
    global _OCR
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "3")
    from paddleocr import TextDetection
    _OCR = TextDetection()   # detection only — we only need field boxes, not recognized text


def _result_to_boxes(r) -> np.ndarray:
    get = r.get if hasattr(r, "get") else (lambda k, d=None: r[k] if k in r else d)
    polys = get("dt_polys", None)
    out = []
    for b in (polys if polys is not None else []):
        b = np.asarray(b, np.float32)
        out.append([b[:, 0].min(), b[:, 1].min(), b[:, 0].max(), b[:, 1].max()])
    return np.asarray(out, np.float32) if out else np.zeros((0, 4), np.float32)


def _task(item):
    image_id, path = item
    cache = BOX_CACHE / f"{image_id}.npy"
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
    except Exception as exc:  # cache empty so we don't spin on a bad file
        np.save(cache, np.zeros((0, 4), np.float32))
        return f"err:{type(exc).__name__}"


def build_items(split: str, include_clean: bool):
    """Public first (eval-relevant, small), then train. Fakes first within each split."""
    pub_rows, train_rows = [], []
    if split in ("public", "both"):
        pub = pd.read_csv(PUBLIC_MANUAL_CSV)
        pub = pub[pub["text_label"].notna()].copy()
        pub["text_label"] = pub["text_label"].astype(int)
        pub = pub.sort_values("text_label", ascending=False)   # fakes first
        for _, r in pub.iterrows():
            if include_clean or int(r["text_label"]) == 1:
                pub_rows.append((r["id"], str(PUBLIC_DIR / f"{r['id']}.jpeg")))
    if split in ("train", "both"):
        sub = pd.read_csv(SUBTASK_CSV).rename(columns={"details_tamper": "text_label"})
        sub = sub[sub["text_label"].notna()].copy()
        sub["text_label"] = sub["text_label"].astype(int)
        sub = sub.sort_values("text_label", ascending=False)   # fakes first
        for _, r in sub.iterrows():
            if include_clean or int(r["text_label"]) == 1:
                train_rows.append((r["id"], str(TRAIN_DIR / f"{r['id']}.jpeg")))
    items = pub_rows + train_rows
    return [(i, p) for i, p in items if Path(p).exists()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "public", "both"], default="both")
    ap.add_argument("--include-clean", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    BOX_CACHE.mkdir(parents=True, exist_ok=True)
    items = build_items(args.split, args.include_clean)
    todo = [it for it in items if not (BOX_CACHE / f"{it[0]}.npy").exists()]
    print(f"queue={len(items)} already_cached={len(items) - len(todo)} todo={len(todo)} "
          f"workers={args.workers}", flush=True)
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
    print(f"done in {(time.time() - t0)/60:.1f}min | {counts} | cache dir: {BOX_CACHE}", flush=True)


if __name__ == "__main__":
    main()
