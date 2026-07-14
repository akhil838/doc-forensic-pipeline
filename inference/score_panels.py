#!/usr/bin/env python3
"""Process 2: GPU-score forensic panels from /dev/shm/panels/ → text_scores.csv

Reads .npy panel files written by build_panels.py, scores with ConvNeXt-B,
writes results, deletes scored panels to free ramdisk.

Usage:
    python score_panels.py --panel-dir /dev/shm/panels --model-dir ./models/weights --output submission.csv --device cuda
"""
import argparse, os, sys, time, glob
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
from tqdm import tqdm

_ROOT = str(Path(__file__).resolve().parent.parent)
_TRAIN = str(Path(__file__).resolve().parent.parent / "train")
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
if _TRAIN not in sys.path: sys.path.insert(0, _TRAIN)

from models.models import DINOv3Classifier

PANEL_H, PANEL_W = 224, 1008


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-dir", default="/dev/shm/panels")
    ap.add_argument("--model-dir", default="./models/weights")
    ap.add_argument("--output", default="text_scores.csv")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--agg", default="max", choices=["max", "top3", "mean"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--poll-interval", type=float, default=0.5, help="Seconds between polling for new panels")
    args = ap.parse_args()

    DEVICE = args.device
    USE_FP16 = (DEVICE == "cuda") and not args.no_fp16
    panel_dir = Path(args.panel_dir)

    # Load model
    weights = Path(args.model_dir) / "dinov3_convnext_base_tamper_clf.pt"
    model = DINOv3Classifier()
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=False)
    model = model.to(DEVICE).eval()
    if DEVICE == "cuda":
        model = torch.compile(model, mode='reduce-overhead')
        autocast = torch.amp.autocast('cuda', dtype=torch.float16) if USE_FP16 else torch.amp.autocast('cuda', enabled=False)
        with torch.no_grad(), autocast:
            _ = model(torch.randn(args.batch_size, 3, PANEL_H, PANEL_W, device=DEVICE))
            _ = model(torch.randn(1, 3, PANEL_H, PANEL_W, device=DEVICE))
        print(f"torch.compile warmup done (bs={args.batch_size})", flush=True)
    else:
        autocast = torch.amp.autocast(DEVICE, enabled=False)

    print(f"Model loaded: {weights}", flush=True)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    # Output CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_f = open(out_path, 'w')
    csv_f.write("id,text_prob,n_fields\n")

    doc_probs = defaultdict(list)
    total_panels = 0
    total_docs = 0
    t0 = time.time()
    scored_files = set()

    pbar = tqdm(desc="score panels", unit=" panels")

    def _score_batch(files):
        nonlocal total_panels
        panels = []
        ids = []
        for f in files:
            try:
                panel = np.load(f, allow_pickle=False)
                # Parse image_id from filename: {image_id}_{field_idx}.npy
                fname = Path(f).stem
                image_id = fname.rsplit('_', 1)[0]
                panels.append(panel)
                ids.append(image_id)
            except Exception:
                continue
        if not panels:
            return
        arr = np.stack(panels).astype(np.float32) / 255.0
        arr = (arr - mean) / std
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).to(DEVICE)
        with torch.no_grad(), autocast:
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
        for doc_id, prob in zip(ids, probs):
            doc_probs[doc_id].append(prob)
        total_panels += len(panels)
        pbar.update(len(panels))
        # Delete scored files
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass

    def _flush_docs():
        nonlocal total_docs
        for did in list(doc_probs.keys()):
            probs = doc_probs.pop(did)
            probs.sort(reverse=True)
            if args.agg == 'max':
                score = probs[0]
            elif args.agg == 'top3':
                score = float(np.mean(probs[:min(3, len(probs))]))
            else:
                score = float(np.mean(probs))
            csv_f.write(f"{did},{score},{len(probs)}\n")
            total_docs += 1
        csv_f.flush()

    # Poll loop — keep scoring until builder signals done
    print("Waiting for panels...", flush=True)
    while True:
        files = sorted(glob.glob(str(panel_dir / "*.npy")))
        new_files = [f for f in files if f not in scored_files]

        if new_files:
            # Score in batches
            for i in range(0, len(new_files), args.batch_size):
                batch_files = new_files[i:i + args.batch_size]
                _score_batch(batch_files)
                scored_files.update(batch_files)

            if total_panels % 50000 < args.batch_size:
                _flush_docs()
                elapsed = time.time() - t0
                print(f"  [{total_docs} docs, {total_panels} panels, "
                      f"{total_panels/max(elapsed,1):.0f} panels/s]", flush=True)
        else:
            # Check if builder is done
            if (panel_dir / "_DONE").exists():
                # Score any remaining
                files = sorted(glob.glob(str(panel_dir / "*.npy")))
                for i in range(0, len(files), args.batch_size):
                    _score_batch(files[i:i + args.batch_size])
                _flush_docs()
                break
            time.sleep(args.poll_interval)

    pbar.close()
    csv_f.close()

    elapsed = time.time() - t0
    print(f"\nDone: {total_panels} panels, {total_docs} docs in {elapsed:.0f}s "
          f"({total_panels/max(elapsed,1):.0f} panels/s)")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
