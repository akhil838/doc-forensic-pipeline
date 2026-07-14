#!/usr/bin/env python3
"""
FREUID Challenge 2026 — full inference pipeline.

Usage:
  python run_inference.py --image-dir /data --model-dir ../models --output submission.csv

Steps:
  1. Discover all images in --image-dir
  2. MediaPipe face detection → portrait crop → DINOv2-small face classifier → photo_prob
  3. PaddleOCR TextDetection (GPU) → field boxes (with size/position filtering)
  4. Field crops → forensic panels → ConvNeXt-B classifier (FP16) → text_prob (top-3 agg)
  5. fraud_score = max(photo_prob, text_prob)
  6. Write submission.csv (id, label)

Timings on A100 (142k docs):
  MediaPipe face detect:  2ms/image  → 0.08h  (CPU)
  DINOv2-small face clf:  1.2ms/face → 0.05h  (GPU, FP16)
  PaddleOCR TextDetection: 35ms/image → 1.4h  (GPU)
  ConvNeXt-B text scorer: 1.5ms/field → 1.2h  (GPU, FP16, ~20 fields/doc)
  Total: ~2.7h
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import shared modules
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_TRAIN = str(Path(_ROOT) / "train")
if _TRAIN not in sys.path:
    sys.path.insert(0, _TRAIN)
from models.models import DINOv3Classifier, DINOClassifier
from scripts.forensic_panels import forensic_panel_from_crop

# ── Config ───────────────────────────────────────────────────────────────────

SEED = 42
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Face portrait crop constants (from unified_opencv_face_photo_model.ipynb)
STANDARD_PHOTO_SIZE = (256, 320)  # (width, height)
MODEL_IMAGE_SIZE = 224
PHOTO_ASPECT = STANDARD_PHOTO_SIZE[0] / STANDARD_PHOTO_SIZE[1]

# Text field constants (from field_annotation_server.py — must match training)
TOP_OCR_IGNORE_FRAC = 0.17  # skip top 17% of image (header/title band)
PAD_X, PAD_Y = 15, 5
MIN_W, MIN_H = 10, 8
MAX_FIELDS = 40

# Panel constants
PANEL_H, PANEL_W = 224, 1008
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
FACE_NORM = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
USE_FP16 = DEVICE == "cuda"  # FP16 autocast on CUDA only (MPS doesn't support it)


def _autocast():
    """Device-agnostic autocast context manager."""
    if DEVICE == "cuda" and USE_FP16:
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return torch.amp.autocast(device_type=DEVICE, enabled=False)  # no-op on MPS/CPU


def _empty_cache():
    """Free GPU memory on any device."""
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()


# ════════════════════════════════════════════════════════════════════════════
# STEP 1: Image discovery
# ════════════════════════════════════════════════════════════════════════════

def discover_images(image_dir: Path) -> list[tuple[str, Path]]:
    pairs = []
    for p in sorted(image_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            pairs.append((p.stem, p))
    if not pairs:
        raise FileNotFoundError(f"No images in {image_dir}")
    return pairs


# ════════════════════════════════════════════════════════════════════════════
# STEP 2: MediaPipe face detection → portrait crop → DINOv2-small classify
# MediaPipe BlazeFace: 2ms/image (CPU), then DINOv2: 1.2ms/face (GPU FP16)
# ════════════════════════════════════════════════════════════════════════════

# --- MediaPipe face detection (from kaggle_pipeline_benchmark.py) ---

def _clamp_aspect_box(x0, y0, x1, y1, W, H, aspect=PHOTO_ASPECT):
    """Clamp box to image bounds while preserving aspect ratio.
    From unified_opencv_face_photo_model.ipynb."""
    x0, y0, x1, y1 = map(float, (x0, y0, x1, y1))
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    current = bw / bh
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if current < aspect:
        bw = bh * aspect
    else:
        bh = bw / aspect
    x0, x1 = cx - bw / 2, cx + bw / 2
    y0, y1 = cy - bh / 2, cy + bh / 2
    if x0 < 0: x1 -= x0; x0 = 0
    if y0 < 0: y1 -= y0; y0 = 0
    if x1 > W: x0 -= (x1 - W); x1 = W
    if y1 > H: y0 -= (y1 - H); y1 = H
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


def generic_portrait_fallback_box(image_shape):
    """Generic left-side portrait crop when no face detected.
    From unified_opencv_face_photo_model.ipynb."""
    H, W = image_shape[:2]
    return _clamp_aspect_box(0.02 * W, 0.16 * H, 0.38 * W, 0.92 * H, W, H)


def regularize_photo_box(box, image_shape, target_height_frac=0.58,
                          min_height_frac=0.42, max_height_frac=0.72, snap=16):
    """Snap portrait box to stable document fraction.
    From unified_opencv_face_photo_model.ipynb."""
    H, W = image_shape[:2]
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    raw_h = max(1, y1 - y0)
    target_h = max(raw_h, target_height_frac * H)
    target_h = min(max(target_h, min_height_frac * H), max_height_frac * H)
    if snap:
        target_h = max(snap, round(target_h / snap) * snap)
    target_w = target_h * PHOTO_ASPECT
    return _clamp_aspect_box(cx - target_w / 2, cy - target_h / 2,
                             cx + target_w / 2, cy + target_h / 2, W, H)


def _init_mediapipe(model_dir: Path):
    """Initialize MediaPipe BlazeFace detector."""
    import mediapipe as mp
    model_path = model_dir / "blaze_face.tflite"
    if not model_path.exists():
        raise FileNotFoundError(f"MediaPipe model not found: {model_path}")
    det = mp.tasks.vision.FaceDetector.create_from_options(
        mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            min_detection_confidence=0.3))
    return det, mp


def mediapipe_detect_face(det, mp_mod, rgb: np.ndarray):
    """Detect largest face with MediaPipe → (x0,y0,x1,y1,source) or None."""
    mp_image = mp_mod.Image(image_format=mp_mod.ImageFormat.SRGB, data=rgb)
    result = det.detect(mp_image)
    if not result.detections:
        return None
    H, W = rgb.shape[:2]
    best = None
    best_area = 0
    for d in result.detections:
        bb = d.bounding_box
        x0, y0, w, h = bb.origin_x, bb.origin_y, bb.width, bb.height
        area = w * h
        # Same filtering as unified notebook: reject tiny/huge, prefer left-side face
        if area < 0.002 * W * H or area > 0.20 * W * H:
            continue
        cx = x0 + w / 2
        is_primary = cx <= 0.48 * W
        if best is None or (is_primary and not best[5]) or (is_primary == best[5] and area > best_area):
            best = (int(x0), int(y0), int(w), int(h), 'mediapipe_blaze', is_primary)
            best_area = area
    if best is None:
        return None
    return best[:5]


def face_to_photo_box(face_box, image_shape, portrait_scale=2.15):
    """Expand face box to portrait frame.
    From unified_opencv_face_photo_model.ipynb."""
    H, W = image_shape[:2]
    fx, fy, fw, fh, source = face_box
    face_side = max(fw, fh)
    face_cx = fx + fw / 2
    face_cy = fy + fh / 2
    photo_h = face_side * portrait_scale
    photo_w = photo_h * PHOTO_ASPECT
    x0 = face_cx - photo_w / 2
    y0 = face_cy - photo_h / 2
    x1 = face_cx + photo_w / 2
    y1 = face_cy + photo_h / 2
    return _clamp_aspect_box(x0, y0, x1, y1, W, H), f'face:{source}'


def detect_standard_photo_box(det, mp_mod, rgb, img_bgr_shape):
    """Detect face → portrait box. Fallback to generic left-side crop."""
    face = mediapipe_detect_face(det, mp_mod, rgb)
    if face is not None:
        return face_to_photo_box(face, img_bgr_shape)
    return regularize_photo_box(
        generic_portrait_fallback_box(img_bgr_shape), img_bgr_shape
    ), 'generic_portrait_fallback'


def crop_to_standard_photo(pil_img, box):
    """Crop and resize to standard portrait size.
    From unified_opencv_face_photo_model.ipynb."""
    crop = pil_img.crop(tuple(map(int, box)))
    return crop.resize(STANDARD_PHOTO_SIZE, Image.BICUBIC)


# DINOClassifier imported from models.models above

def resize_with_padding(img, size=MODEL_IMAGE_SIZE, fill=(245, 245, 245)):
    """From unified notebook cell 6."""
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new('RGB', (size, size), fill)
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


from torchvision import transforms
FACE_EVAL_TF = transforms.Compose([
    transforms.Lambda(lambda img: resize_with_padding(img, size=MODEL_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(FACE_NORM['mean'], FACE_NORM['std']),
])


class FacePhotoDataset(Dataset):
    """From unified notebook cell 6 (eval mode)."""
    def __init__(self, manifest_df):
        self.df = manifest_df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = Image.open(r['full_image_path']).convert('RGB')
        W, H = im.size
        box = (int(r.x0 * W), int(r.y0 * H), int(r.x1 * W), int(r.y1 * H))
        crop = crop_to_standard_photo(im, box)
        return FACE_EVAL_TF(crop), r['id']


def run_face_pipeline(image_rows, model_dir, batch_size=32):
    """Step 2: MediaPipe detect → DINOv2 classify → photo_prob per image."""
    print(f"[FACE] Processing {len(image_rows)} images...", file=sys.stderr)

    # Init MediaPipe
    mp_det, mp_mod = _init_mediapipe(model_dir)

    # Detect face boxes (threaded: per-thread MediaPipe instances, no lock needed)
    import threading
    _local = threading.local()
    manifest_rows = []
    def _detect_one(args):
        image_id, image_path = args
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            return None
        if not hasattr(_local, "det"):
            _local.det, _local.mp = _init_mediapipe(model_dir)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        box, source = detect_standard_photo_box(_local.det, _local.mp, rgb, bgr.shape)
        x0, y0, x1, y1 = box
        return {"id": image_id, "full_image_path": str(image_path),
                "x0": x0 / W, "y0": y0 / H, "x1": x1 / W, "y1": y1 / H, "source": source}
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=16) as pool:
        for row in tqdm(pool.map(_detect_one, image_rows), total=len(image_rows), desc="face detect (mediapipe)"):
            if row is not None:
                manifest_rows.append(row)
    mp_det.close()
    manifest = pd.DataFrame(manifest_rows)
    print(f"[FACE] Sources: {manifest['source'].value_counts().to_dict()}", file=sys.stderr)

    # Load DINOv2-small face model
    weights = model_dir / "face" / "dinov2_small_unified_face.pt"
    model = DINOClassifier(2)
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    model = model.to(DEVICE).eval()
    print(f"[FACE] Loaded {weights}", file=sys.stderr)

    # Classify (GPU, FP16, batched via DataLoader, ~1.2ms/face)
    loader = DataLoader(FacePhotoDataset(manifest), batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=(DEVICE == 'cuda'), persistent_workers=True)
    ids, probs = [], []
    with torch.no_grad(), _autocast():
        for x, batch_ids in tqdm(loader, desc="face classify (dinov2)"):
            p = torch.softmax(model(x.to(DEVICE)), dim=1)[:, 1].cpu().numpy()
            ids.extend(batch_ids)
            probs.extend(p.tolist())

    del model
    _empty_cache()
    return pd.DataFrame({'id': ids, 'photo_prob': probs})


# ════════════════════════════════════════════════════════════════════════════
# STEP 3: PaddleOCR text field detection + filtering
# PaddleOCR GPU: 35ms/image → 1.4h for 142k docs
# ════════════════════════════════════════════════════════════════════════════

_PADDLE_DET = {'model': None}



def _init_paddle_worker():
    """Process-pool worker initializer: each worker gets its own TextDetection."""
    global _PADDLE_DET
    import os, multiprocessing
    # OMP threads per worker — set by parent before pool creation
    cores = multiprocessing.cpu_count()
    # Read from env (parent sets this before spawning)
    n_workers = int(os.environ.get("_FREUID_PADDLE_WORKERS", "1"))
    threads = max(1, cores // max(1, n_workers))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    from paddleocr import TextDetection
    _PADDLE_DET['model'] = TextDetection()

def get_paddle():
    if _PADDLE_DET['model'] is None:
        _init_paddle_worker()
    return _PADDLE_DET['model']


def _result_to_boxes(r) -> np.ndarray:
    """From precache_paddle_boxes.py."""
    get = r.get if hasattr(r, "get") else (lambda k, d=None: r[k] if k in r else d)
    polys = get("dt_polys", None)
    out = []
    for b in (polys if polys is not None else []):
        b = np.asarray(b, np.float32)
        out.append([b[:, 0].min(), b[:, 1].min(), b[:, 0].max(), b[:, 1].max()])
    return np.asarray(out, np.float32) if out else np.zeros((0, 4), np.float32)


def filter_boxes(boxes: np.ndarray, H: int, W: int,
                 min_aspect: float = 0.5, max_w_frac: float = 0.85,
                 max_h_frac: float = 0.09) -> np.ndarray:
    """Post-filter text field boxes:
    - Skip top 17% header band (TOP_OCR_IGNORE_FRAC)
    - Skip tiny boxes (< MIN_W or MIN_H)
    - Skip vertical/tall boxes (w/h < 0.5) — borders, decorative lines
    - Skip tall boxes (h > 9% of image height) — flags, QR codes, photos
    - Skip wide boxes (w > 85% of image width) — full-width headers
    - Cap at MAX_FIELDS
    """
    kept = []
    for x0, y0, x1, y1 in boxes:
        bw, bh = x1 - x0, y1 - y0
        if y0 < TOP_OCR_IGNORE_FRAC * H:
            continue
        if bw < MIN_W or bh < MIN_H:
            continue
        if bh > 0 and bw / bh < min_aspect:
            continue  # vertical/tall — not a text field
        if bw > max_w_frac * W or bh > max_h_frac * H:
            continue  # too large — probably full-width header or border
        kept.append([x0, y0, x1, y1])
    if not kept:
        return np.zeros((0, 4), np.float32)
    return np.array(kept[:MAX_FIELDS], np.float32)


def _detect_single(args_tuple):
    """Worker function for parallel PaddleOCR detection."""
    image_id, image_path, cache_dir = args_tuple
    det = _PADDLE_DET['model']
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raw = np.zeros((0, 4), np.float32)
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = list(det.predict(rgb))
        raw = _result_to_boxes(result[0] if result else {})
    # Save to cache
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.save(Path(cache_dir) / f"{image_id}.npy", raw)
    return image_id, raw


def _auto_paddle_workers() -> int:
    """Auto-select paddle worker count."""
    try:
        import paddle
        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return 1  # GPU: single process (CUDA can't fork into child processes)
    except Exception:
        pass
    import platform, multiprocessing
    if platform.system() == "Darwin":
        return 1
    return max(1, multiprocessing.cpu_count())


def run_text_detection(image_rows, box_cache_dir=None, n_workers=None):
    """Step 3: PaddleOCR detect + filter per image. Returns dict[id] → Nx4 boxes.

    On CUDA: single-process (GPU handles parallelism).
    On CPU:  multiprocessing with min(4, cpu_count) workers for ~2-3x speedup.
    Caches results to box_cache_dir if set.
    """
    print(f"[TEXT DET] Detecting fields in {len(image_rows)} images...", file=sys.stderr)

    # Load from cache first (threaded for speed)
    all_boxes = {}
    cached, todo = 0, []

    if box_cache_dir:
        import threading
        from concurrent.futures import ThreadPoolExecutor
        _lock = threading.Lock()
        cache_dir = Path(box_cache_dir)

        def _load_cached(args):
            image_id, image_path = args
            cache_file = cache_dir / f"{image_id}.npy"
            if not cache_file.exists():
                return image_id, image_path, None
            raw = np.load(cache_file, allow_pickle=False)
            # Get image size for filtering — try size cache first, fall back to imread
            size_file = cache_dir / f"{image_id}_size.npy"
            if size_file.exists():
                sz = np.load(size_file)
                H, W = int(sz[0]), int(sz[1])
            else:
                bgr = cv2.imread(str(image_path))
                if bgr is None:
                    return image_id, image_path, np.zeros((0, 4), np.float32)
                H, W = bgr.shape[:2]
                np.save(size_file, np.array([H, W], dtype=np.int32))
            return image_id, image_path, filter_boxes(raw, H, W)

        with ThreadPoolExecutor(max_workers=16) as pool:
            for image_id, image_path, boxes in tqdm(
                pool.map(_load_cached, image_rows), total=len(image_rows), desc="load cache"):
                if boxes is not None:
                    all_boxes[image_id] = boxes
                    cached += 1
                else:
                    todo.append((image_id, image_path))
    else:
        todo = list(image_rows)

    if cached > 0:
        print(f"[TEXT DET] Loaded {cached} from cache, {len(todo)} to detect", file=sys.stderr)

    if todo:
        if n_workers is None:
            n_workers = _auto_paddle_workers()

        t0 = time.time()
        if n_workers <= 1:
            # Single-process — PaddleOCR can't batch variable-size images
            det = get_paddle()
            for idx, (image_id, image_path) in enumerate(tqdm(todo, desc="text detect (paddle)")):
                bgr = cv2.imread(str(image_path))
                if bgr is None:
                    all_boxes[image_id] = np.zeros((0, 4), np.float32)
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                H, W = rgb.shape[:2]
                result = list(det.predict(rgb))
                raw = _result_to_boxes(result[0] if result else {})
                if box_cache_dir:
                    Path(box_cache_dir).mkdir(parents=True, exist_ok=True)
                    np.save(Path(box_cache_dir) / f"{image_id}.npy", raw)
                    np.save(Path(box_cache_dir) / f"{image_id}_size.npy", np.array([H, W], dtype=np.int32))
                all_boxes[image_id] = filter_boxes(raw, H, W)
                if (idx + 1) % 500 == 0:
                    elapsed = time.time() - t0
                    rate = (idx + 1) / elapsed
                    eta = (len(todo) - idx - 1) / max(rate, 0.01)
                    print(f"  [{idx+1}/{len(todo)}] {rate:.1f} img/s, ETA {eta/60:.0f}min",
                          file=sys.stderr)
        else:
            global _PADDLE_N_WORKERS
            _PADDLE_N_WORKERS = n_workers
            os.environ["_FREUID_PADDLE_WORKERS"] = str(n_workers)
            threads_per = max(1, (os.cpu_count() or 1) // n_workers)
            print(f"[TEXT DET] Using {n_workers} parallel workers "
                  f"(OMP_NUM_THREADS={threads_per} per worker)",
                  file=sys.stderr)
            todo_lookup = {iid: str(ip) for iid, ip in todo}
            work_items = [(iid, str(ip), box_cache_dir) for iid, ip in todo]
            with ProcessPoolExecutor(max_workers=n_workers,
                                     initializer=_init_paddle_worker) as pool:
                for image_id, raw in tqdm(
                    pool.map(_detect_single, work_items, chunksize=4),
                    total=len(work_items), desc=f"text detect (paddle ×{n_workers})"
                ):
                    bgr = cv2.imread(todo_lookup[image_id])
                    if bgr is not None:
                        H, W = bgr.shape[:2]
                        all_boxes[image_id] = filter_boxes(raw, H, W)
                    else:
                        all_boxes[image_id] = np.zeros((0, 4), np.float32)

        elapsed = time.time() - t0
        print(f"[TEXT DET] Detected {len(todo)} images in {elapsed:.0f}s "
              f"({elapsed/max(len(todo),1)*1000:.0f}ms/img, {n_workers} workers)",
              file=sys.stderr)

    n_fields = [len(b) for b in all_boxes.values()]
    print(f"[TEXT DET] Total: {len(all_boxes)} images, avg fields/doc: {np.mean(n_fields):.1f}, "
          f"panels: {sum(n_fields)} (cached={cached}, detected={len(todo)})",
          file=sys.stderr)
    return all_boxes


# ════════════════════════════════════════════════════════════════════════════
# STEP 4: Forensic panels → ConvNeXt-B text classifier (FP16, cross-doc batching)
# ConvNeXt-B: 1.5ms/field (GPU FP16) → ~1.2h for 142k × 20 fields
# ════════════════════════════════════════════════════════════════════════════

# forensic_panel_from_crop, DINOv3Classifier, DINOClassifier
# all imported from train/scripts/ at the top of this file.

def extract_panels_for_doc(image_path, boxes):
    """Crop fields → forensic panels for one document. From block14.doc_field_panels."""
    rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    panels = []
    for x0, y0, x1, y1 in boxes:
        cx0 = max(0, int(x0) - PAD_X)
        cy0 = max(0, int(y0) - PAD_Y)
        cx1 = min(W, int(x1) + PAD_X)
        cy1 = min(H, int(y1) + PAD_Y)
        crop = rgb[cy0:cy1, cx0:cx1]
        if crop.shape[0] >= 6 and crop.shape[1] >= 6:
            panels.append(forensic_panel_from_crop(crop))
    return panels


def preprocess_panel(panel: np.ndarray) -> torch.Tensor:
    """Single panel → (3, 224, 1008) normalized tensor."""
    if panel.shape[:2] != (PANEL_H, PANEL_W):
        panel = cv2.resize(panel, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(panel.astype(np.float32) / 255.0).permute(2, 0, 1)
    return (t - IMAGENET_MEAN.squeeze(0)) / IMAGENET_STD.squeeze(0)


def run_text_scoring(image_rows, all_boxes, model_dir, batch_size=32, agg='top3',
                     box_cache_dir=None, output_path=None):
    """Step 4: Per-doc panel extraction + scoring.

    Simple sequential loop — panels built and scored per document.
    torch.compile + FP16 for GPU speed.
    """
    weights = model_dir / "dinov3_convnext_base_tamper_clf.pt"
    model = DINOv3Classifier()
    sd = torch.load(weights, map_location='cpu', weights_only=True)
    model.load_state_dict(sd, strict=False)
    model = model.to(DEVICE).eval()
    if DEVICE == "cuda":
        model = torch.compile(model, mode='reduce-overhead')
        with torch.no_grad(), _autocast():
            for bs in [1, batch_size]:
                _ = model(torch.randn(bs, 3, 224, 1008, device=DEVICE))
        print(f"[TEXT SCORE] torch.compile warmup done", file=sys.stderr)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[TEXT SCORE] Loaded {weights} ({params_m:.1f}M params, "
          f"batch_size={batch_size}, FP16={USE_FP16})", file=sys.stderr)

    cache_dir = Path(box_cache_dir) if box_cache_dir else None

    def _get_boxes(image_id, image_path):
        if all_boxes and image_id in all_boxes:
            return all_boxes[image_id]
        if cache_dir:
            npy = cache_dir / f"{image_id}.npy"
            sz = cache_dir / f"{image_id}_size.npy"
            if npy.exists():
                raw = np.load(npy, allow_pickle=False)
                if sz.exists():
                    s = np.load(sz)
                    return filter_boxes(raw, int(s[0]), int(s[1]))
                bgr = cv2.imread(str(image_path))
                if bgr is not None:
                    return filter_boxes(raw, bgr.shape[0], bgr.shape[1])
        return None

    results = {}
    total_panels = 0
    t0 = time.time()

    # Accumulate panels across docs, score in fixed batches
    batch_tensors = []
    batch_ids = []
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    doc_probs = {}  # image_id → [prob, ...]

    def _flush():
        nonlocal total_panels
        if not batch_tensors:
            return
        arr = np.stack(batch_tensors).astype(np.float32) / 255.0
        arr = (arr - mean) / std
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).to(DEVICE)
        with torch.no_grad(), _autocast():
            probs = torch.sigmoid(model(x)).cpu().numpy().tolist()
        for doc_id, prob in zip(batch_ids, probs):
            doc_probs.setdefault(doc_id, []).append(prob)
        total_panels += len(batch_tensors)
        batch_tensors.clear()
        batch_ids.clear()

    for idx, (image_id, image_path) in enumerate(tqdm(image_rows, desc="text score")):
        boxes = _get_boxes(image_id, image_path)
        if boxes is None or len(boxes) == 0:
            continue
        panels = extract_panels_for_doc(image_path, boxes)
        for panel in panels:
            if panel.shape[:2] != (PANEL_H, PANEL_W):
                panel = cv2.resize(panel, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
            batch_tensors.append(panel)
            batch_ids.append(image_id)
            if len(batch_tensors) >= batch_size:
                _flush()

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(image_rows) - idx - 1) / max(rate, 0.01)
            print(f"  [{idx+1}/{len(image_rows)}] {total_panels} panels, "
                  f"{total_panels/elapsed:.0f} panels/s, "
                  f"{rate:.1f} img/s, ETA {eta/60:.0f}min", file=sys.stderr)

    _flush()  # remaining panels

    # Aggregate per document
    for image_id, _ in image_rows:
        dp = doc_probs.get(image_id, [])
        if not dp:
            results[image_id] = (0.0, 0)
            continue
        dp.sort(reverse=True)
        if agg == 'max':
            score = dp[0]
        elif agg == 'top3':
            score = float(np.mean(dp[:min(3, len(dp))]))
        else:
            score = float(np.mean(dp))
        results[image_id] = (score, len(dp))

    print(f"[TEXT SCORE] Done: {total_panels} panels from {len(image_rows)} docs", file=sys.stderr)
    del model
    _empty_cache()
    return results


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 + 6: Combine scores → submission CSV
# fraud_score = max(photo_prob, text_prob)
# (Verified from public_test_subtask_predictions.csv)
# ════════════════════════════════════════════════════════════════════════════

def combine_and_write(image_rows, face_df, text_results, output_path):
    face_lookup = face_df.set_index('id')['photo_prob'].to_dict() if face_df is not None else {}
    rows = []
    for image_id, _ in image_rows:
        photo_prob = face_lookup.get(image_id, 0.0)
        text_prob, n_fields = text_results.get(image_id, (0.0, 0))
        fraud_score = max(photo_prob, text_prob)
        rows.append({
            'id': image_id,
            'label': float(fraud_score),
            'photo_prob': float(photo_prob),
            'text_prob': float(text_prob),
            'n_fields': n_fields,
        })

    df = pd.DataFrame(rows)
    assert len(df) == len(image_rows), f"Row count mismatch: {len(df)} vs {len(image_rows)}"
    assert np.isfinite(df['label'].to_numpy()).all(), "Non-finite labels"

    # Write submission (id, label only)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df[['id', 'label']].to_csv(output_path, index=False)

    # Write detailed predictions alongside
    detail_path = str(output_path).replace('.csv', '_detailed.csv')
    df.to_csv(detail_path, index=False)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Wrote {len(df)} rows → {output_path}", file=sys.stderr)
    print(f"Detailed → {detail_path}", file=sys.stderr)
    print(f"  photo_fake (>0.5): {(df['photo_prob'] > 0.5).sum()}", file=sys.stderr)
    print(f"  text_fake  (>0.5): {(df['text_prob'] > 0.5).sum()}", file=sys.stderr)
    print(f"  fraud (label>0.5): {(df['label'] > 0.5).sum()}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    return df


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="FREUID full inference pipeline.")
    ap.add_argument("--image-dir", type=str, required=True)
    ap.add_argument("--model-dir", type=str, default="../models")
    ap.add_argument("--output", type=str, default="submission.csv")
    ap.add_argument("--box-cache", type=str, default=None,
                    help="Dir to cache/load PaddleOCR boxes (.npy per image).")
    ap.add_argument("--log", type=str, default=None,
                    help="Log file path (also prints to stderr).")
    ap.add_argument("--face-batch-size", type=int, default=32)
    ap.add_argument("--text-batch-size", type=int, default=32)
    ap.add_argument("--agg", default="max", choices=["max", "top3", "mean"])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--no-fp16", action="store_true", help="Disable FP16 autocast")
    ap.add_argument("--steps", default="all", choices=["all", "face", "text", "text-detect", "text-score"],
                    help="Run only specific steps: face, text (detect+score), text-detect, text-score, or all")
    args = ap.parse_args()

    # Tee stderr to log file (captures all prints + tqdm)
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        _log_file = open(args.log, 'w')
        class _Tee:
            def __init__(self, *streams):
                self.streams = streams
            def write(self, msg):
                for s in self.streams:
                    s.write(msg)
                    s.flush()
            def flush(self):
                for s in self.streams:
                    s.flush()
        sys.stderr = _Tee(sys.__stderr__, _log_file)
        print(f"Logging to {args.log}", file=sys.stderr)

    global DEVICE, USE_FP16
    if args.device != "auto":
        DEVICE = args.device
    USE_FP16 = (DEVICE == "cuda") and not args.no_fp16

    print(f"Device: {DEVICE}  FP16: {USE_FP16}", file=sys.stderr)
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", file=sys.stderr)

    image_dir = Path(args.image_dir)
    model_dir = Path(args.model_dir)

    # Step 1: discover images
    image_rows = discover_images(image_dir)
    total_images = len(image_rows)
    print(f"[STEP 1] Found {total_images} images in {image_dir}", file=sys.stderr)


    t_total = time.time()
    run_steps = args.steps

    # Step 2: face detect + classify
    face_df = None
    t_face = 0
    if run_steps in ("all", "face"):
        t1 = time.time()
        face_df = run_face_pipeline(image_rows, model_dir, batch_size=args.face_batch_size)
        t_face = time.time() - t1
        print(f"[STEP 2] Face pipeline done in {t_face:.0f}s ({t_face/60:.1f}min)", file=sys.stderr)
        # Save intermediate face results
        face_csv = Path(args.output).parent / "face_probs.csv"
        face_df.to_csv(face_csv, index=False)
        print(f"[FACE] Saved intermediate → {face_csv}", file=sys.stderr)
    else:
        # Try loading saved face results
        face_csv = Path(args.output).parent / "face_probs.csv"
        if face_csv.exists():
            face_df = pd.read_csv(face_csv, dtype={"id": str})
            print(f"[FACE] Loaded {len(face_df)} from {face_csv}", file=sys.stderr)
        else:
            print(f"[FACE] Skipped (no saved face_probs.csv found)", file=sys.stderr)

    # Step 3: text field detection (with caching)
    all_boxes = {}
    t_det = 0
    if run_steps in ("all", "text", "text-detect"):
        t2 = time.time()
        all_boxes = run_text_detection(image_rows, box_cache_dir=args.box_cache)
        t_det = time.time() - t2
        print(f"[STEP 3] Text detection done in {t_det:.0f}s ({t_det/60:.1f}min)", file=sys.stderr)
    elif run_steps == "text-score" and args.box_cache:
        # Don't load all boxes — scoring will stream from cache per-doc
        print(f"[STEP 3] Boxes will be loaded per-doc from {args.box_cache}", file=sys.stderr)

    # Step 4: text field scoring
    text_results = {}
    t_score = 0
    if run_steps in ("all", "text", "text-score"):
        t3 = time.time()
        text_results = run_text_scoring(image_rows, all_boxes, model_dir,
                                        batch_size=args.text_batch_size, agg=args.agg,
                                        box_cache_dir=args.box_cache,
                                        output_path=args.output)
        t_score = time.time() - t3
        print(f"[STEP 4] Text scoring done in {t_score:.0f}s ({t_score/60:.1f}min)", file=sys.stderr)

    # Step 5+6: combine + write
    if face_df is not None or text_results:
        df = combine_and_write(image_rows, face_df, text_results, args.output)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"TIMING SUMMARY ({len(image_rows)} images)", file=sys.stderr)
    print(f"  Face (detect+classify): {t_face:7.0f}s  ({t_face/60:5.1f}min)", file=sys.stderr)
    print(f"  Text detection:         {t_det:7.0f}s  ({t_det/60:5.1f}min)", file=sys.stderr)
    print(f"  Text scoring:           {t_score:7.0f}s  ({t_score/60:5.1f}min)", file=sys.stderr)
    print(f"  Total:                  {elapsed:7.0f}s  ({elapsed/60:5.1f}min)", file=sys.stderr)
    print(f"  Rate: {len(image_rows)/max(elapsed,0.01):.1f} img/s", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


if __name__ == "__main__":
    main()
