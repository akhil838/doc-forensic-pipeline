#!/usr/bin/env python
"""Train unified DINOv2-small face/photo detector.

Usage:
  python train_face.py --data-root /path/to/competition/data

Uses OpenCV face detection for portrait crop extraction.
Trains on subtask_annotations.csv photo_cutout labels.
Saves checkpoint to --output-dir (default: models/).
"""

# === Original notebook cell 0 ===
import json, os, random, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModel

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Defaults — overridden by CLI args in main()
DATA_ROOT = Path('.')
ANN_CSV = Path('annotations/subtask_annotations.csv')
MODEL_DIR = Path('models')

# Standard portrait crop before model preprocessing: (width, height).
STANDARD_PHOTO_SIZE = (256, 320)
MODEL_IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 8
VAL_FRACTION = 0.20
NUM_WORKERS = 0
DINO_MODEL_NAME = 'facebook/dinov2-small'
DINO_HEAD_LR = 3e-4
DINO_BACKBONE_LR = 1e-5
WEIGHT_DECAY = 1e-4
DINO_FREEZE_BACKBONE = False
TRAIN_CAP_PER_CLASS = None  # set an int for faster experiments; None uses all positives + matched negatives

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print('device:', DEVICE)

# === Data loading (deferred to main() for CLI usage) ===
# These globals are populated by main() or by notebook %run.
face_train = None
COUNTRY_TYPES = []

def _load_module_data():
    """Load data at module level — used when running as notebook, not CLI."""
    global face_train, COUNTRY_TYPES
    train = pd.read_csv(DATA_ROOT / 'train_labels.csv')
    ann = pd.read_csv(ANN_CSV).set_index('id')

    image_root_candidates = [DATA_ROOT, DATA_ROOT / 'train', DATA_ROOT.parent]
    IMAGE_ROOT = next((r for r in image_root_candidates if (r / train.iloc[0]['image_path']).exists()), DATA_ROOT)
    train['full_image_path'] = train['image_path'].map(lambda p: str(IMAGE_ROOT / p))
    train['photo_label'] = train['id'].map(ann['photo_cutout'])
    face_train = train[train['photo_label'].notna()].copy()
    face_train['photo_label'] = face_train['photo_label'].astype(int)
    COUNTRY_TYPES = sorted(face_train['type'].unique())
    print('train rows:', len(train), 'manual photo labels:', len(face_train))

# Only auto-load if data exists at default path (notebook mode)
if (DATA_ROOT / 'train_labels.csv').exists():
    _load_module_data()

# === Face detection — MediaPipe BlazeFace (matches inference pipeline) ===
_BLAZE_DET = None
_BLAZE_MP = None

def _init_blaze(model_dir=None):
    """Initialize MediaPipe BlazeFace detector (lazy singleton)."""
    global _BLAZE_DET, _BLAZE_MP
    if _BLAZE_DET is not None:
        return _BLAZE_DET, _BLAZE_MP
    import mediapipe as mp
    # Search for blaze_face.tflite
    candidates = [
        Path(model_dir) / "blaze_face.tflite" if model_dir else None,
        Path(__file__).parent.parent / "models" / "weights" / "blaze_face.tflite",
        Path("models/weights/blaze_face.tflite"),
    ]
    model_path = next((p for p in candidates if p and p.exists()), None)
    if model_path is None:
        raise FileNotFoundError("blaze_face.tflite not found — download from HuggingFace weights")
    _BLAZE_DET = mp.tasks.vision.FaceDetector.create_from_options(
        mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            min_detection_confidence=0.3))
    _BLAZE_MP = mp
    print(f"MediaPipe BlazeFace loaded: {model_path}")
    return _BLAZE_DET, _BLAZE_MP

def detect_largest_face(img_bgr):
    """Detect largest left-side face with MediaPipe BlazeFace."""
    det, mp = _init_blaze()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_bgr.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = det.detect(mp_image)
    if not result.detections:
        return None
    best = None
    best_area = 0
    for d in result.detections:
        bb = d.bounding_box
        x0, y0, w, h = bb.origin_x, bb.origin_y, bb.width, bb.height
        area = w * h
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

PHOTO_ASPECT = STANDARD_PHOTO_SIZE[0] / STANDARD_PHOTO_SIZE[1]

def _clamp_aspect_box(x0, y0, x1, y1, W, H, aspect=PHOTO_ASPECT):
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


def face_to_photo_box(face_box, image_shape, portrait_scale=2.15):
    """Expand face detection box into a centered portrait frame."""
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

def generic_portrait_fallback_box(image_shape):
    H, W = image_shape[:2]
    return _clamp_aspect_box(0.02 * W, 0.16 * H, 0.38 * W, 0.92 * H, W, H)

def regularize_photo_box(box, image_shape, target_height_frac=0.58, min_height_frac=0.42, max_height_frac=0.72, snap=16):
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

def detect_standard_photo_box(img_bgr, row=None, allow_manual_fallback=False):
    """Detect face → portrait box. Fallback to generic left-side crop (matches inference)."""
    face = detect_largest_face(img_bgr)
    if face is not None:
        return face_to_photo_box(face, img_bgr.shape)
    return regularize_photo_box(generic_portrait_fallback_box(img_bgr.shape), img_bgr.shape), 'generic_portrait_fallback'

def face_size_summary(face_box, image_shape):
    """Pixel + relative size for the detected face box."""
    if face_box is None:
        return {'face_x': np.nan, 'face_y': np.nan, 'face_w': np.nan, 'face_h': np.nan,
                'face_area_pct': np.nan, 'face_source': 'none'}
    H, W = image_shape[:2]
    fx, fy, fw, fh, source = face_box
    return {
        'face_x': int(fx), 'face_y': int(fy), 'face_w': int(fw), 'face_h': int(fh),
        'face_area_pct': float(100.0 * fw * fh / (W * H)),
        'face_source': source,
    }


def crop_to_standard_photo(pil_img, box):
    crop = pil_img.crop(tuple(map(int, box)))
    return crop.resize(STANDARD_PHOTO_SIZE, Image.BICUBIC)

# === Original notebook cell 4 ===
# --- Manifest cache: one standardized photo crop per labeled document ---
FACE_MANIFEST = Path('unified_face_photo_manifest.csv')

def build_or_load_face_manifest(df, path=FACE_MANIFEST, force=False, limit=None, workers=None):
    if path.exists() and not force and limit is None:
        out = pd.read_csv(path)
        print(f'loaded manifest: {path} rows={len(out)}')
        return out

    work = df if limit is None else df.sample(min(limit, len(df)), random_state=SEED)
    records = work.to_dict('records')
    if workers is None:
        workers = min(8, os.cpu_count() or 1)

    def detect_one(r):
        img_bgr = cv2.imread(r['full_image_path'])
        if img_bgr is None:
            return None
        H, W = img_bgr.shape[:2]
        box, source = detect_standard_photo_box(img_bgr, row=r, allow_manual_fallback=True)
        x0, y0, x1, y1 = box
        return {
            'id': r['id'], 'type': r['type'], 'label': int(r['photo_label']),
            'document_label': int(r['label']), 'full_image_path': r['full_image_path'],
            'x0': x0 / W, 'y0': y0 / H, 'x1': x1 / W, 'y1': y1 / H,
            'source': source, 'image_w': W, 'image_h': H,
        }

    if workers <= 1:
        rows = [detect_one(r) for r in tqdm(records, desc='detect face/photo boxes')]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(tqdm(pool.map(detect_one, records), total=len(records),
                             desc=f'detect face/photo boxes [{workers} threads]'))
    rows = [r for r in rows if r is not None]
    out = pd.DataFrame(rows)
    if limit is None:
        out.to_csv(path, index=False)
        print(f'saved manifest: {path} rows={len(out)} workers={workers}')
    return out

face_manifest = None
if face_train is not None:
    face_manifest = build_or_load_face_manifest(face_train)

# === Original notebook cell 5 ===
# --- Transforms, dataset, balanced unified split ---
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
PAD_FILL = (245, 245, 245)

def resize_with_padding(img, size=MODEL_IMAGE_SIZE, fill=PAD_FILL):
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new('RGB', (size, size), fill)
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas

class RandomCameraLook:
    def __init__(self, blur_p=0.25, tint_p=0.25, blur_sigma=(0.10, 0.65), tint_strength=(0.02, 0.06)):
        self.blur_p = blur_p; self.tint_p = tint_p
        self.blur_sigma = blur_sigma; self.tint_strength = tint_strength
        self.tint_colors = ((255, 235, 210), (220, 235, 255))
    def __call__(self, img):
        if random.random() < self.blur_p:
            img = transforms.functional.gaussian_blur(img, kernel_size=3, sigma=self.blur_sigma)
        if random.random() < self.tint_p:
            img = img.convert('RGB')
            img = Image.blend(img, Image.new('RGB', img.size, random.choice(self.tint_colors)), random.uniform(*self.tint_strength))
        return img

TRAIN_TF = transforms.Compose([
    transforms.Lambda(lambda img: resize_with_padding(img, size=MODEL_IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.92, 1.08), fill=PAD_FILL),
    transforms.ColorJitter(brightness=0.18, contrast=0.25, saturation=0.15, hue=0.04),
    RandomCameraLook(),
    transforms.RandomGrayscale(p=0.15),
    transforms.ToTensor(), NORM,
])
EVAL_TF = transforms.Compose([
    transforms.Lambda(lambda img: resize_with_padding(img, size=MODEL_IMAGE_SIZE)),
    transforms.ToTensor(), NORM,
])

class UnifiedFacePhotoDataset(Dataset):
    def __init__(self, manifest_df, transform):
        self.df = manifest_df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = Image.open(r['full_image_path']).convert('RGB')
        W, H = im.size
        box = (int(r.x0 * W), int(r.y0 * H), int(r.x1 * W), int(r.y1 * H))
        crop = crop_to_standard_photo(im, box)
        return self.transform(crop), int(r['label']), r['id']

def balanced_unified_split(manifest_df, val_fraction=VAL_FRACTION, cap_per_class=TRAIN_CAP_PER_CLASS):
    fake = manifest_df[manifest_df['label'] == 1]
    real = manifest_df[manifest_df['label'] == 0]
    n = min(len(fake), len(real)) if cap_per_class is None else min(len(fake), len(real), cap_per_class)
    bal = pd.concat([
        fake.sample(n, random_state=SEED),
        real.sample(n, random_state=SEED),
    ]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    strat = bal['type'].astype(str) + '_' + bal['label'].astype(str)
    if strat.value_counts().min() < 2:
        strat = bal['label']
    tr, va = train_test_split(bal, test_size=val_fraction, stratify=strat, random_state=SEED)
    return tr.reset_index(drop=True), va.reset_index(drop=True), n

train_df, val_df, n_per_class = None, None, None
if face_manifest is not None:
    train_df, val_df, n_per_class = balanced_unified_split(face_manifest)
    print(f'balanced unified split: n/class={n_per_class} train={len(train_df)} val={len(val_df)}')

# === Original notebook cell 6 ===
# --- Model + train/eval loops ---
class DINOClassifier(nn.Module):
    def __init__(self, n_classes=2, model_name=DINO_MODEL_NAME, dropout=0.1):
        super().__init__()
        import os
        token = os.environ.get("HF_TOKEN")
        kwargs = {"token": token} if token else {}
        self.backbone = AutoModel.from_pretrained(model_name, **kwargs)
        hidden = self.backbone.config.hidden_size
        if DINO_FREEZE_BACKBONE:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, n_classes))
    def forward(self, x):
        out = self.backbone(pixel_values=x, interpolate_pos_encoding=True)
        feat = out.pooler_output if getattr(out, 'pooler_output', None) is not None else out.last_hidden_state[:, 0]
        return self.head(feat)

def make_optimizer(model):
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if name.startswith('backbone.') else head_params).append(p)
    groups = []
    if backbone_params: groups.append({'params': backbone_params, 'lr': DINO_BACKBONE_LR})
    if head_params: groups.append({'params': head_params, 'lr': DINO_HEAD_LR})
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)

@torch.no_grad()
def evaluate(model, loader, desc='eval'):
    model.eval(); probs, labels, ids = [], [], []
    for x, y, batch_ids in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        p = torch.softmax(model(x.to(DEVICE)), dim=1)[:, 1].cpu().numpy()
        probs.extend(p.tolist()); labels.extend(y.numpy().tolist()); ids.extend(batch_ids)
    probs, labels = np.asarray(probs), np.asarray(labels)
    preds = (probs >= 0.5).astype(int)
    return {
        'accuracy': float(accuracy_score(labels, preds)),
        'f1': float(f1_score(labels, preds, zero_division=0)),
        'auc': float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else float('nan'),
        'n': int(len(labels)),
    }, pd.DataFrame({'id': ids, 'label': labels, 'prob': probs})

def train_unified_face_model(train_df=train_df, val_df=val_df):
    train_loader = DataLoader(UnifiedFacePhotoDataset(train_df, TRAIN_TF), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE == 'cuda'), drop_last=False)
    val_loader = DataLoader(UnifiedFacePhotoDataset(val_df, EVAL_TF), batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=(DEVICE == 'cuda'))
    val_aug_loader = DataLoader(UnifiedFacePhotoDataset(val_df, TRAIN_TF), batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=(DEVICE == 'cuda'))
    model = DINOClassifier(2).to(DEVICE)
    opt = make_optimizer(model)
    crit = nn.CrossEntropyLoss()
    best, best_state = {'auc': -1.0}, None
    for epoch in range(EPOCHS):
        model.train(); running = 0.0
        pbar = tqdm(train_loader, desc=f'unified face train ep{epoch+1}/{EPOCHS}', dynamic_ncols=True)
        for bi, (x, y, _) in enumerate(pbar, 1):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            running += loss.item()
            pbar.set_postfix(loss=f'{running / bi:.4f}')
        clean, _ = evaluate(model, val_loader, desc=f'val clean ep{epoch+1}/{EPOCHS}')
        aug, _ = evaluate(model, val_aug_loader, desc=f'val aug ep{epoch+1}/{EPOCHS}')
        print(f"epoch {epoch+1}: clean acc={clean['accuracy']:.3f} f1={clean['f1']:.3f} auc={clean['auc']:.3f} | aug acc={aug['accuracy']:.3f} f1={aug['f1']:.3f} auc={aug['auc']:.3f}")
        if clean['auc'] >= best['auc']:
            best = clean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(best_state, MODEL_DIR / 'unified_face.pt')
    meta = {
        'task': 'photo_cutout', 'countries': COUNTRY_TYPES, 'single_model_all_countries': True,
        'crop_source': 'mediapipe blazeface face detection → portrait expansion; generic fallback',
        'manual_label_source': str(ANN_CSV), 'standard_photo_size': list(STANDARD_PHOTO_SIZE),
        'model_image_size': MODEL_IMAGE_SIZE, 'backbone': DINO_MODEL_NAME,
        'balanced_n_per_class': int(n_per_class), 'val_metrics': best,
        'source_counts': face_manifest['source'].value_counts().to_dict(),
        'norm_mean': [0.485, 0.456, 0.406], 'norm_std': [0.229, 0.224, 0.225],
        'classes': {'0': 'real', '1': 'fake'},
    }
    (MODEL_DIR / 'unified_face.json').write_text(json.dumps(meta, indent=2))
    print(f"saved -> {MODEL_DIR / 'unified_face.pt'} best val auc={best['auc']:.4f}")
    return model, best

# Training runs via main() CLI or notebook — not at import time
unified_model, unified_result = None, None

# === CLI entry point ===
def main():
    import argparse

    global DATA_ROOT, ANN_CSV, MODEL_DIR, EPOCHS, BATCH_SIZE, DINO_BACKBONE_LR, DINO_HEAD_LR
    global IMAGE_ROOT, TRAIN_DIR, COUNTRY_TYPES

    ap = argparse.ArgumentParser(description="Train unified DINOv2-small face detector.")
    ap.add_argument("--data-root", type=str, default=str(DATA_ROOT),
                    help="Competition data root (contains train_labels.csv)")
    ap.add_argument("--ann-csv", type=str, default=str(ANN_CSV),
                    help="Path to subtask_annotations.csv")
    ap.add_argument("--output-dir", type=str, default=str(MODEL_DIR),
                    help="Output dir for checkpoint")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--backbone-lr", type=float, default=DINO_BACKBONE_LR)
    ap.add_argument("--head-lr", type=float, default=DINO_HEAD_LR)
    args = ap.parse_args()

    DATA_ROOT = Path(args.data_root)
    ANN_CSV = Path(args.ann_csv)
    MODEL_DIR = Path(args.output_dir); MODEL_DIR.mkdir(exist_ok=True)
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    DINO_BACKBONE_LR = args.backbone_lr
    DINO_HEAD_LR = args.head_lr

    # Re-derive paths
    train_labels = pd.read_csv(DATA_ROOT / "train_labels.csv")
    IMAGE_ROOT = next(r for r in [DATA_ROOT, DATA_ROOT / "train", DATA_ROOT.parent]
                      if (r / train_labels.iloc[0]["image_path"]).exists())
    TRAIN_DIR = IMAGE_ROOT / "train"

    ann = pd.read_csv(ANN_CSV).set_index("id")
    train_labels["full_image_path"] = train_labels["image_path"].map(lambda p: str(IMAGE_ROOT / p))
    train_labels["photo_label"] = train_labels["id"].map(ann["photo_cutout"])
    face_data = train_labels[train_labels["photo_label"].notna()].copy()
    face_data["photo_label"] = face_data["photo_label"].astype(int)
    COUNTRY_TYPES = sorted(face_data["type"].unique())

    print(f"data_root: {DATA_ROOT}")
    print(f"ann_csv: {ANN_CSV}")
    print(f"output_dir: {MODEL_DIR}")
    print(f"device: {DEVICE}")
    print(f"train rows with photo labels: {len(face_data)}")

    global face_manifest, face_train, n_per_class, train_df, val_df
    face_train = face_data
    face_manifest = build_or_load_face_manifest(face_train)
    train_df, val_df, n_per_class = balanced_unified_split(face_manifest)
    print(f"balanced split: n/class={n_per_class} train={len(train_df)} val={len(val_df)}")

    unified_model, unified_result = train_unified_face_model(train_df, val_df)
    print(f"Training complete. Best val AUC: {unified_result['auc']:.4f}")


if __name__ == "__main__":
    main()
