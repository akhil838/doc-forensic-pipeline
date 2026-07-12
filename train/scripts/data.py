"""
Data loading: annotations, field extraction, paddle box cache, dataset classes.
"""
from __future__ import annotations
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .forensic_panels import forensic_panel_from_crop

# ── Constants ────────────────────────────────────────────────────────────────

INPUT_H = 224
INPUT_W = 1008
TOP_OCR_IGNORE_FRAC = 0.17  # skip top 17% of image (header/title band)
PAD_X, PAD_Y = 15, 5
MIN_W, MIN_H = 10, 8
MAX_FIELDS = 40
SEED = 42

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ── Annotation loading ───────────────────────────────────────────────────────

def load_train_labels(data_root: Path) -> pd.DataFrame:
    """Load competition train_labels.csv."""
    return pd.read_csv(data_root / "train_labels.csv", dtype={"id": str})


def load_subtask_annotations(ann_dir: Path) -> pd.DataFrame:
    """Load document-level photo/text subtask labels."""
    return pd.read_csv(ann_dir / "subtask_annotations.csv", dtype={"id": str})


def load_field_annotations(ann_dir: Path) -> pd.DataFrame:
    """Load per-field tamper annotations. Returns only 'done' rows with tampered fields."""
    df = pd.read_csv(ann_dir / "field_tamper_annotations.csv", dtype={"id": str})
    df = df[df["status"] == "done"].copy()
    df["tidx"] = df["tampered_idxs"].map(
        lambda x: set(json.loads(x)) if isinstance(x, str) and x else set()
    )
    return df


# ── Paddle box cache ─────────────────────────────────────────────────────────

def load_paddle_boxes(paddle_cache: Path, image_id: str) -> np.ndarray:
    """Load cached PaddleOCR field boxes (Nx4 float32: x0, y0, x1, y1)."""
    cache_file = paddle_cache / f"{image_id}.npy"
    if cache_file.exists():
        return np.load(cache_file, allow_pickle=False)
    return np.zeros((0, 4), np.float32)


# ── Field crop extraction ────────────────────────────────────────────────────

def extract_field_crops(image_path: str | Path, image_id: str, paddle_cache: Path,
                        top_ignore: float = TOP_OCR_IGNORE_FRAC,
                        min_w: int = MIN_W, min_h: int = MIN_H,
                        pad_x: int = PAD_X, pad_y: int = PAD_Y,
                        max_fields: int = MAX_FIELDS,
                        min_aspect: float = 0.5,
                        max_w_frac: float = 0.85,
                        max_h_frac: float = 0.09) -> list[np.ndarray]:
    """Extract text field crops from an image using cached paddle boxes.

    Filters: header band, tiny boxes, vertical/tall boxes, oversized boxes.
    """
    rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    boxes = load_paddle_boxes(paddle_cache, image_id)
    crops = []
    for x0, y0, x1, y1 in boxes:
        bw, bh = x1 - x0, y1 - y0
        if y0 < top_ignore * H:
            continue
        if bw < min_w or bh < min_h:
            continue
        if bh > 0 and bw / bh < min_aspect:
            continue
        if bw > max_w_frac * W or bh > max_h_frac * H:
            continue
        cx0 = max(0, int(x0) - pad_x)
        cy0 = max(0, int(y0) - pad_y)
        cx1 = min(W, int(x1) + pad_x)
        cy1 = min(H, int(y1) + pad_y)
        crop = rgb[cy0:cy1, cx0:cx1]
        if crop.shape[0] >= 6 and crop.shape[1] >= 6:
            crops.append(crop)
        if len(crops) >= max_fields:
            break
    return crops


def split_long_crop(crop: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """Split wide crops into 2-3 segments for training variety."""
    h, w = crop.shape[:2]
    if w < 160:
        return []
    n = int(rng.integers(2, 4))
    cuts = np.linspace(0, w, n + 1).astype(int)
    return [crop[:, cuts[i]:cuts[i + 1]] for i in range(n) if cuts[i + 1] - cuts[i] >= 40]


# ── Panel dataset ────────────────────────────────────────────────────────────

class PanelDataset(Dataset):
    """Dataset of forensic panels (224×1008×3 uint8 or paths to saved JPEGs).

    Augmentations (when augment=True):
      - HSV hue/saturation shift
      - Brightness/contrast jitter
      - Gaussian blur
      - JPEG recompression
      - Gaussian noise
      - Horizontal/vertical flip
    """

    def __init__(self, panels: list, labels: np.ndarray, augment: bool = False):
        """panels: list of np.ndarray (HxWx3) or list of file paths (str)."""
        self.panels = panels
        self.labels = labels.astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.panels)

    def _load(self, idx):
        p = self.panels[idx]
        if isinstance(p, (str, Path)):
            img = cv2.imread(str(p))
            if img is None:
                return np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return p

    def __getitem__(self, idx):
        p = self._load(idx).copy()

        if self.augment:
            # HSV hue/saturation shift
            if random.random() < 0.6:
                hsv = cv2.cvtColor(p, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-30, 30)) % 180
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.7, 1.3), 0, 255)
                p = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            # Brightness/contrast
            if random.random() < 0.5:
                alpha = random.uniform(0.7, 1.3)
                beta = random.uniform(-20, 20)
                p = np.clip(p.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            # Gaussian blur
            if random.random() < 0.2:
                k = random.choice([3, 5])
                p = cv2.GaussianBlur(p, (k, k), 0)
            # JPEG recompression
            if random.random() < 0.3:
                _, buf = cv2.imencode(".jpg", cv2.cvtColor(p, cv2.COLOR_RGB2BGR),
                                      [cv2.IMWRITE_JPEG_QUALITY, random.randint(30, 80)])
                p = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            # Gaussian noise
            if random.random() < 0.2:
                noise_std = random.uniform(3, 12)
                p = np.clip(p.astype(np.float32) + np.random.normal(0, noise_std, p.shape),
                            0, 255).astype(np.uint8)
            # Horizontal flip
            if random.random() < 0.5:
                p = np.ascontiguousarray(p[:, ::-1])
            # Vertical flip
            if random.random() < 0.2:
                p = np.ascontiguousarray(p[::-1, :])

        if p.shape[:2] != (INPUT_H, INPUT_W):
            p = cv2.resize(p, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(p.astype(np.float32) / 255.0).permute(2, 0, 1)
        x = (x - IMAGENET_MEAN.squeeze(0)) / IMAGENET_STD.squeeze(0)
        return x, self.labels[idx]
