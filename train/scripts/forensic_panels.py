"""
Forensic panel generation: text field crop → 224×1008 six-view composite.

Views (2×3 grid of 112×336 tiles):
  Row 1: grayscale | ink mask | |L| residual
  Row 2: chroma residual | texture variance | edge magnitude
"""
from __future__ import annotations
import cv2
import numpy as np

TILE_W = 336
TILE_H = 112
PANEL_H = 224  # 2 × TILE_H
PANEL_W = 1008  # 3 × TILE_W


def _robust_uint8(a: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    a = a.astype(np.float32)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros(a.shape, np.uint8)
    lo, hi = np.percentile(a[finite], [p_low, p_high])
    out = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    out[~finite] = 0
    return (out * 255).astype(np.uint8)


def letterbox_rgb(crop_rgb: np.ndarray, out_w: int = TILE_W, out_h: int = TILE_H,
                  fill: int = 245) -> tuple[np.ndarray, np.ndarray, float]:
    h, w = crop_rgb.shape[:2]
    scale = min(out_h / max(h, 1), out_w / max(w, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((out_h, out_w, 3), fill, dtype=np.uint8)
    y0 = (out_h - new_h) // 2
    x0 = (out_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    valid = np.zeros((out_h, out_w), bool)
    valid[y0:y0 + new_h, x0:x0 + new_w] = True
    return canvas, valid, scale


def forensic_panel_from_crop(crop_rgb: np.ndarray,
                              tile_w: int = TILE_W, tile_h: int = TILE_H) -> np.ndarray:
    """Build 224×1008 six-view forensic panel from a text-field crop (RGB uint8).

    Layout (2 rows × 3 cols of 112×336 tiles):
        Row 1: grayscale | ink mask | |L| residual
        Row 2: chroma residual | texture variance | edge magnitude

    Each view highlights a different tampering signal:
        - Grayscale: raw text appearance, font/weight consistency
        - Ink mask: detects reprinted text (different ink density/coverage)
        - L residual: background-subtracted luminance reveals inserted/pasted content
        - Chroma: color inconsistency in CIELAB space (different paper/ink color)
        - Texture: local variance detects smooth reprinted patches vs textured originals
        - Edge: Sobel gradient reveals halos/blurring around tampered text boundaries
    """
    # Letterbox to standard tile size, track valid (non-padded) pixels
    crop_rgb, valid, _ = letterbox_rgb(crop_rgb, out_w=tile_w, out_h=tile_h)
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # Ink mask: dual threshold — adaptive (dark relative to local mean) + absolute dark
    # Catches both printed text on light backgrounds and very dark ink on any background
    local_gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=5, sigmaY=5)
    ink = ((gray < local_gray - 22) & (gray < 175)) | (gray < 80)
    ink = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)

    # Background-subtracted residuals in CIELAB space
    # Removes slowly-varying background pattern, leaving only local anomalies
    bg_sigma = max(8, min(tile_w, tile_h) // 8)
    L_resid = L - cv2.GaussianBlur(L, (0, 0), sigmaX=bg_sigma, sigmaY=bg_sigma)
    A_resid = A - cv2.GaussianBlur(A, (0, 0), sigmaX=bg_sigma, sigmaY=bg_sigma)
    B_resid = B - cv2.GaussianBlur(B, (0, 0), sigmaX=bg_sigma, sigmaY=bg_sigma)
    chroma = np.sqrt(A_resid ** 2 + B_resid ** 2)

    # Local texture variance — smooth patches (reprinted/pasted) vs textured (original)
    mean = cv2.blur(gray, (15, 15))
    texture = np.sqrt(np.maximum(cv2.blur(gray * gray, (15, 15)) - mean * mean, 0))

    # Edge magnitude (Sobel) — detects halos/blurring at tamper boundaries
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(sx * sx + sy * sy)

    # Assemble 6 views into 2×3 grid; grayscale pad=245 (matches doc background), others=0
    views = [
        cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.uint8),  # 1. grayscale
        ink * 255,                                                      # 2. ink mask
        _robust_uint8(np.where(valid, np.abs(L_resid), np.nan)),       # 3. L residual
        _robust_uint8(np.where(valid, chroma, np.nan)),                # 4. chroma
        _robust_uint8(np.where(valid, texture, np.nan)),               # 5. texture
        _robust_uint8(np.where(valid, edge, np.nan)),                  # 6. edge
    ]
    for i in range(len(views)):
        views[i][~valid] = 245 if i == 0 else 0
    views_rgb = [cv2.cvtColor(v, cv2.COLOR_GRAY2RGB) for v in views]
    top = np.concatenate(views_rgb[:3], axis=1)    # row 1: gray | ink | L_resid
    bottom = np.concatenate(views_rgb[3:], axis=1)  # row 2: chroma | texture | edge
    return np.concatenate([top, bottom], axis=0)     # 224 × 1008 × 3


def real_ink_mask(crop: np.ndarray) -> np.ndarray:
    """Binary ink mask from a real text field crop."""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    local = cv2.GaussianBlur(gray, (0, 0), sigmaX=5, sigmaY=5)
    ink = ((gray < local - 22) & (gray < 175)) | (gray < 80)
    return cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8), 1).astype(bool)
