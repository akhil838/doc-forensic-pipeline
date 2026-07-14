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


def _robust_uint8(a: np.ndarray, valid: np.ndarray = None, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    """Normalize to 0-255 using robust percentiles. Optimized: accepts pre-computed valid mask."""
    vals = a[valid] if valid is not None else a[np.isfinite(a)]
    if len(vals) == 0:
        return np.zeros(a.shape, np.uint8)
    lo, hi = np.percentile(vals, [p_low, p_high])
    rng = max(hi - lo, 1e-6)
    return np.clip((a - lo) * (255.0 / rng), 0, 255).astype(np.uint8)


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

    Optimized: single gray conversion, shared bg_sigma blur, minimal allocations.
    """
    crop_rgb, valid, _ = letterbox_rgb(crop_rgb, out_w=tile_w, out_h=tile_h)
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Ink mask: adaptive + absolute threshold
    local_gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=5, sigmaY=5)
    ink = (((gray < local_gray - 22) & (gray < 175)) | (gray < 80)).astype(np.uint8)
    ink = cv2.dilate(ink, np.ones((3, 3), np.uint8), iterations=1)

    # LAB for residuals — compute L, A, B from crop
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # Background subtraction — shared sigma
    bg_sigma = max(8, min(tile_w, tile_h) // 8)
    L_bg = cv2.GaussianBlur(L, (0, 0), sigmaX=bg_sigma, sigmaY=bg_sigma)
    L_resid = np.abs(L - L_bg)
    A_resid = A - cv2.GaussianBlur(A, (0, 0), sigmaX=bg_sigma, sigmaY=bg_sigma)
    B_resid = B - cv2.GaussianBlur(B, (0, 0), sigmaX=bg_sigma, sigmaY=bg_sigma)
    chroma = np.sqrt(A_resid * A_resid + B_resid * B_resid)

    # Texture variance
    mean = cv2.blur(gray, (15, 15))
    texture = np.sqrt(np.maximum(cv2.blur(gray * gray, (15, 15)) - mean * mean, 0))

    # Edge (Sobel)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(sx * sx + sy * sy)

    # Assemble — avoid redundant gray conversion, build directly
    gray_u8 = gray.astype(np.uint8)
    gray_u8[~valid] = 245
    ink_u8 = ink * 255
    ink_u8[~valid] = 0
    L_u8 = _robust_uint8(L_resid, valid); L_u8[~valid] = 0
    ch_u8 = _robust_uint8(chroma, valid); ch_u8[~valid] = 0
    tx_u8 = _robust_uint8(texture, valid); tx_u8[~valid] = 0
    ed_u8 = _robust_uint8(edge, valid); ed_u8[~valid] = 0

    # Build panel without per-view cvtColor — stack as 3-channel directly
    panel = np.empty((PANEL_H, PANEL_W, 3), np.uint8)
    # Row 1
    panel[:tile_h, :tile_w] = gray_u8[:, :, None]
    panel[:tile_h, tile_w:2*tile_w] = ink_u8[:, :, None]
    panel[:tile_h, 2*tile_w:] = L_u8[:, :, None]
    # Row 2
    panel[tile_h:, :tile_w] = ch_u8[:, :, None]
    panel[tile_h:, tile_w:2*tile_w] = tx_u8[:, :, None]
    panel[tile_h:, 2*tile_w:] = ed_u8[:, :, None]
    return panel


def real_ink_mask(crop: np.ndarray) -> np.ndarray:
    """Binary ink mask from a real text field crop."""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    local = cv2.GaussianBlur(gray, (0, 0), sigmaX=5, sigmaY=5)
    ink = ((gray < local - 22) & (gray < 175)) | (gray < 80)
    return cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8), 1).astype(bool)
