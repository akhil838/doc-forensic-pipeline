"""
FREUID official metric: AuDET + APCER@1%BPCER → FREUID score (lower is better).
"""
import numpy as np


def _det_curve(y_true, y_score):
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    order = np.argsort(-y_score, kind="mergesort")
    s_sorted = y_score[order]
    y_sorted = y_true[order]
    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    distinct = np.r_[np.diff(s_sorted) != 0, True]
    tp_cum = tp_cum[distinct]
    fp_cum = fp_cum[distinct]
    bpcer = fp_cum / n_neg
    apcer = 1.0 - tp_cum / n_pos
    bpcer = np.concatenate(([0.0], bpcer))
    apcer = np.concatenate(([1.0], apcer))
    return bpcer, apcer


def compute_freuid_score(y_true, y_score, bpcer_target=0.01):
    """Official FREUID metric. Lower is better."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    bpcer, apcer = _det_curve(y_true, y_score)
    audet = float(np.trapezoid(apcer, bpcer))
    eps = 1e-12
    feasible = bpcer <= bpcer_target + eps
    apcer_at_bpcer = float(apcer[int(np.flatnonzero(feasible).max())]) if feasible.any() else 1.0
    g_a, g_p = 1 - audet, 1 - apcer_at_bpcer
    denom = g_a + g_p
    freuid = 1.0 - (2 * g_a * g_p / denom) if denom > 0 else 1.0
    return {
        "audet": audet,
        "apcer_at_1_bpcer": apcer_at_bpcer,
        "freuid_score": freuid,
    }
