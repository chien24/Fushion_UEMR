"""Temporal IoU and matching between predicted segments and GT segments (pure numpy / scipy)."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def seg_array(segs) -> np.ndarray:
    """Segments as an (N, 2) float array. Accepts (ts, te[, ...]) tuples or dicts with ts/te or t_s/t_e."""
    rows = []
    for s in segs:
        if isinstance(s, dict):
            rows.append((float(s["ts"] if "ts" in s else s["t_s"]), float(s["te"] if "te" in s else s["t_e"])))
        else:
            rows.append((float(s[0]), float(s[1])))
    return np.asarray(rows, dtype=np.float64).reshape(-1, 2)


def tiou(a: Sequence[float], b: Sequence[float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return float(inter / union) if union > 0 else 0.0


def tiou_matrix(preds, gts) -> np.ndarray:
    """(P, G) temporal IoU between every prediction and every GT segment."""
    p, g = seg_array(preds), seg_array(gts)
    if len(p) == 0 or len(g) == 0:
        return np.zeros((len(p), len(g)))
    inter = np.clip(np.minimum(p[:, None, 1], g[None, :, 1]) - np.maximum(p[:, None, 0], g[None, :, 0]), 0, None)
    union = (p[:, 1] - p[:, 0])[:, None] + (g[:, 1] - g[:, 0])[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def hungarian_match(preds, gts, thr: float) -> list[tuple[int, int, float]]:
    """One-to-one matching with ``scipy.optimize.linear_sum_assignment`` on the tIoU matrix.

    Only pairs with tIoU >= thr count. The weight ``M + tIoU`` (M > number of possible pairs) makes the
    assignment maximise the number of valid pairs first, then their total tIoU.
    Returns ``[(pred_idx, gt_idx, tiou)]`` sorted by gt index.
    """
    from scipy.optimize import linear_sum_assignment
    m = tiou_matrix(preds, gts)
    if m.size == 0:
        return []
    valid = m >= thr
    if not valid.any():
        return []
    big = float(min(m.shape) + 1)
    w = np.where(valid, big + m, 0.0)
    rows, cols = linear_sum_assignment(w, maximize=True)
    pairs = [(int(i), int(j), float(m[i, j])) for i, j in zip(rows, cols) if valid[i, j]]
    return sorted(pairs, key=lambda x: x[1])


def best_match(preds, gts) -> list[tuple[int, float]]:
    """For every GT: (index of the prediction with the highest tIoU or -1, that tIoU). Not one-to-one."""
    m = tiou_matrix(preds, gts)
    if m.shape[0] == 0:
        return [(-1, 0.0)] * m.shape[1]
    j = m.argmax(axis=0)
    return [(int(j[g]), float(m[j[g], g])) for g in range(m.shape[1])]
