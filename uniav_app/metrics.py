"""Localization metrics (class-agnostic), all on per-video dicts.

Inputs: ``preds_by_video = {vid: [(ts, te, score), ...]}`` and ``gts_by_video = {vid: [(ts, te), ...]}``.

* ``recall_at_n``: a GT counts when the best of the N highest-scoring proposals has tIoU >= t (not
  one-to-one). With N = min(K_GT, #pred) it is exactly Athena's ``train.py::evaluate`` R@t / mIoU.
* ``average_precision``: ActivityNet-style (``compute_average_precision_detection`` +
  ``interpolated_prec_rec``): predictions of all videos sorted by score, each GT matched at most once
  (the unmatched GT with the highest tIoU), interpolated AP.
* ``prf``: precision / recall / F1 after one-to-one Hungarian matching (``match.hungarian_match``).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .match import hungarian_match, tiou_matrix


def _by_score(preds) -> list:
    return sorted(preds, key=lambda p: -float(p[2]))


def _n_for(n, vid: str, n_gt: int, n_pred: int) -> int:
    if n == "K_GT":
        return min(n_gt, n_pred)
    if n == "K_pred" or n is None:
        return n_pred
    if isinstance(n, dict):
        return min(int(n[vid]), n_pred)
    return min(int(n), n_pred)


def best_tious(preds_by_video, gts_by_video, n=None) -> dict[str, np.ndarray]:
    """Best tIoU of every GT among the top-N proposals of its video (N: int, 'K_GT', 'K_pred', dict)."""
    out = {}
    for vid, gts in gts_by_video.items():
        preds = _by_score(preds_by_video.get(vid, []))
        k = _n_for(n, vid, len(gts), len(preds))
        m = tiou_matrix(preds[:k], gts)
        out[vid] = m.max(axis=0) if m.shape[0] else np.zeros(len(gts))
    return out


def recall_at_n(preds_by_video, gts_by_video, n=None, thrs=(0.3, 0.5, 0.7)) -> dict[float, float]:
    """Fraction of GT (pooled over videos) covered with tIoU >= t by the top-N proposals of their video."""
    b = np.concatenate(list(best_tious(preds_by_video, gts_by_video, n).values()) or [np.zeros(0)])
    return {t: float((b >= t).mean()) if b.size else 0.0 for t in thrs}


def mean_best_tiou(preds_by_video, gts_by_video, n=None) -> float:
    b = np.concatenate(list(best_tious(preds_by_video, gts_by_video, n).values()) or [np.zeros(0)])
    return float(b.mean()) if b.size else 0.0


def interpolated_ap(prec: np.ndarray, rec: np.ndarray) -> float:
    """ActivityNet ``interpolated_prec_rec``."""
    mprec = np.hstack([[0.0], prec, [0.0]])
    mrec = np.hstack([[0.0], rec, [1.0]])
    for i in range(len(mprec) - 2, -1, -1):
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx]))


def average_precision(preds_by_video, gts_by_video, thr: float) -> float:
    """Class-agnostic AP at one tIoU threshold (ActivityNet detection protocol)."""
    npos = sum(len(g) for g in gts_by_video.values())
    if npos == 0:
        return 0.0
    flat = [(vid, float(p[0]), float(p[1]), float(p[2])) for vid, ps in preds_by_video.items()
            if vid in gts_by_video for p in ps]
    if not flat:
        return 0.0
    order = np.argsort(-np.array([f[3] for f in flat]), kind="mergesort")
    lock = {vid: np.full(len(g), -1) for vid, g in gts_by_video.items()}
    tp = np.zeros(len(flat))
    fp = np.zeros(len(flat))
    for rank, i in enumerate(order):
        vid, ts, te, _ = flat[i]
        gts = gts_by_video[vid]
        if not gts:
            fp[rank] = 1
            continue
        ious = tiou_matrix([(ts, te)], gts)[0]
        for j in np.argsort(-ious, kind="mergesort"):
            if ious[j] < thr:
                fp[rank] = 1
                break
            if lock[vid][j] >= 0:
                continue
            tp[rank] = 1
            lock[vid][j] = rank
            break
        if tp[rank] == 0 and fp[rank] == 0:
            fp[rank] = 1
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    rec = tp_c / npos
    prec = tp_c / (tp_c + fp_c)
    return interpolated_ap(prec, rec)


def mean_ap(preds_by_video, gts_by_video, thrs=(0.3, 0.4, 0.5, 0.6, 0.7)) -> dict:
    aps = {t: average_precision(preds_by_video, gts_by_video, t) for t in thrs}
    return {"AP": aps, "mAP": float(np.mean(list(aps.values()))) if aps else 0.0}


def prf(preds_by_video, gts_by_video, thr: float, match: Callable = hungarian_match) -> dict:
    """Precision / recall / F1 after one-to-one matching at tIoU >= thr, pooled over videos."""
    tp = n_pred = n_gt = 0
    for vid, gts in gts_by_video.items():
        preds = preds_by_video.get(vid, [])
        n_pred += len(preds)
        n_gt += len(gts)
        tp += len(match(preds, gts, thr))
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"tp": tp, "n_pred": n_pred, "n_gt": n_gt, "precision": p, "recall": r, "f1": f}
