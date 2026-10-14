"""Exp 5 (UEMR_FINAL §8, plan ``UEMR_BUOC_TIEP_P6.md`` §4): segment score ``cos(q, e_i) + beta * log conf_i``.

Only R6 (``uniav_pred``) has a confidence: ``conf`` of ``uniav_cache/ft/segments_pred.jsonl`` (Athena
``sigmoid(event)^0.7 * sigmoid(IoU)^0.3`` after soft-NMS; a merged segment keeps the max, a segment stretched
to ``d_min`` keeps its own). The video score stays MaxSim over the biased segment scores, and the argmax
segment (VCMR, ``mom@.5|top1``) is taken on the biased scores too. No training, no encoding.

    rows = conf_rows(local_cache, segments_pred_jsonl)
    dev  = sweep(rows, load_text(local_cache, "text_dev.npz"), BETAS, dev_videos)
    beta = pick_beta(dev)                                   # on dev only; val is scored once afterwards
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .evaluate import evaluate_partition
from .search import SegmentIndex

BETAS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1)
TOL = 0.011   # segments.npz keeps ts/te rounded to 0.01 s (partitions_uemr._round)


def read_conf(path) -> dict[str, list[tuple[float, float, float]]]:
    """``video_id -> [(ts, te, conf), ...]`` sorted by time, from ``segments_pred.jsonl``."""
    out = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["video_id"]].append((float(r["ts"]), float(r["te"]), float(r["conf"])))
    return {v: sorted(s) for v, s in out.items()}


def align_conf(video_id, ts, te, conf_by_video: dict, tol: float = TOL) -> np.ndarray:
    """``conf`` of every row, matched by order inside each video.

    ``partitions_uemr`` built ``uniav_pred`` from ``segments_pred.jsonl`` by rounding to 0.01 s and clipping
    ``te`` to the mp4 duration, so the rows of a video, sorted by ``(ts, te)``, pair one to one with its
    lines. Raises if a video has another segment count, or a pair differs by more than ``tol`` (only the
    last-ending segment may be shorter, by the clip).
    """
    video_id, ts, te = np.asarray(video_id), np.asarray(ts, np.float64), np.asarray(te, np.float64)
    conf = np.empty(len(video_id))
    for v in np.unique(video_id):
        rows = np.flatnonzero(video_id == v)
        rows = rows[np.lexsort((te[rows], ts[rows]))]
        pred = np.asarray(conf_by_video.get(str(v), []), dtype=np.float64).reshape(-1, 3)
        if len(pred) != len(rows):
            raise ValueError(f"{v}: {len(rows)} segments in the cache, {len(pred)} in segments_pred.jsonl")
        d_ts, d_te = np.abs(ts[rows] - pred[:, 0]), pred[:, 1] - te[rows]
        clipped = (d_te > tol) & (te[rows] == te[rows].max())
        bad = (d_ts > tol) | ((np.abs(d_te) > tol) & ~clipped)
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            raise ValueError(f"{v}: segment {i} is ({ts[rows][i]:.2f}, {te[rows][i]:.2f}) in the cache but "
                             f"({pred[i, 0]:.2f}, {pred[i, 1]:.2f}) in segments_pred.jsonl")
        conf[rows] = pred[:, 2]
    if not (conf > 0).all():
        raise ValueError(f"{int((conf <= 0).sum())} segments have conf <= 0: log conf undefined")
    return conf


def conf_rows(cache_dir, conf_path, partition: str = "uniav_pred") -> dict[str, np.ndarray]:
    """The ``partition`` rows of ``segments.npz`` plus ``log_conf`` aligned with them."""
    with np.load(Path(cache_dir) / "segments.npz") as blob:
        sel = blob["partition"] == partition
        rows = {k: blob[k][sel] for k in ("emb", "video_id", "ts", "te")}
    rows["emb"] = rows["emb"].astype(np.float32)
    rows["log_conf"] = np.log(align_conf(rows["video_id"], rows["ts"], rows["te"], read_conf(conf_path)))
    rows["partition"] = partition
    return rows


def index_for(rows: dict, beta: float, videos=None) -> SegmentIndex:
    """``SegmentIndex`` with ``bias = beta * log conf`` (none at beta 0, so beta 0 is exactly column b)."""
    index = SegmentIndex(rows["emb"], rows["video_id"], rows["ts"], rows["te"], rows["partition"],
                         beta * rows["log_conf"] if beta else None)
    return index if videos is None else index.subset(videos)


def sweep(rows: dict, text: dict, betas=BETAS, videos=None) -> list[dict]:
    """One ``evaluate_partition`` row per beta (R@k, MedR, MnR, VCMR R@1, ``mom@.5|top1``)."""
    return [dict(evaluate_partition(index_for(rows, b, videos), text, "max"), beta=float(b)) for b in betas]


def pick_beta(rows: list[dict], metric: str = "R@1") -> float:
    """Best ``metric``; a tie goes to the smaller beta."""
    return float(min(rows, key=lambda r: (-round(r[metric], 6), r["beta"]))["beta"])


def spread(rows: dict, text: dict, videos=None) -> dict:
    """Scale of the two terms: std of ``cos(q, e_i)`` over every (query, gallery segment) pair, and of ``log conf``."""
    m = np.ones(len(rows["video_id"]), bool) if videos is None else np.isin(rows["video_id"], list(videos))
    cos = text["emb"] @ rows["emb"][m].T
    lc = rows["log_conf"][m]
    return {"segments": int(m.sum()), "cos mean": float(cos.mean()), "cos std": float(cos.std()),
            "log conf mean": float(lc.mean()), "log conf std": float(lc.std()),
            "log conf min": float(lc.min()), "log conf max": float(lc.max())}
