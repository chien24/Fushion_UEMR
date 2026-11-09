"""The rows of plan §5 in the YC2 code paths, with every parameter from the protocol, and the two count rules of §4.2.

R6 is the chain of the YC2 runs (``uemr_yc2_bo_sung_*.ipynb``): ``postprocess_uemr`` on the raw proposals -> 3 decimals
-> ``partitions_uemr._round`` (2 decimals, never past the mp4). Names are the YC2 partition names, so ``SegmentIndex``
reads them unchanged.
"""

from __future__ import annotations

import numpy as np

from omni_retrieval.bo_sung import fill_gaps, pick_level
from omni_retrieval.partitions import uniform
from omni_retrieval.partitions_uemr import _round, uni_chunk
from uniav_app.postprocess import postprocess_uemr

ROWS = {"global": "R0", "uni_chunk": "R2", "uni_M": "R3", "uniav_pred": "R6", "r8": "R8", "uniav_gtart": "R6gt",
        "gt": "R7"}
FILES = {name: row.lower() for name, row in ROWS.items()}


def r6(proposals: list[dict], duration: float, theta: float, k_max: int, d_min: float, merge_tiou: float) -> list[list[float]]:
    """Segments of one video; a proposal the mp4 does not contain (UniAV clips to the annotated duration) is dropped."""
    segs = postprocess_uemr(proposals, duration, theta, k_max, d_min, merge_tiou)
    return [s for s in _round([[round(float(s["ts"]), 3), round(float(s["te"]), 3)] for s in segs], duration) if s[1] > s[0]]


def mean_k(proposals: dict, durations: dict, theta: float, k_max: int, d_min: float, merge_tiou: float) -> float:
    return float(np.mean([len(r6(proposals[v], durations[v], theta, k_max, d_min, merge_tiou)) for v in sorted(durations)]))


def pick_theta(proposals: dict, durations: dict, target: float, k_max: int, d_min: float, merge_tiou: float,
               grid) -> tuple[float, float, list[dict]]:
    """θ whose mean segment count over the gallery is closest to ``target`` (tie: larger θ, fewer vectors; the
    rule of ``bo_sung.pick_level``). Uses the predicted segments only, no label. ``(θ, mean K, every grid row)``."""
    rows = [{"method": "theta", "param": float(t), "K dev": mean_k(proposals, durations, float(t), k_max, d_min, merge_tiou)}
            for t in grid]
    best = pick_level(rows, target)
    return best["param"], best["K dev"], rows


def pick_chunk(durations: dict, target: int, d_min: float) -> tuple[float, int]:
    """Length L of the R2 chunks whose total clip count over the gallery is closest to ``target`` (R8's); bisection on L
    to 0.01 s. Durations only. ``(L, total clips at L)``; tie: the larger L, fewer vectors."""
    vids = sorted(durations)

    def total(length: float) -> int:
        return sum(len(uni_chunk(durations[v], length, d_min)) for v in vids)

    lo, hi = 0.5, float(max(durations.values())) + 1.0
    while hi - lo > 0.01:
        mid = (lo + hi) / 2
        if total(mid) > target:
            lo = mid
        else:
            hi = mid
    best = min((round(lo, 2), round(hi, 2)), key=lambda x: (abs(total(x) - target), -x))
    return best, total(best)


def resolve(props_pred: dict, props_gt: dict, durations: dict, gt: dict, target_k: float, p: dict) -> dict:
    """The two rules of §4.2 in order, then every row: ``p`` holds ``k_max, d_min, merge_tiou, l_min, l_fill, theta_grid``
    (start, stop, step). ``{'theta', 'mean_k', 'target_k', 'chunk_sec', 'clips_r8', 'clips_r2', 'parts'}``."""
    vids = sorted(durations)
    a, b, s = p["theta_grid"]
    grid = np.round(np.arange(a, b, s), 6)
    theta, k, _ = pick_theta({v: props_pred[v] for v in vids}, durations, target_k, p["k_max"], p["d_min"], p["merge_tiou"], grid)
    r6_ = {v: r6(props_pred[v], durations[v], theta, p["k_max"], p["d_min"], p["merge_tiou"]) for v in vids}
    r6gt = {v: r6(props_gt[v], durations[v], theta, p["k_max"], p["d_min"], p["merge_tiou"]) for v in vids}
    clips_r8 = sum(len(fill_gaps(r6_[v], durations[v], p["l_min"], p["l_fill"])[0]) for v in vids)
    chunk, clips_r2 = pick_chunk(durations, clips_r8, p["d_min"])
    parts = build(r6_, r6gt, {v: gt[v] for v in vids}, durations, chunk, p["d_min"], p["l_min"], p["l_fill"])
    return {"theta": float(theta), "mean_k": k, "target_k": target_k, "chunk_sec": chunk, "clips_r8": int(clips_r8),
            "clips_r2": int(clips_r2), "parts": parts}


def build(r6_parts: dict, r6gt_parts: dict, gt: dict, durations: dict, chunk_sec: float, d_min: float, l_min: float,
          l_fill: float) -> dict[str, dict]:
    """``{name: {video_id: [[ts, te], ...]}}`` for every video of ``durations``. R3 has ``K_pred`` = the number of
    R6 segments of the video; R2 chunks of ``chunk_sec``; R8 = R6 + gap windows (``fill_gaps``). R6gt: the oracle
    row (GT article, same θ); R7: the distinct GT windows."""
    vids = sorted(durations)
    for name, part in (("R6", r6_parts), ("R6gt", r6gt_parts), ("GT", gt)):
        missing = [v for v in vids if not part.get(v)]
        if missing:
            raise ValueError(f"{len(missing)} videos have no {name} segment: {missing[:5]}")
    return {
        "global": {v: _round([[0.0, durations[v]]], durations[v]) for v in vids},
        "uni_chunk": {v: _round(uni_chunk(durations[v], chunk_sec, d_min), durations[v]) for v in vids},
        "uni_M": {v: _round(uniform(durations[v], len(r6_parts[v])), durations[v]) for v in vids},
        "uniav_pred": {v: r6_parts[v] for v in vids},
        "uniav_gtart": {v: r6gt_parts[v] for v in vids},
        "gt": {v: gt[v] for v in vids},
        "r8": {v: _round(fill_gaps(r6_parts[v], durations[v], l_min, l_fill)[0], durations[v]) for v in vids},
    }
