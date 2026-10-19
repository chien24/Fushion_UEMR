"""Dense windows and gap windows for E3 (docs/uemr2/UEMR2_AGENT_SO_BO.md §3.2).

``dense(duration, W)``: windows of length W, step W/2, starts 0, W/2, ... kept while ``start + W/2 < duration`` (so
every second is covered and the last window, clipped to the video, is longer than W/2); a video shorter than W/2 has
one window [0, duration].

``gap_mask(windows, r6)``: windows with tIoU = 0 with every R6 segment of the video **and** at least ``min_out`` of
their length outside the union of the R6 segments. (tIoU = 0 already means no overlap, so the second condition
only matters if ``tiou_max`` is raised above 0.)
"""

from __future__ import annotations

import numpy as np

from omni_retrieval.search import tiou_vec

DENSE_W = (4, 8, 16)


def dense(duration: float, w: float) -> np.ndarray:
    """``[K, 2]`` windows in seconds."""
    duration = float(duration)
    starts = np.arange(0.0, max(duration - w / 2, 0.0), w / 2) if duration > w / 2 else np.zeros(1)
    return np.stack([starts, np.minimum(starts + w, duration)], 1).astype(np.float32)


def outside_fraction(windows: np.ndarray, segs: np.ndarray) -> np.ndarray:
    """Share of each window's length not covered by the union of ``segs``."""
    windows = np.asarray(windows, np.float64).reshape(-1, 2)
    segs = np.asarray(segs, np.float64).reshape(-1, 2)
    if not len(segs):
        return np.ones(len(windows))
    s = segs[np.argsort(segs[:, 0])]
    merged = [list(s[0])]
    for a, b in s[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    m = np.asarray(merged)
    inter = np.clip(np.minimum(windows[:, None, 1], m[None, :, 1]) - np.maximum(windows[:, None, 0], m[None, :, 0]),
                    0, None).sum(1)
    length = np.clip(windows[:, 1] - windows[:, 0], 1e-9, None)
    return 1 - inter / length


def gap_mask(windows: np.ndarray, r6: np.ndarray, tiou_max: float = 0.0, min_out: float = 0.5) -> np.ndarray:
    windows = np.asarray(windows, np.float64).reshape(-1, 2)
    r6 = np.asarray(r6, np.float64).reshape(-1, 2)
    if not len(r6):
        return np.ones(len(windows), bool)
    t = tiou_vec(windows[:, None, 0], windows[:, None, 1], r6[None, :, 0], r6[None, :, 1]).max(1)
    return (t <= tiou_max) & (outside_fraction(windows, r6) >= min_out)
