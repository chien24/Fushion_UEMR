"""Partitions and dev rules of the YC2 supplementary run (``notebooks/uemr/UEMR_YC2_BO_SUNG.md``,
``configs/bo_sung_protocol.json``). Pure functions on ``[[ts, te], ...]`` lists in seconds.

* ``fill_gaps`` (R8): the R6 segments, plus equal windows of about ``l_fill`` s over every stretch of the video that
  no R6 segment covers and that lasts at least ``l_min`` s (the start and the end of the video count as stretches);
  ``max(1, round(length / l_fill))`` windows per stretch, rounded half up.
* ``centred_windows`` (R5b): one window of ``duration / K`` s centred on each K-Medoids medoid, clipped to the video.
* ``overlap``: share of the summed segment length that lies under another segment of the same video.
* ``pick_level`` (budget curve) and ``pick_fill`` (R8): the dev selection rules written down before val is scored.
* ``frame_index``: the 8 frames Omni-fix's ``video_decord`` reads from a segment, for the CLIP baseline.
"""

from __future__ import annotations

import numpy as np

from .partitions_uemr import coverage

EPS = 1e-9


def uncovered(segs, duration: float) -> list[list[float]]:
    """Stretches of ``[0, duration]`` that no segment covers, sorted by time."""
    out, end = [], 0.0
    for ts, te in sorted(segs):
        if float(ts) > end + EPS:
            out.append([end, float(ts)])
        end = max(end, float(te))
    if float(duration) > end + EPS:
        out.append([end, float(duration)])
    return out


def fill_gaps(segs, duration: float, l_min: float, l_fill: float) -> tuple[list[list[float]], int]:
    """``(R6 segments + fill windows sorted by time, number of fill windows)``."""
    fills = []
    for a, b in uncovered(segs, duration):
        if b - a >= l_min - EPS:
            n = max(1, int((b - a) / l_fill + 0.5))
            edges = np.linspace(a, b, n + 1)
            fills += [[float(x), float(y)] for x, y in zip(edges[:-1], edges[1:])]
    return sorted([[float(ts), float(te)] for ts, te in segs] + fills), len(fills)


def centred_windows(t_medoids, duration: float, k: int) -> list[list[float]]:
    """``[t - d/2K, t + d/2K]`` clipped to ``[0, duration]`` for every medoid time ``t`` (clamped into the video
    first: Z's ``t_sec`` may lie past the end of the mp4)."""
    half = float(duration) / int(k) / 2
    out = []
    for t in t_medoids:
        t = min(max(float(t), 0.0), float(duration))
        out.append([max(0.0, t - half), min(float(duration), t + half)])
    return sorted(out)


def overlap(segs) -> float:
    """``(sum of lengths - length of their union) / sum of lengths``; 0 for disjoint segments."""
    total = sum(float(te) - float(ts) for ts, te in segs)
    return 1.0 - coverage(segs, 1.0) / total if total > 0 else 0.0


def pick_level(rows: list[dict], target: float, tol: float = 0.5) -> dict:
    """Budget level for one target K: ``rows`` = ``{'method': 'theta' | 'topk', 'param', 'K dev'}``.

    The theta whose mean K on dev is closest to ``target`` (tie -> larger theta, fewer vectors); if even that one
    misses the target by more than ``tol``, the same rule on the top-k rows (tie -> smaller k).
    """
    def best(method, sign):
        cand = [r for r in rows if r["method"] == method]
        return min(cand, key=lambda r: (abs(r["K dev"] - target), sign * r["param"])) if cand else None

    th = best("theta", -1)
    if th is not None and abs(th["K dev"] - target) <= tol:
        return th
    tk = best("topk", 1)
    if tk is None:
        if th is None:
            raise ValueError("no candidate rows")
        return th
    return tk if th is None or abs(tk["K dev"] - target) < abs(th["K dev"] - target) else th


def pick_fill(rows: list[dict], tie: float = 0.3) -> dict:
    """R8 config: highest dev R@1; configs less than ``tie`` points below it -> fewest vectors/video, then R@1."""
    top = max(r["R@1"] for r in rows)
    cand = [r for r in rows if r["R@1"] > top - tie]
    return min(cand, key=lambda r: (r["vec/video"], -r["R@1"]))


def frame_index(ts: float, te: float, fps: float, n_frames: int, k: int = 8) -> np.ndarray:
    """``linspace(round(ts fps), min(round(te fps), n - 1), k)`` as ``video_decord`` (base_interval 0.5 s, 8 frames)."""
    start, end = round(float(ts) * fps), min(round(float(te) * fps), int(n_frames) - 1)
    if start > end:
        raise ValueError(f"segment [{ts}, {te}] lies past the last frame ({n_frames} at {fps} fps)")
    return np.linspace(start, end, k, dtype=int)
