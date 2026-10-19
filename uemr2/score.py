"""Scoring of UEMR 2 (docs/uemr2/UEMR2_AGENT_SO_BO.md §4.1-4.3), one code path for branch A (q with t_iv2) and
branch B (h with t_omni).

* ``video_max``: segment score = sum_k w_k * (cos(T_k, X_k) - mu_k) / sd_k over sources that share the same rows
  (A-e: Omni e and q of the same segment), then max over the segments of each video (MaxSim) and its argmax row.
  One source with w = 1, mu = 0, sd = 1 is exactly column (b) / (d).
* ``e3``: video score = max(video_max over R6 by Omni, video_max over windows by the cheap source + gamma).
* ``ranks``: strict, 1 + videos scored higher than the true one (as ``omni_retrieval.evaluate``).
* ``fold_split`` / ``cv_pick`` / ``oof``: 2-fold cross-validation by video on val; the gallery is never shrunk.

Matrix products run in float32 (numpy has no fast float16 product); vectors are stored in float16.
"""

from __future__ import annotations

import numpy as np

from omni_retrieval.evaluate import rank_metrics
from omni_retrieval.search import tiou_vec


def layout(seg_video, videos) -> np.ndarray:
    """``pad [V, K]``: rows of each gallery video (in ``videos`` order), -1 = padding; other videos' rows are dropped."""
    vi = {v: i for i, v in enumerate(videos)}
    by = [[] for _ in videos]
    for r, v in enumerate(np.asarray(seg_video)):
        i = vi.get(str(v), -1)
        if i >= 0:
            by[i].append(r)
    pad = np.full((len(videos), max(1, max(map(len, by)))), -1, np.int64)
    for i, rows in enumerate(by):
        pad[i, :len(rows)] = rows
    return pad


def video_max(terms, pad: np.ndarray, chunk: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """``terms``: ``[(T [Q, d], X [N, d], mu, sd, w), ...]`` on the same N rows. Returns ``(S [Q, V], arg [Q, V])``
    with ``arg`` the row of the best segment of each video (-1 for a video without rows)."""
    terms = [(np.asarray(T, np.float32), np.asarray(X, np.float32), float(mu), float(sd), float(w))
             for T, X, mu, sd, w in terms if w != 0]
    q = len(terms[0][0])
    valid = pad >= 0
    safe = np.where(valid, pad, 0)
    S = np.empty((q, pad.shape[0]), np.float32)
    A = np.empty((q, pad.shape[0]), np.int64)
    for a in range(0, q, chunk):
        b = min(a + chunk, q)
        c = sum(w * (T[a:b] @ X.T - mu) / sd for T, X, mu, sd, w in terms)
        g = np.where(valid[None], c[:, safe], -np.inf)
        j = g.argmax(2)
        S[a:b] = np.take_along_axis(g, j[..., None], 2)[..., 0]
        A[a:b] = np.where(valid.any(1)[None], safe[np.arange(pad.shape[0])[None], j], -1)
    return S, A


def e3(s_main, a_main, s_win, a_win, gamma: float):
    """E3 union: ``(S, arg, from_window)``; a tie goes to the main (Omni) segment."""
    s2 = s_win + gamma
    w = s2 > s_main
    return np.where(w, s2, s_main), np.where(w, a_win, a_main), w


def ranks(S: np.ndarray, pos: np.ndarray) -> np.ndarray:
    ps = S[np.arange(len(pos)), pos]
    return 1 + (S > ps[:, None]).sum(1)


def moment_hit(arg, pos, ts, te, gt_ts, gt_te, thr: float = 0.5) -> np.ndarray:
    """tIoU of the argmax segment of the true video with the caption's GT >= ``thr``."""
    r = arg[np.arange(len(pos)), pos]
    return tiou_vec(ts[r], te[r], gt_ts, gt_te) >= thr


def metrics(rk: np.ndarray, hit: np.ndarray | None = None) -> dict:
    out = rank_metrics(rk)
    if hit is not None:
        out["VCMR R@1 (tIoU.5)"] = 100 * float(((rk == 1) & hit).mean())
    return out


def cos_stats(T, X, chunk: int = 256) -> tuple[float, float]:
    """Mean and std of every cosine T x X (z-score statistics, §4.1: dev captions x every dev segment)."""
    T, X = np.asarray(T, np.float32), np.asarray(X, np.float32)
    s = ss = 0.0
    for a in range(0, len(T), chunk):
        c = (T[a:a + chunk] @ X.T).astype(np.float64)
        s += c.sum(); ss += (c * c).sum()
    n = len(T) * len(X)
    mu = s / n
    return float(mu), float(np.sqrt(max(ss / n - mu * mu, 1e-12)))


def fold_split(val_videos, seed: int = 0) -> tuple[list[str], list[str]]:
    """F1, F2: a seeded permutation of the sorted val videos cut in two halves (§4.1)."""
    v = np.random.default_rng(seed).permutation(sorted(map(str, val_videos)))
    h = len(v) // 2
    return sorted(v[:h].tolist()), sorted(v[h:].tolist())


def cv_pick(rank_by_cand: list[np.ndarray], fold_of_query: np.ndarray) -> list[int]:
    """Candidate chosen on each fold f (index into ``rank_by_cand``): most R@1 hits over the queries of f; a tie
    goes to the earlier candidate, so the caller orders candidates by the tie rule."""
    out = []
    for f in (0, 1):
        m = fold_of_query == f
        hits = [int((r[m] == 1).sum()) for r in rank_by_cand]
        out.append(int(np.argmax(hits)))
    return out


def oof(per_cand: list[np.ndarray], pick: list[int], fold_of_query: np.ndarray) -> np.ndarray:
    """Out-of-fold values: queries of fold f take the candidate chosen on the other fold."""
    out = np.array(per_cand[pick[1]], copy=True)            # fold 0 queries <- choice made on fold 1
    m1 = fold_of_query == 1
    out[m1] = np.asarray(per_cand[pick[0]])[m1]
    return out
