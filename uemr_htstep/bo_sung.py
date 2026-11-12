"""Score columns and analyses kept from the HT-Step add-on plan for the final plan (docs/uemr_chot/UEMR_CHOT_00 §5): the
label-free z-score (C1z = UEMR-Hybrid) and the query groups. The rows and columns that are not run (R3@K_GT, R4, R5, boundary
noise, Exp 5, Context Fusion folds, S_G) are in ``uemr_htstep/_unused/bo_sung.py``.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from omni_retrieval.search import tiou_vec


# --------------------------------------------------------------------------- #
# Score columns                                                                #
# --------------------------------------------------------------------------- #

def zstats(sim: np.ndarray) -> tuple[float, float]:
    """Mean and std of every (query, segment) score; no label."""
    s = np.asarray(sim, np.float64)
    return float(s.mean()), float(s.std())


def c1z(sim_b, sim_s, fill, stats_b, stats_s) -> np.ndarray:
    """C1 with z-scores: fill segments z_(S), R6 segments z_(b)."""
    zb, zs = (sim_b - stats_b[0]) / stats_b[1], (sim_s - stats_s[0]) / stats_s[1]
    return np.where(np.asarray(fill)[None, :], zs, zb)



# --------------------------------------------------------------------------- #
# Analyses without GPU (plan §6)                                               #
# --------------------------------------------------------------------------- #

def missed_by(part: dict, captions: list[dict], thr: float = 0.3) -> np.ndarray:
    """(Q,) bool: no segment of ``part`` in the query's own video has tIoU >= ``thr`` with its GT step."""
    out = []
    for c in captions:
        segs = np.asarray(part.get(c["video_id"], []), np.float64).reshape(-1, 2)
        best = float(tiou_vec(segs[:, 0], segs[:, 1], c["gt_ts"], c["gt_te"]).max()) if len(segs) else 0.0
        out.append(best < thr)
    return np.array(out)


def shared_text(captions: list[dict]) -> np.ndarray:
    """(Q,) bool: the query's text is a step of >= 2 gallery videos."""
    vids = defaultdict(set)
    for c in captions:
        vids[c["headline"]].add(c["video_id"])
    return np.array([len(vids[c["headline"]]) >= 2 for c in captions])


def hit_table(rank_b, rank_s) -> dict:
    """2 x 2 of hit@1 for (b) and (S)."""
    b, s = np.asarray(rank_b) <= 1, np.asarray(rank_s) <= 1
    return {"b đúng, S đúng": int((b & s).sum()), "b đúng, S sai": int((b & ~s).sum()),
            "b sai, S đúng": int((~b & s).sum()), "b sai, S sai": int((~b & ~s).sum())}
