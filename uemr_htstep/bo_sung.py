"""Rows, score columns and analyses of the HT-Step add-on plan (docs/uemr/HTSTEP_BO_SUNG.md), on top of the main run.

Rows reuse the YC2 code of ``omni_retrieval.partitions_uemr``: R4 = ``rand_m`` (K contiguous segments >= d_min; d_min +
Dirichlet(1) lengths, the same law as K - 1 uniform cuts kept only when every piece is >= d_min), R5 = ``kmedoids_m``
(cosine K-Medoids on the level-0 features, 2 s window per medoid), R7n = ``perturb`` with directions U(-1, 1) (centre +
a L u_c, length x exp(a u_s)). Every segment list goes through ``_round`` like the main rows.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from omni_retrieval.bo_sung import fill_gaps
from omni_retrieval.partitions import uniform
from omni_retrieval.partitions_uemr import _round, kmedoids_m, perturb, rand_m, video_rng
from omni_retrieval.search import tiou_vec
from uniav_app.postprocess import postprocess_uemr

from .metrics import row_key
from .partitions import pick_theta, r6

R4_SEEDS = (0, 1, 2)
NOISE_TARGETS = (0.9, 0.7, 0.5, 0.3)
NOISE_TOL = 0.01


def noise_name(target: float) -> str:
    return f"r7n{int(round(target * 100))}"


# --------------------------------------------------------------------------- #
# Rows (plan §3, §4, §7)                                                       #
# --------------------------------------------------------------------------- #

def r3_kgt(gt: dict, durations: dict) -> dict:
    """K_GT equal segments per video."""
    return {v: _round(uniform(durations[v], len(gt[v])), durations[v]) for v in sorted(gt)}


def r4(k_pred: dict, durations: dict, seed: int, d_min: float) -> dict:
    return {v: _round(rand_m(durations[v], k_pred[v], video_rng(seed, f"{v}#r4"), d_min), durations[v]) for v in sorted(k_pred)}


def r5(feats: dict, k_pred: dict, durations: dict, seed: int, half: float = 1.0) -> tuple[dict, dict]:
    """``feats[v] = (Z (T, D), t_sec (T,))``. ``({v: segments}, {'not converged': [...], 'iterations max': n})``."""
    out, bad, it_max = {}, [], 0
    for v in sorted(k_pred):
        z, t = feats[v]
        segs, it, ok = kmedoids_m(z, t, durations[v], k_pred[v], video_rng(seed, f"{v}#r5"), half)
        out[v] = _round(segs, durations[v])
        it_max = max(it_max, it)
        if not ok:
            bad.append(v)
    return out, {"not converged": bad, "iterations max": it_max}


def noise_dirs(gt: dict, seed: int = 0) -> dict:
    """``(u_c, u_s) ~ U(-1, 1)`` per GT segment, drawn once and shared by every level."""
    return {v: video_rng(seed, f"{v}#r7n").uniform(-1.0, 1.0, (len(gt[v]), 2)) for v in sorted(gt)}


def noisy(gt: dict, durations: dict, dirs: dict, a: float, d_min: float) -> dict:
    """``{v: [noisy segment of GT segment i, ...]}`` in GT order (the pairing the tIoU target is measured on)."""
    return {v: [perturb([s], durations[v], [d], a, d_min)[0] for s, d in zip(gt[v], dirs[v])] for v in sorted(gt)}


def paired_tiou(gt: dict, part: dict) -> float:
    """Mean tIoU of every noisy segment with its own GT segment."""
    a = np.array([s for v in sorted(gt) for s in gt[v]], np.float64)
    b = np.array([s for v in sorted(gt) for s in part[v]], np.float64)
    inter = np.clip(np.minimum(a[:, 1], b[:, 1]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    union = np.maximum(a[:, 1], b[:, 1]) - np.minimum(a[:, 0], b[:, 0])
    return float(np.mean(np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)))


def calibrate(gt: dict, durations: dict, dirs: dict, target: float, d_min: float, a_max: float = 10.0,
              iters: int = 40, tol: float = NOISE_TOL) -> tuple[float, float, dict]:
    """Bisection on ``a`` for mean paired tIoU = ``target``; ``(a, achieved, rows sorted by time)``. Not strictly monotone
    (clipping), so the result is checked against ``tol``."""
    lo, hi = 0.0, a_max
    if paired_tiou(gt, noisy(gt, durations, dirs, hi, d_min)) > target:
        raise ValueError(f"a = {a_max} still gives mean tIoU > {target}")
    for _ in range(iters):
        mid = (lo + hi) / 2
        if paired_tiou(gt, noisy(gt, durations, dirs, mid, d_min)) > target:
            lo = mid
        else:
            hi = mid
    a = (lo + hi) / 2
    part = noisy(gt, durations, dirs, a, d_min)
    got = paired_tiou(gt, part)
    if abs(got - target) > tol:
        raise ValueError(f"target {target}: a {a:.4f} gives {got:.4f} (tolerance {tol})")
    return a, got, {v: sorted(s) for v, s in part.items()}


def r6k(props_pred: dict, durations: dict, target: float, p: dict) -> tuple[float, float, dict]:
    """Option RUN_R6K: the θ rule of the main plan §4.2 with another target K. ``(θ, mean K, rows)``."""
    a, b, s = p["theta_grid"]
    theta, k, _ = pick_theta(props_pred, durations, target, p["k_max"], p["d_min"], p["merge_tiou"], np.round(np.arange(a, b, s), 6))
    return float(theta), float(k), {v: r6(props_pred[v], durations[v], theta, p["k_max"], p["d_min"], p["merge_tiou"]) for v in sorted(durations)}


def r7fill(gt: dict, durations: dict, l_min: float, l_fill: float) -> dict:
    """Option RUN_R7FILL: GT segments + the R8 gap windows."""
    return {v: _round(fill_gaps(gt[v], durations[v], l_min, l_fill)[0], durations[v]) for v in sorted(gt)}


def r6_conf(props_pred: dict, durations: dict, theta: float, p: dict) -> dict:
    """``{(video, ts, te) rounded 2: conf}`` of the R6 segments (the Exp 5 term ``log conf``)."""
    out = {}
    for v in sorted(durations):
        for s in postprocess_uemr(props_pred[v], durations[v], theta, p["k_max"], p["d_min"], p["merge_tiou"]):
            seg = _round([[round(float(s["ts"]), 3), round(float(s["te"]), 3)]], durations[v])[0]
            out[row_key(v, seg[0], seg[1])] = float(s["conf"])     # the key of metrics.row_key (an end clipped to the video has 3 decimals)
    return out


# --------------------------------------------------------------------------- #
# Folds of Context Fusion (plan §5.2)                                          #
# --------------------------------------------------------------------------- #

def article_folds(video_article: dict, n_queries: dict, seed: int = 0) -> dict:
    """``{video: 'A' | 'B'}``: every video of an article in one fold; articles in random order (seed), heaviest first (number of
    queries), each to the fold with fewer queries so far."""
    by_art = defaultdict(list)
    for v, art in video_article.items():
        by_art[int(art)].append(v)
    arts = list(by_art)
    order = np.random.default_rng(seed).permutation(len(arts))
    weight = {a: sum(n_queries.get(v, 0) for v in by_art[a]) for a in arts}
    arts = sorted((arts[i] for i in order), key=lambda a: -weight[a])          # stable: ties keep the random order
    total, out = {"A": 0, "B": 0}, {}
    for a in arts:
        f = "A" if total["A"] <= total["B"] else "B"
        total[f] += weight[a]
        out.update({v: f for v in by_art[a]})
    return out


# --------------------------------------------------------------------------- #
# Score columns (plan §5.1, §5.3)                                              #
# --------------------------------------------------------------------------- #

def zstats(sim: np.ndarray) -> tuple[float, float]:
    """Mean and std of every (query, segment) score; no label."""
    s = np.asarray(sim, np.float64)
    return float(s.mean()), float(s.std())


def c1z(sim_b, sim_s, fill, stats_b, stats_s) -> np.ndarray:
    """C1 with z-scores: fill segments z_(S), R6 segments z_(b)."""
    zb, zs = (sim_b - stats_b[0]) / stats_b[1], (sim_s - stats_s[0]) / stats_s[1]
    return np.where(np.asarray(fill)[None, :], zs, zb)


def global_bonus(q: np.ndarray, index, global_index, gamma: float) -> np.ndarray:
    """``gamma * cos(q, e_g)`` per (query, video of ``index``): the second term of S_G."""
    g = {v: e for v, e in zip(global_index.seg_video, global_index.emb)}
    return gamma * (np.asarray(q, np.float32) @ np.stack([g[v] for v in index.videos]).T)


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
