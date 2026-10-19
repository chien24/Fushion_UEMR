"""Experiments of branch A, E3 and branch B (docs/uemr2/UEMR2_AGENT_SO_BO.md §4.2, §5.3) on one in-memory ``world``.

``world`` (built by the scoring notebook from the caches):

    videos      {'val': [394], 'dev': [104], 'all': [498]}
    text        {'val' | 'dev': {'omni' [Q, 3584], 'iv2' [Q, 512], 'video_id', 'gt_ts', 'gt_te', 'caption_id'}}
    parts       {name: {'video_id', 'ts', 'te', 'e' [N, 3584] (Omni ft), 'q' [N, 512], 'h' [N, 3584] | None}}
    windows     {4 | 8 | 16: {'video_id', 'ts', 'te', 'q', 'h'}}

Sources: ``omni`` = e with text omni (column (b)); ``A`` = q with text iv2; ``B`` = h with text omni.
Every score is z-scored with dev statistics (§4.1) before two sources are added or compared.
"""

from __future__ import annotations

import numpy as np

from omni_retrieval.search import tiou_vec

from .score import cos_stats, cv_pick, e3, layout, metrics, moment_hit, oof, ranks, video_max
from .windows import DENSE_W, gap_mask, outside_fraction

SRC = {"omni": ("e", "omni"), "A": ("q", "iv2"), "B": ("h", "omni")}
BETAS = (0.0, 0.25, 0.5, 1.0, 2.0)
GAMMAS = (-1.0, -0.5, 0.0, 0.5, 1.0)
E3_W = (16, 8, 4, "multi")          # tie order: larger W (fewer vectors) first, multi last
MISS_TIOU = 0.3


# --------------------------------------------------------------------------- rows
def window_rows(world, w) -> dict:
    """Rows of ``dense_W`` (``multi`` = the three levels together)."""
    if w not in world["windows"]:                   # "multi": built once and kept (scored many times)
        ws = [world["windows"][k] for k in DENSE_W]
        world["windows"][w] = {k: (None if any(x.get(k) is None for x in ws) else np.concatenate([x[k] for x in ws]))
                               for k in ("video_id", "ts", "te", "q", "h")}
    return world["windows"][w]


def gap_rows(world, w) -> dict:
    """Windows of ``dense_W`` that pass ``gap_mask`` against the R6 segments of their video."""
    win, r6 = window_rows(world, w), world["parts"]["uniav_pred"]
    keep = np.zeros(len(win["video_id"]), bool)
    r6_by = {}
    for v, a, b in zip(r6["video_id"], r6["ts"], r6["te"]):
        r6_by.setdefault(str(v), []).append((a, b))
    for v in np.unique(win["video_id"]):
        m = win["video_id"] == v
        keep[m] = gap_mask(np.stack([win["ts"][m], win["te"][m]], 1), np.asarray(r6_by.get(str(v), []), np.float64))
    return {k: (None if x is None else x[keep]) for k, x in win.items()}


def _x(rows, src):
    return rows.get(SRC[src][0])


def _t(world, split, src):
    return world["text"][split][SRC[src][1]]


# --------------------------------------------------------------------------- z-score
def zstats(world, sources=("omni", "A", "B")) -> dict:
    """``{(source, rows name): (mu, sd)}`` over dev captions x every segment of the dev videos (§4.1)."""
    dev = set(world["videos"]["dev"])
    out = {}
    named = {**{p: r for p, r in world["parts"].items()},
             **{f"dense_{w}": window_rows(world, w) for w in (*DENSE_W, "multi")}}
    for name, rows in named.items():
        m = np.isin(rows["video_id"], list(dev))
        for s in sources:
            x = _x(rows, s)
            if x is not None and m.any():
                out[(s, name)] = cos_stats(_t(world, "dev", s), x[m])
    for w in (*DENSE_W, "multi"):                          # gap windows are scored on the dense_W scale
        for s in sources:
            if (s, f"dense_{w}") in out:
                out[(s, f"gap_{w}")] = out[(s, f"dense_{w}")]
    return out


# --------------------------------------------------------------------------- one configuration
def _pos(world, split, gallery):
    vi = {v: i for i, v in enumerate(world["videos"][gallery])}
    return np.array([vi[str(v)] for v in world["text"][split]["video_id"]])


def score_part(world, z, split, gallery, part, src=None, beta=0.0, alone=False):
    """Ranks and VCMR hits of one partition. ``alone``: only the cheap source (column (d)); else Omni z-score plus
    ``beta`` * cheap source z-score (column (e); ``beta`` 0 or ``src`` None = column (b))."""
    rows, T = world["parts"][part], world["text"][split]
    pad = layout(rows["video_id"], world["videos"][gallery])
    if alone:
        terms = [(_t(world, split, src), _x(rows, src), 0.0, 1.0, 1.0)]
    else:
        mo, so = z[("omni", part)]
        terms = [(T["omni"], rows["e"], mo, so, 1.0)]
        if src is not None and beta:
            for s, b in ((src, beta),) if isinstance(src, str) else zip(src, beta):
                if b:
                    ma, sa = z[(s, part)]
                    terms.append((_t(world, split, s), _x(rows, s), ma, sa, b))
    S, A = video_max(terms, pad)
    pos = _pos(world, split, gallery)
    rk = ranks(S, pos)
    return rk, moment_hit(A, pos, rows["ts"], rows["te"], T["gt_ts"], T["gt_te"])


def e3_scores(world, z, split, gallery, src, w, kind="gap"):
    """Video scores of R6 by Omni and of the windows by the cheap source (both z-scored), before adding gamma."""
    r6, T = world["parts"]["uniav_pred"], world["text"][split]
    win = gap_rows(world, w) if kind == "gap" else window_rows(world, w)
    mo, so = z[("omni", "uniav_pred")]
    ma, sa = z[(src, f"{kind}_{w}")]
    vids = world["videos"][gallery]
    s6, a6 = video_max([(T["omni"], r6["e"], mo, so, 1.0)], layout(r6["video_id"], vids))
    if len(win["video_id"]):
        sw, aw = video_max([(_t(world, split, src), _x(win, src), ma, sa, 1.0)], layout(win["video_id"], vids))
    else:
        sw, aw = np.full_like(s6, -np.inf), np.full_like(a6, -1)
    return (s6, a6), (sw, aw), win


def e3_ranks(world, split, gallery, pre, gamma):
    (s6, a6), (sw, aw), win = pre
    r6, T = world["parts"]["uniav_pred"], world["text"][split]
    S, A, from_w = e3(s6, a6, sw, aw, gamma)
    pos = _pos(world, split, gallery)
    rk = ranks(S, pos)
    q = np.arange(len(pos))
    row, fw = A[q, pos], from_w[q, pos]
    wts, wte = (win["ts"], win["te"]) if len(win["ts"]) else (np.zeros(1), np.zeros(1))
    ts = np.where(fw, wts[np.where(fw, row, 0)], r6["ts"][np.where(fw, 0, row)])
    te = np.where(fw, wte[np.where(fw, row, 0)], r6["te"][np.where(fw, 0, row)])
    return rk, tiou_vec(ts, te, T["gt_ts"], T["gt_te"]) >= 0.5


# --------------------------------------------------------------------------- cross-validation on val
def fold_of_queries(world, folds) -> np.ndarray:
    f1 = set(folds[0])
    return np.array([0 if str(v) in f1 else 1 for v in world["text"]["val"]["video_id"]])


def cv_beta(world, z, gallery, fq, src, parts, betas=BETAS):
    """β chosen on R6 per fold (tie -> smaller β), applied out of fold to every partition (§4.1)."""
    r6 = [score_part(world, z, "val", gallery, "uniav_pred", src, b)[0] for b in betas]
    pick = cv_pick(r6, fq)
    chosen = [betas[pick[0]], betas[pick[1]]]
    out = {}
    for p in parts:
        per = {b: score_part(world, z, "val", gallery, p, src, b) for b in set(chosen)}
        out[p] = (oof([per[chosen[0]][0], per[chosen[1]][0]], [0, 1], fq),
                  oof([per[chosen[0]][1], per[chosen[1]][1]], [0, 1], fq))
    return chosen, out


def cv_beta_ab(world, z, gallery, fq, parts, betas=BETAS):
    """A-e + B-e: Omni + βA·q + βB·h; candidates ordered by (βA + βB, βB) so a tie takes the smaller weights."""
    cands = sorted(((a, b) for a in betas for b in betas), key=lambda c: (c[0] + c[1], c[1]))
    r6 = [score_part(world, z, "val", gallery, "uniav_pred", ("A", "B"), c)[0] for c in cands]
    pick = cv_pick(r6, fq)
    chosen = [cands[pick[0]], cands[pick[1]]]
    out = {}
    for p in parts:
        per = {c: score_part(world, z, "val", gallery, p, ("A", "B"), c) for c in set(chosen)}
        out[p] = (oof([per[chosen[0]][0], per[chosen[1]][0]], [0, 1], fq),
                  oof([per[chosen[0]][1], per[chosen[1]][1]], [0, 1], fq))
    return chosen, out


def cv_e3(world, z, gallery, fq, src, kind="gap", ws=E3_W, gammas=GAMMAS):
    """(W, γ) per fold (tie -> larger W, then smaller γ), out of fold. Also the window count per video."""
    cands, res, nwin = [], [], {}
    for w in ws:
        pre = e3_scores(world, z, "val", gallery, src, w, kind)
        nwin[w] = float(np.isin(pre[2]["video_id"], world["videos"][gallery]).sum()) / len(world["videos"][gallery])
        for g in gammas:
            cands.append((w, g))
            res.append(e3_ranks(world, "val", gallery, pre, g))
    pick = cv_pick([r[0] for r in res], fq)
    chosen = [cands[pick[0]], cands[pick[1]]]
    rk = oof([res[pick[0]][0], res[pick[1]][0]], [0, 1], fq)
    hit = oof([res[pick[0]][1], res[pick[1]][1]], [0, 1], fq)
    return chosen, rk, hit, nwin


# --------------------------------------------------------------------------- text -> clip on GT clips (A-0, B-0)
def text_to_clip(world, split, src) -> dict:
    """Every caption against every GT clip of the split (3.030 on val), strict ranks."""
    gt, T = world["parts"]["gt"], world["text"][split]
    vids = set(map(str, T["video_id"]))
    m = np.isin(gt["video_id"], list(vids))
    rows = np.flatnonzero(m)
    pos = []
    for v, a, b in zip(T["video_id"], T["gt_ts"], T["gt_te"]):
        cand = rows[gt["video_id"][rows] == v]
        pos.append(cand[np.argmax(tiou_vec(gt["ts"][cand], gt["te"][cand], a, b))])
    col = {r: i for i, r in enumerate(rows)}
    pos = np.array([col[p] for p in pos])
    S = np.asarray(_t(world, split, src), np.float32) @ np.asarray(_x(gt, src)[rows], np.float32).T
    return metrics(ranks(S, pos))


def miss_labels(world) -> np.ndarray:
    """UniAV missed the caption's step: best tIoU of its GT with the video's R6 segments < 0.3 (UEMR_KET_QUA 4.9)."""
    r6, T = world["parts"]["uniav_pred"], world["text"]["val"]
    out = []
    for v, a, b in zip(T["video_id"], T["gt_ts"], T["gt_te"]):
        m = r6["video_id"] == v
        out.append(float(tiou_vec(r6["ts"][m], r6["te"][m], a, b).max()) < MISS_TIOU if m.any() else True)
    return np.array(out)


def _r6_union_by_video(world):
    r6 = world["parts"]["uniav_pred"]
    out = {}
    for v, a, b in zip(r6["video_id"], r6["ts"], r6["te"]):
        out.setdefault(str(v), []).append((a, b))
    return {v: np.asarray(x, np.float64) for v, x in out.items()}


def gap_stats(world, split) -> tuple[list[dict], dict]:
    """Descriptive only (no scoring), amendment 1: gap_W windows per video for each W, share of the video covered by
    the union of R6, and for the captions whose step UniAV missed (best tIoU with R6 < 0.3) the share with at least
    one gap_W window touching the step (tIoU > 0), at tIoU >= 0.3, and with a dense_W window at tIoU >= 0.3.
    The video duration is the end of its last dense window (windows are clipped to the mp4)."""
    vids = world["videos"][split]
    r6 = _r6_union_by_video(world)
    d4 = window_rows(world, 4)
    dur = {v: float(d4["te"][d4["video_id"] == v].max()) for v in vids}
    cover = np.array([1 - outside_fraction([[0.0, dur[v]]], r6.get(v, np.zeros((0, 2))))[0] for v in vids])
    T = world["text"][split]
    best = np.array([float(tiou_vec(r6[str(v)][:, 0], r6[str(v)][:, 1], a, b).max()) if str(v) in r6 else 0.0
                     for v, a, b in zip(T["video_id"], T["gt_ts"], T["gt_te"])])
    miss = best < MISS_TIOU
    rows = []
    for w in (*DENSE_W, "multi"):
        win, gap = window_rows(world, w), gap_rows(world, w)
        n = np.array([(gap["video_id"] == v).sum() for v in vids])
        def any_hit(rows_, thr):
            out = []
            for v, a, b in zip(T["video_id"][miss], T["gt_ts"][miss], T["gt_te"][miss]):
                m = rows_["video_id"] == v
                t = tiou_vec(rows_["ts"][m], rows_["te"][m], a, b)
                out.append(bool(m.any()) and (t.max() > thr if thr == 0 else t.max() >= thr))
            return 100 * float(np.mean(out)) if out else float("nan")
        rows.append({"tập": split, "W": w, "gap_W/video TB": float(n.mean()), "gap_W/video trung vị": float(np.median(n)),
                     "video có 0 gap_W": int((n == 0).sum()),
                     "dense_W/video TB": float(np.isin(win["video_id"], vids).sum()) / len(vids),
                     "% bước bỏ sót có gap_W chạm (tIoU>0)": any_hit(gap, 0),
                     "% bước bỏ sót có gap_W tIoU>=0.3": any_hit(gap, 0.3),
                     "% bước bỏ sót có dense_W tIoU>=0.3": any_hit(win, 0.3)})
    summary = {"tập": split, "video": len(vids), "câu": int(len(miss)), "câu bỏ sót": int(miss.sum()),
               "hợp R6 phủ video: TB %": 100 * float(cover.mean()), "trung vị %": 100 * float(np.median(cover)),
               "min %": 100 * float(cover.min()), "max %": 100 * float(cover.max())}
    return rows, summary
