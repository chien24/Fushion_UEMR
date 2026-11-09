"""Strict and multi-positive scoring (docs/uemr/HTSTEP_UEMR_OMNI_UNIAV.md §7).

A query's right answers are a set of candidates (videos, or GT clips):

* ``strict``: its own video / its own clip only;
* ``multi``: every candidate holding a step whose normalised text equals the query's.

The rank is the strict rank of the best right answer, ``1 + #(score > best right score)``: with one right answer it
is ``omni_retrieval.evaluate.video_ranks`` exactly (a query without any right answer in the gallery gets
``N + 1``, as there). R@k = a right answer in the top k, MdR = median of that rank.

VCMR R@1: rank 1, and the best-scoring segment of the best right video (``SegmentIndex.score_matrix``'s argmax) has
tIoU >= 0.5 with the query's own step (strict) or with a step of that video that has the query's headline (multi).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from omni_retrieval.search import tiou_vec
from uemr2.uemr2_core import maxsim

MODES = ("strict", "multi")


def right_answers(q_keys, q_heads, cand_keys, cand_heads, mode: str) -> np.ndarray:
    """``[Q, N]`` bool. ``q_keys[i]``: the query's own candidate (video id / ``gt_seg_key``); ``cand_heads[j]``:
    the set of headlines candidate j holds."""
    if mode == "strict":
        return np.asarray(q_keys)[:, None] == np.asarray(cand_keys)[None, :]
    if mode == "multi":
        out = np.array([[h in hs for hs in cand_heads] for h in q_heads], dtype=bool).reshape(len(q_heads), -1)
        return out | right_answers(q_keys, q_heads, cand_keys, cand_heads, "strict")
    raise ValueError(f"mode must be one of {MODES}")


def ranks_and_best(scores: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(rank of the best right answer, its column)`` per query; rank ``N + 1`` when there is none."""
    masked = np.where(right, scores, -np.inf)
    col = masked.argmax(1)
    best = masked[np.arange(len(scores)), col]
    ranks = 1 + (scores > best[:, None]).sum(1)
    return np.where(right.any(1), ranks, scores.shape[1] + 1), col



def row_key(video, ts, te) -> tuple:
    """Key of a segment across arrays that hold its times in float32 or float64."""
    return str(video), round(float(ts), 2), round(float(te), 2)


def student_rows(index, video_id, ts, te, emb) -> np.ndarray:
    """The student vectors of ``index``'s segments in ``index``'s row order (``run_uniav student`` keeps its own order)."""
    of = {row_key(v, a, b): e for v, a, b, e in zip(video_id, ts, te, emb)}
    return np.stack([of[row_key(v, a, b)] for v, a, b in zip(index.seg_video, index.ts, index.te)]).astype(np.float32)


def column_sim(col: str, q_omni, q_student, e_omni, e_student, beta: float = 0.1, fill=None) -> np.ndarray:
    """(Q, N) score of every segment for one column of plan §6: ``b`` Omni, ``S`` student, ``U`` b + beta S, ``C1`` S on the
    fill segments of R8 (``fill`` (N,) bool) and b on the R6 ones."""
    if col == "b":
        return q_omni @ e_omni.T
    if col == "S":
        return q_student @ e_student.T
    if col == "U":
        return q_omni @ e_omni.T + beta * (q_student @ e_student.T)
    if col == "C1":
        return np.where(np.asarray(fill)[None, :], q_student @ e_student.T, q_omni @ e_omni.T)
    raise ValueError(f"unknown column {col}")

def headlines_by_video(captions: list[dict]) -> dict[str, list[tuple[float, float, str]]]:
    """``video_id -> [(ts, te, headline)]`` of every query step (clamped times)."""
    out = defaultdict(list)
    for c in captions:
        out[c["video_id"]].append((float(c["gt_ts"]), float(c["gt_te"]), c["headline"]))
    return dict(out)


def text_to_video(index, text: dict, captions: list[dict], mode: str, agg: str = "max", sim=None) -> dict:
    """Ranks and VCMR hits of every query (``captions`` in the order of ``text``) on a ``SegmentIndex``.

    ``sim`` (Q, N rows of ``index``): the score of every segment when it is not ``q @ index.emb`` (columns (U), C1 of §6);
    the video score is still MaxSim.
    """
    if [c["caption_id"] for c in captions] != [str(x) for x in text["caption_id"]]:
        raise ValueError("captions and text.npz are not in the same order")
    steps = headlines_by_video(captions)
    heads = [c["headline"] for c in captions]
    if sim is None:
        scores, best = index.score_matrix(text["emb"], agg=agg)
    else:
        scores, best = maxsim(sim, np.searchsorted(index.videos, index.seg_video), index.n_videos)
    right = right_answers(text["video_id"], heads, index.videos, [{h for *_, h in steps.get(v, [])}
                                                                    for v in index.videos], mode)
    ranks, col = ranks_and_best(scores, right)
    q = np.arange(len(ranks))
    row = best[q, col]
    ts, te = index.ts[row], index.te[row]
    if mode == "strict":
        hit = tiou_vec(ts, te, text["gt_ts"], text["gt_te"]) >= 0.5
    else:
        hit = np.zeros(len(ranks), dtype=bool)
        for i, j in enumerate(col):
            same = [(a, b) for a, b, h in steps.get(str(index.videos[j]), []) if h == heads[i]]
            hit[i] = bool(same) and bool((tiou_vec(ts[i], te[i], np.array([a for a, _ in same]),
                                                   np.array([b for _, b in same])) >= 0.5).any())
    return {"ranks": ranks, "vcmr": (ranks == 1) & hit}


def gt_clips(segments_npz) -> tuple[np.ndarray, np.ndarray]:
    """``(seg_key, emb)`` of the distinct ``gt`` clips, the gallery of ``evaluate.text_to_clip`` ('subset')."""
    with np.load(segments_npz) as b:
        sel = b["partition"] == "gt"
        keys, emb = b["seg_key"][sel], b["emb"][sel].astype(np.float32)
    uniq, first = np.unique(keys, return_index=True)
    return uniq, emb[first]


def text_to_clip(text: dict, captions: list[dict], clip_keys, clip_emb, mode: str) -> np.ndarray:
    """Rank of the best right GT clip for every query (gallery = every distinct GT clip of the gallery videos)."""
    heads_of = defaultdict(set)
    for c in captions:
        heads_of[c["gt_seg_key"]].add(c["headline"])
    right = right_answers(text["gt_seg_key"], [c["headline"] for c in captions], clip_keys,
                          [heads_of[k] for k in clip_keys], mode)
    return ranks_and_best(np.asarray(text["emb"], np.float32) @ np.asarray(clip_emb, np.float32).T, right)[0]
