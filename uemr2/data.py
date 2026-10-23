"""Fushion caches -> the arrays of ``uemr2_core`` (spec UEMR_ALIGN_FINAL.md 3-5). numpy only.

Gallery (one split, one row R0 … R8):  ``seg_key``, ``video`` (M,) str, ``videos`` (V,) sorted ids, ``seg_vid`` (M,) index
into ``videos``, ``segs`` (M, 2) float32 seconds, ``E`` (M, 3584) float32 Omni ft vectors as stored (fp16, not re-normalised:
this is what UEMR scores, so the ranks are reproduced exactly), ``is_fill`` (R8 only).
Rows are ordered like ``omni_retrieval.search.SegmentIndex`` (by video, then start), and ``scores`` is the product of
``SegmentIndex.score_matrix``, so MaxSim of ``uemr2_core`` gives UEMR's ranks bit for bit.

Queries: ``caption_id``, ``video``, ``sentence``, ``gt`` (C, 2) float32, ``gt_seg_key``, ``T`` (C, 3584).
Train (spec 4.3): GT and R6 rows of the trainsplit, the caption of each row, the pool of unique captions.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np

from omni_retrieval.evaluate import QUERY_BLOCK, load_index, load_text, video_ranks
from omni_retrieval.search import tiou_vec

# spec 7.1 row -> partition in segments.npz (R8 lives in the bo_sung_val cache, the others in the AZ cache)
ROWS = {"R0": "global", "R3": "uni_M", "R6": "uniav_pred", "R2": "uni_chunk", "R8": "r8",
        "R6∪R2": "uniav_pred+uni_chunk", "R7": "gt"}
# logged numbers the reproduction must hit (configs/uemr_yc2_final.json reference; dev: uemr_yc2_bo_sung_dev ô 12)
REF = {"val": {"R0": {"R@1": 8.32}, "R3": {"R@1": 18.02}, "R6": {"R@1": 24.52}, "R2": {"R@1": 27.16},
               "R8": {"R@1": 27.39, "R@5": 56.86, "VCMR@1": 14.32}, "R6∪R2": {"R@1": 29.90}, "R7": {"R@1": 37.52}},
       "dev": {"R8": {"R@1": 46.3130659767141}, "R6": {"R@1": 43.855109961190166}, "R2": {"R@1": 49.417852522639066}},
       "all": {"R8": {"R@1": 25.05}, "R6": {"R@1": 22.41}, "R2": {"R@1": 24.36}, "R6∪R2": {"R@1": 27.06}}}


@lru_cache(maxsize=4)
def _segments(db) -> dict:
    """``<db>/segments.npz`` read once (it holds every partition, ~1 GB): one gallery per row and split reads it."""
    with np.load(f"{db}/segments.npz") as b:
        return {k: b[k] for k in ("seg_key", "video_id", "partition", "ts", "te", "emb")}


def gallery(db, partition: str, videos=None) -> dict:
    """Rows of one partition of ``<db>/segments.npz``, restricted to ``videos`` (None: every video)."""
    b = _segments(str(db))
    m = b["partition"] == partition
    if videos is not None:
        m &= np.isin(b["video_id"], list(videos))
    if not m.any():
        raise KeyError(f"partition {partition!r} has no row in {db}/segments.npz for these videos")
    vid, ts, te = b["video_id"][m], b["ts"][m], b["te"][m]
    order = np.lexsort((ts, vid))                # = SegmentIndex.__init__
    g = {"seg_key": b["seg_key"][m][order], "video": vid[order],
         "segs": np.stack((ts[order], te[order]), 1).astype(np.float32),
         "E": np.asarray(b["emb"][m], dtype=np.float32)[order]}
    g["videos"] = np.unique(g["video"])
    g["seg_vid"] = np.searchsorted(g["videos"], g["video"])
    g["partition"] = partition
    return g


def mark_fill(g8: dict, g6: dict) -> dict:
    """``is_fill``: R8 rows that are gap windows, not R6 segments (spec 7.2 C1). R8 = sorted(R6 + fills)
    (``omni_retrieval.bo_sung.fill_gaps``), so the R6 rows of a video are its rows whose seg_key is an R6 seg_key;
    asserts that every video keeps exactly its K_pred R6 segments."""
    r6 = defaultdict(Counter)
    for v, k in zip(g6["video"], g6["seg_key"]):
        r6[v][k] += 1
    left = {v: c.copy() for v, c in r6.items()}
    fill = np.ones(len(g8["seg_key"]), bool)
    for i, (v, k) in enumerate(zip(g8["video"], g8["seg_key"])):
        if left.get(v, {}).get(k, 0) > 0:
            left[v][k] -= 1
            fill[i] = False
    kept = Counter(g8["video"][~fill].tolist())
    bad = [v for v in set(r6) | set(g8["video"].tolist()) if kept.get(v, 0) != sum(r6[v].values())]
    assert not bad, f"{len(bad)} videos: R8 rows that are R6 segments != K_pred of R6, e.g. {bad[:3]}"
    g8["is_fill"] = fill
    return g8


def queries(db, name: str = "text.npz", videos=None) -> dict:
    t = load_text(db, name)
    m = np.ones(len(t["emb"]), bool) if videos is None else np.isin(t["video_id"], list(videos))
    return {"caption_id": t["caption_id"][m], "video": t["video_id"][m], "sentence": t["sentence"][m],
            "gt": np.stack((t["gt_ts"][m], t["gt_te"][m]), 1).astype(np.float32),
            "gt_seg_key": t["gt_seg_key"][m], "T": t["emb"][m]}


def cap_vid(q: dict, g: dict) -> np.ndarray:
    """Index of every query's video in the gallery's sorted video list (every query video must be there)."""
    i = np.searchsorted(g["videos"], q["video"])
    ok = (i < len(g["videos"])) & (g["videos"][np.minimum(i, len(g["videos"]) - 1)] == q["video"])
    assert ok.all(), f"{int((~ok).sum())} queries whose video has no segment in the gallery"
    return i


def scores(t: np.ndarray, x: np.ndarray, chunk: int = QUERY_BLOCK) -> np.ndarray:
    """(C, M) cosines, the same product as ``SegmentIndex.score_matrix`` (``q @ emb.T`` per block of 512)."""
    t = np.asarray(t, dtype=np.float32)
    x = np.ascontiguousarray(x, dtype=np.float32)
    out = np.empty((len(t), len(x)), np.float32)
    for s in range(0, len(t), chunk):
        out[s:s + chunk] = t[s:s + chunk] @ x.T
    return out


def reference(db, partition: str, text: dict, videos=None) -> dict:
    """UEMR's own ranks (``load_index`` + ``video_ranks``, bo_sung val ô 3 ``ranks``): rank and VCMR hit per query.
    ``text`` is a ``load_text`` dict (all its queries)."""
    idx = load_index(db, partition, videos)
    r = video_ranks(idx, text)
    hit = (r["ranks"] == 1) & r["covered"] & (tiou_vec(r["best_ts"], r["best_te"], text["gt_ts"], text["gt_te"]) >= 0.5)
    return {"rank": r["ranks"], "vcmr": hit, "vec/video": float(np.mean(idx.counts))}


def take(cache: dict, keys) -> np.ndarray:
    """Stack ``cache[k]`` for every key (a q / u / h cache keyed by seg_key or caption_id)."""
    miss = [k for k in keys if k not in cache]
    assert not miss, f"{len(miss)} keys missing from the vector cache, e.g. {miss[:3]}"
    return np.stack([cache[k] for k in keys])


# --------------------------------------------------------------------------- train (spec 4.3)
def train_rows(db, partitions=("gt", "uniav_pred")) -> dict:
    """GT and R6 rows of ``omni_cache_train/ft/<hash>/segments.npz`` (E as stored, fp16 -> fp32)."""
    with np.load(f"{db}/segments.npz") as b:
        m = np.isin(b["partition"], list(partitions))
        return {"seg_key": b["seg_key"][m], "video": b["video_id"][m], "partition": b["partition"][m],
                "segs": np.stack((b["ts"][m], b["te"][m]), 1).astype(np.float32),
                "E": np.asarray(b["emb"][m], dtype=np.float32)}


def iv2_key(caption_id: str) -> str:
    """Omni caption id ``<vid>_<i>`` -> key ``<vid>#<i>`` of caption_emb_iv2j*.npz (i = annotation index)."""
    v, i = caption_id.rsplit("_", 1)
    return f"{v}#{int(i)}"


def train_set(rows: dict, caps: dict, iv2: dict, thr: float = 0.5) -> dict:
    """Caption of every train row and the caption pool.

    GT row: its own caption (``gt_seg_key`` == row seg_key; captions sharing a clip are handed out in file order).
    R6 row: the caption of its video with the largest tIoU if >= ``thr``, else -1.
    Pool: unique sentences of ``caps`` in first-seen order; ``T`` = Omni text vector and ``V`` = IV2 vector
    (``iv2[<vid>#<i>]``) of the sentence's first caption. ``caps`` is a ``queries`` dict of the train text.npz."""
    pool, first = {}, []
    for j, s in enumerate(caps["sentence"].tolist()):
        if s not in pool:
            pool[s] = len(first)
            first.append(j)
    cap_pool = np.array([pool[s] for s in caps["sentence"].tolist()])
    by_key = defaultdict(list)
    for j, k in enumerate(caps["gt_seg_key"].tolist()):
        by_key[k].append(j)
    by_vid = defaultdict(list)
    for j, v in enumerate(caps["video"].tolist()):
        by_vid[v].append(j)
    cap = np.full(len(rows["seg_key"]), -1, np.int64)
    used = Counter()
    for i, (p, k, v) in enumerate(zip(rows["partition"].tolist(), rows["seg_key"].tolist(), rows["video"].tolist())):
        if p == "gt":
            js = by_key.get(k, [])
            assert used[k] < len(js), f"GT row {k} has no caption left"
            cap[i] = cap_pool[js[used[k]]]
            used[k] += 1
        else:
            js = by_vid.get(v, [])
            if js:
                ti = tiou_vec(rows["segs"][i, 0], rows["segs"][i, 1], caps["gt"][js, 0], caps["gt"][js, 1])
                if ti.max() >= thr:
                    cap[i] = cap_pool[js[int(ti.argmax())]]
    gt = rows["partition"] == "gt"
    assert sum(used.values()) == int(gt.sum()) == len(caps["sentence"]), "GT rows and captions do not pair up"
    keys = [iv2_key(c) for c in caps["caption_id"][first].tolist()]
    miss = [k for k in keys if k not in iv2]
    assert not miss, f"{len(miss)} pool captions without an IV2 vector, e.g. {miss[:3]}"
    videos = np.unique(rows["video"])
    return {"cap": cap, "vid": np.searchsorted(videos, rows["video"]), "videos": videos,
            "pool_sentence": caps["sentence"][first], "T_pool": caps["T"][first],
            "V_pool": np.stack([np.asarray(iv2[k], np.float32) for k in keys]),
            "counts": {"rows": len(cap), "gt rows": int(gt.sum()), "r6 rows": int((~gt).sum()),
                       "r6 with caption": int((cap[~gt] >= 0).sum()), "r6 without caption": int((cap[~gt] < 0).sum()),
                       "captions": len(cap_pool), "unique captions (pool)": len(first)}}


def load_iv2(path) -> dict:
    """``{<vid>#<i>: vector}`` of caption_emb_iv2j*.npz (already centred on its mean and L2-normalised)."""
    z = np.load(path, allow_pickle=True)   # own file of the Athena repo; keys / sentences may be object arrays
    return {str(k): e for k, e in zip(z["keys"], z["emb"])}


# --------------------------------------------------------------------------- analysis helpers
def missed(q: dict, g6: dict, thr: float) -> np.ndarray:
    """Per query: no R6 segment of its video reaches tIoU ``thr`` with its GT step (bo_sung val ô 5 ``MISSED_Q``)."""
    out = np.empty(len(q["video"]), bool)
    for j, (v, (a, b)) in enumerate(zip(q["video"], q["gt"])):
        s = g6["segs"][g6["video"] == v]
        out[j] = not len(s) or float(tiou_vec(s[:, 0], s[:, 1], a, b).max()) < thr
    return out


def text_to_clip(T: np.ndarray, G: np.ndarray) -> dict:
    """Text -> clip, one clip per caption (row i of G = caption i's own GT clip), the 'eval' gallery of
    ``omni_retrieval.evaluate.text_to_clip``."""
    rank = np.empty(len(T), np.int64)
    for s in range(0, len(T), QUERY_BLOCK):
        sim = scores(T[s:s + QUERY_BLOCK], G)
        q = np.arange(s, min(s + QUERY_BLOCK, len(T)))
        rank[q] = 1 + (sim > sim[q - s, q][:, None]).sum(1)
    return {"R@1": 100 * float((rank <= 1).mean()), "R@5": 100 * float((rank <= 5).mean()),
            "R@10": 100 * float((rank <= 10).mean()), "MdR": float(np.median(rank)), "clips": len(G)}


def clip_rows(q: dict, g7: dict) -> np.ndarray:
    """Row of every query's own GT clip in a ``gt`` gallery (by ``gt_seg_key``)."""
    row = {k: i for i, k in enumerate(g7["seg_key"].tolist())}
    miss = [k for k in q["gt_seg_key"].tolist() if k not in row]
    assert not miss, f"{len(miss)} captions without their GT clip"
    return np.array([row[k] for k in q["gt_seg_key"].tolist()])
