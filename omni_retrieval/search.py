"""Multi-vector video index with late-interaction scoring (UEMR §4 D). numpy only.

    index = SegmentIndex.load(cache_dir, partition="gt")
    hits  = index.search(q_vec, top_k=10, agg="max")
    # [{video_id, score, best_segment: (ts, te), segment_scores: [...]}, ...]

Video score ``S(q, V) = agg_i cos(q, e_i)`` over the segments of ``V``; the timestamp
is the segment at ``argmax_i``. Scoring is one ``Q x E^T`` product followed by a gather
into a ``[Q, V, K_max]`` tensor (segments padded with -inf), reduced along the last
axis -- no loop over videos.

``agg``: ``max`` (MaxSim, inference default), ``topk_mean`` (mean of the k best
segments), ``lse`` (``tau * log mean exp(s / tau)``, the training score of §4 D).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

AGGS = ("max", "topk_mean", "lse")


class SegmentIndex:
    def __init__(self, emb: np.ndarray, video_id, ts, te, name: str = ""):
        order = np.lexsort((np.asarray(ts), np.asarray(video_id)))  # by video, then time
        self.emb = np.asarray(emb, dtype=np.float32)[order]
        self.seg_video = np.asarray(video_id)[order]
        self.ts = np.asarray(ts, dtype=np.float32)[order]
        self.te = np.asarray(te, dtype=np.float32)[order]
        self.name = name
        self.videos, self.offsets, self.counts = np.unique(self.seg_video, return_index=True,
                                                           return_counts=True)
        self.video_index = {v: i for i, v in enumerate(self.videos)}
        # pad[v, j] = row of the j-th segment of video v, or -1
        k_max = int(self.counts.max()) if len(self.counts) else 0
        j = np.arange(k_max)
        self.pad = np.where(j[None, :] < self.counts[:, None], self.offsets[:, None] + j[None, :], -1)

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, cache_dir, partition: str = "gt") -> "SegmentIndex":
        """Load one partition from ``segments.npz``; ``event_single`` = mean-pooled ``gt``."""
        if partition == "event_single":
            return cls.load(cache_dir, "gt").mean_pooled("event_single")
        with np.load(Path(cache_dir) / "segments.npz") as blob:
            sel = blob["partition"] == partition
            if not sel.any():
                have = sorted(set(blob["partition"].tolist()))
                raise KeyError(f"partition {partition!r} not in segments.npz (have {have})")
            return cls(blob["emb"][sel], blob["video_id"][sel], blob["ts"][sel], blob["te"][sel], partition)

    def mean_pooled(self, name: str = "event_single") -> "SegmentIndex":
        """One vector per video: L2-normalised mean of its segments (R1 from R7/R6)."""
        sums = np.add.reduceat(self.emb, self.offsets, axis=0)
        emb = sums / np.clip(np.linalg.norm(sums, axis=1, keepdims=True), 1e-12, None)
        ts = np.minimum.reduceat(self.ts, self.offsets)
        te = np.maximum.reduceat(self.te, self.offsets)
        return SegmentIndex(emb, self.videos, ts, te, name)

    @property
    def n_videos(self) -> int:
        return len(self.videos)

    # ------------------------------------------------------------------ #
    def score_matrix(self, q: np.ndarray, agg: str = "max", k: int = 2, tau: float = 0.1,
                     chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(scores [Q, V], best [Q, V])``; ``best`` is the global row of the argmax segment."""
        if agg not in AGGS:
            raise ValueError(f"agg must be one of {AGGS}")
        q = np.atleast_2d(np.asarray(q, dtype=np.float32))
        valid = self.pad >= 0
        safe = np.where(valid, self.pad, 0)
        v_idx = np.arange(self.n_videos)[None, :]
        scores = np.empty((len(q), self.n_videos), dtype=np.float32)
        best = np.empty((len(q), self.n_videos), dtype=np.int64)
        for s in range(0, len(q), chunk):
            sim = q[s:s + chunk] @ self.emb.T                       # [q, N]
            g = np.where(valid[None], sim[:, safe], -np.inf)         # [q, V, K]
            best[s:s + chunk] = safe[v_idx, g.argmax(axis=2)]
            if agg == "max":
                scores[s:s + chunk] = g.max(axis=2)
            elif agg == "topk_mean":
                kk = np.minimum(k, self.counts)                      # [V]
                top = -np.sort(-g, axis=2)[:, :, :max(k, 1)]
                mask = np.arange(top.shape[2])[None, None] < kk[None, :, None]
                scores[s:s + chunk] = np.where(mask, top, 0).sum(2) / kk[None]
            else:  # lse, normalised by |C(V)|
                m = g.max(axis=2, keepdims=True)
                lse = m[..., 0] + tau * np.log(np.exp((g - m) / tau).sum(axis=2))
                scores[s:s + chunk] = lse - tau * np.log(self.counts)[None]
        return scores, best

    def search(self, q_vec: np.ndarray, top_k: int = 10, agg: str = "max", **kw) -> list[dict]:
        """Top-k videos for one query vector."""
        q = np.asarray(q_vec, dtype=np.float32).reshape(1, -1)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        scores, best = self.score_matrix(q, agg=agg, **kw)
        order = np.argsort(-scores[0], kind="stable")[:top_k]
        seg_sim = (q @ self.emb.T)[0]
        hits = []
        for v in order:
            rows = range(self.offsets[v], self.offsets[v] + self.counts[v])
            b = int(best[0, v])
            hits.append({
                "video_id": str(self.videos[v]),
                "score": float(scores[0, v]),
                "best_segment": (float(self.ts[b]), float(self.te[b])),
                "segment_scores": [(float(self.ts[r]), float(self.te[r]), float(seg_sim[r])) for r in rows],
            })
        return hits


class EventIndex:
    """Single-event retrieval: one vector per GT event, the answer is an event.

        index = EventIndex.load(cache_dir)                 # reads events.npz
        hits  = index.search(q_vec, top_k=5)
        # [{rank, score, video_id, ts, te, caption, seg_key}, ...]

    Score = cosine(query, event). No grouping by video: two events of the same video
    are two separate results.
    """

    def __init__(self, emb, seg_key, video_id, ts, te, caption, split=None):
        self.emb = np.asarray(emb, dtype=np.float32)
        self.seg_key = np.asarray(seg_key)
        self.video_id = np.asarray(video_id)
        self.ts = np.asarray(ts, dtype=np.float32)
        self.te = np.asarray(te, dtype=np.float32)
        self.caption = np.asarray(caption)
        self.split = np.asarray(split) if split is not None else np.full(len(self.emb), "")
        self.row_of = {k: i for i, k in enumerate(self.seg_key)}

    @classmethod
    def load(cls, cache_dir) -> "EventIndex":
        with np.load(Path(cache_dir) / "events.npz") as b:
            return cls(b["emb"], b["seg_key"], b["video_id"], b["ts"], b["te"], b["caption"],
                       b["split"] if "split" in b.files else None)

    def __len__(self) -> int:
        return len(self.emb)

    def scores(self, q: np.ndarray) -> np.ndarray:
        """Cosine of every query against every event, ``[Q, N]``."""
        q = np.atleast_2d(np.asarray(q, dtype=np.float32))
        q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
        return q @ self.emb.T

    def search(self, q_vec: np.ndarray, top_k: int = 5) -> list[dict]:
        s = self.scores(q_vec)[0]
        k = min(top_k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top], kind="stable")]
        return [{"rank": r + 1, "score": float(s[i]), "video_id": str(self.video_id[i]),
                 "ts": float(self.ts[i]), "te": float(self.te[i]), "caption": str(self.caption[i]),
                 "seg_key": str(self.seg_key[i])} for r, i in enumerate(top)]


def tiou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def tiou_vec(ts, te, gs, ge) -> np.ndarray:
    inter = np.clip(np.minimum(te, ge) - np.maximum(ts, gs), 0, None)
    union = np.maximum(te, ge) - np.minimum(ts, gs)
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)
