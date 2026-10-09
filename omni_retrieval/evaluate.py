"""Quick text->video evaluation on the encoded subset (UEMR Exp 1, columns (a)/(b)).

Queries: every val caption in ``text.npz``. Positive: the caption's source video.
Per partition: R@1/5/10 (%), MedR, MnR, vectors/video, and a simple moment metric --
among queries whose top-1 video is right, the share whose argmax segment has
tIoU >= 0.5 with the caption's GT event (``mom@.5|top1``); ``VCMR R@1`` is the same
over all queries (right video *and* right moment). ``event_single`` is the mean of the
``gt`` vectors per video (R1).

Ranks are strict: ``1 + #(score > positive)``, as ``omniretriever.evaluation.metrics``.

``sanity_check`` must pass before the table is trusted: text -> clip on the ``gt``
segments, which on the full val subset is the exact computation of
``eval_youcookii.py`` and so must reproduce its t2m R@1.

    python -m omni_retrieval.evaluate <cache_dir> [--partitions gt global ...] [--ref best_val.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .search import SegmentIndex, tiou_vec

DEFAULT_PARTITIONS = ("global", "event_single", "uni_M_gt", "gt")
QUERY_BLOCK = 512   # queries per Q x E^T block


class SanityCheckFailed(RuntimeError):
    pass


def load_text(cache_dir) -> dict[str, np.ndarray]:
    with np.load(Path(cache_dir) / "text.npz") as blob:
        out = {k: blob[k] for k in blob.files}
    out["emb"] = out["emb"].astype(np.float32)
    return out


def rank_metrics(ranks: np.ndarray) -> dict[str, float]:
    ranks = np.asarray(ranks)
    return {"R@1": 100 * float(np.mean(ranks <= 1)), "R@5": 100 * float(np.mean(ranks <= 5)),
            "R@10": 100 * float(np.mean(ranks <= 10)), "MedR": float(np.median(ranks)),
            "MnR": float(np.mean(ranks))}


def video_ranks(index: SegmentIndex, text: dict, agg: str = "max") -> dict[str, np.ndarray]:
    """Rank of the positive video and the argmax segment inside it, per query."""
    scores, best = index.score_matrix(text["emb"], agg=agg, chunk=QUERY_BLOCK)
    pos = np.array([index.video_index.get(v, -1) for v in text["video_id"]])
    covered = pos >= 0
    q = np.arange(len(pos))
    pos_score = np.where(covered, scores[q, np.where(covered, pos, 0)], -np.inf)
    ranks = 1 + (scores > pos_score[:, None]).sum(axis=1)
    ranks = np.where(covered, ranks, index.n_videos + 1)
    best_pos = best[q, np.where(covered, pos, 0)]
    return {"ranks": ranks, "covered": covered,
            "best_ts": index.ts[best_pos], "best_te": index.te[best_pos]}


def evaluate_partition(index: SegmentIndex, text: dict, agg: str = "max") -> dict:
    r = video_ranks(index, text, agg)
    row = {"partition": index.name, **rank_metrics(r["ranks"]),
           "vec/video": float(np.mean(index.counts)), "videos": index.n_videos,
           "queries": int(len(r["ranks"])), "coverage": 100 * float(np.mean(r["covered"]))}
    hit = tiou_vec(r["best_ts"], r["best_te"], text["gt_ts"], text["gt_te"]) >= 0.5
    top1 = (r["ranks"] == 1) & r["covered"]
    pooled = index.name == "event_single"   # one vector per video: no moment
    row["mom@.5|top1"] = float("nan") if pooled or not top1.any() else 100 * float(hit[top1].mean())
    row["VCMR R@1 (tIoU.5)"] = float("nan") if pooled else 100 * float((top1 & hit).mean())
    return row


def evaluate(cache_dir, partitions=DEFAULT_PARTITIONS, agg: str = "max") -> list[dict]:
    text = load_text(cache_dir)
    rows = []
    for name in partitions:
        try:
            index = SegmentIndex.load(cache_dir, name)
        except KeyError as e:
            print(f"[evaluate] skip {name}: {e}")
            continue
        rows.append(evaluate_partition(index, text, agg))
    return rows


COLUMNS = ("partition", "vec/video", "R@1", "R@5", "R@10", "MedR", "MnR", "mom@.5|top1",
           "VCMR R@1 (tIoU.5)", "coverage")


def markdown(rows: list[dict], columns=COLUMNS) -> str:
    def fmt(v):
        return v if isinstance(v, str) else ("-" if v != v else f"{v:.2f}")
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in columns) + " |" for r in rows]
    return "\n".join(lines)


def save_csv(rows: list[dict], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# Sanity check                                                                #
# --------------------------------------------------------------------------- #

def _gt_clips(cache_dir) -> tuple[np.ndarray, np.ndarray]:
    """``(seg_key, emb)`` of the GT event clips: ``events.npz``, else the ``gt`` rows of ``segments.npz``."""
    cache_dir = Path(cache_dir)
    if (cache_dir / "events.npz").is_file():
        with np.load(cache_dir / "events.npz") as blob:
            return blob["seg_key"], blob["emb"].astype(np.float32)
    with np.load(cache_dir / "segments.npz") as blob:
        sel = blob["partition"] == "gt"
        return blob["seg_key"][sel], blob["emb"][sel].astype(np.float32)


def evaluate_events(cache_dir, block: int = QUERY_BLOCK) -> tuple[dict, list[dict]]:
    """Single-event retrieval: each val caption searches every GT event in ``events.npz``.

    Correct = the caption's own event (same video *and* same window). Also reported:
    ``video@k`` (an event of the right video in the top k, any window) and
    ``tIoU.5@1`` (top-1 is the right video and overlaps the GT event by tIoU >= 0.5).
    Returns ``(summary, per_query)``; ``per_query`` has the top-1 answer of every caption.
    """
    from .search import EventIndex

    index = EventIndex.load(cache_dir)
    text = load_text(cache_dir)
    pos = np.array([index.row_of.get(k, -1) for k in text["gt_seg_key"]])
    if (pos < 0).any():
        raise SanityCheckFailed(f"{int((pos < 0).sum())} captions have no event in events.npz")

    n = len(pos)
    ranks = np.empty(n, dtype=np.int64)
    top5 = np.empty((n, min(5, len(index))), dtype=np.int64)
    best = np.empty(n, dtype=np.float32)
    for s in range(0, n, block):
        blk = slice(s, s + block)
        sim = index.scores(text["emb"][blk])                                  # [b, N]
        q = np.arange(sim.shape[0])
        ranks[blk] = 1 + (sim > sim[q, pos[blk]][:, None]).sum(1)
        part = np.argpartition(-sim, top5.shape[1] - 1, axis=1)[:, :top5.shape[1]]
        order = np.argsort(-np.take_along_axis(sim, part, 1), axis=1, kind="stable")
        top5[blk] = np.take_along_axis(part, order, 1)
        best[blk] = sim[q, top5[blk][:, 0]]

    top1 = top5[:, 0]
    right_video = index.video_id[top5] == text["video_id"][:, None]            # [n, 5]
    overlap = tiou_vec(index.ts[top1], index.te[top1], text["gt_ts"], text["gt_te"])
    summary = {
        **{f"event {k}": v for k, v in rank_metrics(ranks).items()},
        "event MRR": 100 * float(np.mean(1 / ranks)),
        "video@1": 100 * float(right_video[:, 0].mean()),
        "video@5": 100 * float(right_video.any(1).mean()),
        "tIoU.5@1": 100 * float((right_video[:, 0] & (overlap >= 0.5)).mean()),
        "queries": n, "events": len(index), "videos": len(set(index.video_id.tolist())),
    }
    per_query = [{
        "caption_id": str(text["caption_id"][i]), "query": str(text["sentence"][i]),
        "video_id": str(text["video_id"][i]),
        "gt": f"[{text['gt_ts'][i]:.1f}, {text['gt_te'][i]:.1f}]",
        "rank": int(ranks[i]), "correct": bool(ranks[i] == 1),
        "top1_video": str(index.video_id[top1[i]]),
        "top1": f"[{index.ts[top1[i]]:.1f}, {index.te[top1[i]]:.1f}]",
        "top1_caption": str(index.caption[top1[i]]), "top1_score": round(float(best[i]), 4),
    } for i in range(n)]
    return summary, per_query


def text_to_clip(cache_dir) -> dict:
    """Text -> clip on the ``gt`` segments, two galleries.

    * ``subset``: every distinct GT clip of the subset (val + dev distractors).
    * ``eval``  : one clip per val caption (diagonal), i.e. the matrix eval_youcookii
      scores. On the full val subset it is the same 3030 x 3030 problem.
    """
    text = load_text(cache_dir)
    keys, emb = _gt_clips(cache_dir)
    uniq, first = np.unique(keys, return_index=True)
    row_of = {k: i for i, k in enumerate(uniq)}
    G = emb[first]
    pos = np.array([row_of.get(k, -1) for k in text["gt_seg_key"]])
    if (pos < 0).any():
        raise SanityCheckFailed(f"{int((pos < 0).sum())} captions have no gt clip in segments.npz")
    T = text["emb"]
    G_eval = G[pos]                                           # row i = caption i's own clip

    ranks_subset = np.empty(len(T), dtype=np.int64)
    ranks_eval = np.empty(len(T), dtype=np.int64)
    for s in range(0, len(T), QUERY_BLOCK):                    # Q x E^T one block of queries at a time
        blk = slice(s, s + QUERY_BLOCK)
        q = np.arange(len(T))[blk]
        sim = T[blk] @ G.T
        ranks_subset[blk] = 1 + (sim > sim[q - s, pos[blk]][:, None]).sum(1)
        sim = T[blk] @ G_eval.T
        ranks_eval[blk] = 1 + (sim > sim[q - s, q][:, None]).sum(1)
    return {"subset": dict(rank_metrics(ranks_subset), gallery=len(G), MRR=100 * float(np.mean(1 / ranks_subset))),
            "eval": dict(rank_metrics(ranks_eval), gallery=len(T), MRR=100 * float(np.mean(1 / ranks_eval)))}


def sanity_check(cache_dir, ref_json=None, tol: float = 1.0, strict: bool = True) -> dict:
    """Compare text->clip (eval gallery) with the fine-tune eval's t2m numbers.

    Stops (raises) when the R@1 gap exceeds ``tol`` points and the gallery is the same
    size as the reference, i.e. the subset covers every val caption the eval scored.
    """
    res = text_to_clip(cache_dir)
    for name, m in res.items():
        print(f"[sanity] text->clip ({name:6} gallery {m['gallery']:5d}): R@1 {m['R@1']:.2f}  "
              f"R@5 {m['R@5']:.2f}  R@10 {m['R@10']:.2f}  MRR {m['MRR']:.2f}  MedR {m['MedR']:.0f}")
    ref = None
    if ref_json and Path(ref_json).is_file():
        blob = json.loads(Path(ref_json).read_text(encoding="utf-8"))
        ref = blob.get("text_to_multimodal (t2m)")
        n_ref = blob.get("num_samples")
        res["reference"] = dict(ref or {}, num_samples=n_ref, path=str(ref_json))
    if not ref:
        print(f"[sanity] no reference eval JSON ({ref_json}); compare the numbers above by hand.")
        return res
    gap = res["eval"]["R@1"] - ref["R@1"]
    same_size = n_ref == res["eval"]["gallery"]
    print(f"[sanity] reference t2m R@1 {ref['R@1']:.2f} (n={n_ref}) | here {res['eval']['R@1']:.2f} "
          f"(n={res['eval']['gallery']}) | gap {gap:+.2f}")
    res["gap_R@1"] = gap
    if not same_size:
        print("[sanity] gallery sizes differ (smoke / partial subset): not comparable, no stop.")
    elif abs(gap) > tol:
        msg = (f"\n{'!' * 78}\n!!  SANITY CHECK FAILED: text->clip R@1 {res['eval']['R@1']:.2f} vs eval "
               f"{ref['R@1']:.2f} (gap {gap:+.2f} > {tol}).\n!!  The cached embeddings do not match the "
               f"fine-tune eval: wrong adapter, preprocessing or text path.\n{'!' * 78}")
        print(msg)
        if strict:
            raise SanityCheckFailed(msg)
    else:
        print(f"[sanity] OK (|gap| <= {tol}).")
    return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a collected omni_cache/<tag> directory.")
    p.add_argument("cache_dir")
    p.add_argument("--partitions", nargs="*", default=list(DEFAULT_PARTITIONS))
    p.add_argument("--agg", default="max", choices=("max", "topk_mean", "lse"))
    p.add_argument("--ref", default=None, help="eval_youcookii JSON for the sanity check")
    p.add_argument("--csv", default=None)
    p.add_argument("--no-strict", action="store_true")
    a = p.parse_args(argv)
    sanity_check(a.cache_dir, a.ref, strict=not a.no_strict)
    rows = evaluate(a.cache_dir, a.partitions, a.agg)
    print(markdown(rows))
    if a.csv:
        save_csv(rows, a.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
