"""Quick text->video evaluation on the encoded subset (UEMR Exp 1, columns (a)/(b)).

Queries: every val caption in ``text.npz``. Positive: the caption's source video.
Per partition: R@1/5/10 (%), MedR, MnR, vectors/video, and a simple moment metric --
among queries whose top-1 video is right, the share whose argmax segment has
tIoU >= 0.5 with the caption's GT event (``mom@.5|top1``); ``VCMR R@1`` is the same
over all queries (right video *and* right moment). ``event_single`` (R1_gt) is the mean of the
``gt`` vectors per video, ``event_single_pred`` (R1_pred, UEMR_FINAL §5) the mean of the ``uniav_pred`` vectors.

Ranks are strict: ``1 + #(score > positive)``, as ``omniretriever.evaluation.metrics``.

``sanity_check`` must pass before the table is trusted: text -> clip on the ``gt``
segments, which on the full val subset is the exact computation of
``eval_youcookii.py`` and so must reproduce its t2m R@1.

    python -m omni_retrieval.evaluate <cache_dir> [--partitions gt global ...] [--ref best_val.json] \\
        [--split val|all]

The main gallery is the 394 val videos (``--split val``, filtered at evaluation time); ``all`` keeps the
dev videos as distractors.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .search import POOLED, SegmentIndex, tiou_vec

DEFAULT_PARTITIONS = ("global", "event_single", "uni_M_gt", "gt")
QUERY_BLOCK = 512   # queries per Q x E^T block


class SanityCheckFailed(RuntimeError):
    pass


def load_text(cache_dir, name: str = "text.npz") -> dict[str, np.ndarray]:
    """``text.npz`` (val queries) or another caption file of the same layout, e.g. ``text_dev.npz``."""
    with np.load(Path(cache_dir) / name) as blob:
        out = {k: blob[k] for k in blob.files}
    out["emb"] = out["emb"].astype(np.float32)
    return out


def rank_metrics(ranks: np.ndarray) -> dict[str, float]:
    ranks = np.asarray(ranks)
    return {"R@1": 100 * float(np.mean(ranks <= 1)), "R@5": 100 * float(np.mean(ranks <= 5)),
            "R@10": 100 * float(np.mean(ranks <= 10)), "MedR": float(np.median(ranks)),
            "MnR": float(np.mean(ranks))}


def video_ranks(index: SegmentIndex, text: dict, agg: str = "max", **agg_kw) -> dict[str, np.ndarray]:
    """Rank of the positive video and the argmax segment inside it, per query. ``agg_kw``: ``k`` (topk_mean),
    ``tau`` (lse)."""
    scores, best = index.score_matrix(text["emb"], agg=agg, chunk=QUERY_BLOCK, **agg_kw)
    pos = np.array([index.video_index.get(v, -1) for v in text["video_id"]])
    covered = pos >= 0
    q = np.arange(len(pos))
    pos_score = np.where(covered, scores[q, np.where(covered, pos, 0)], -np.inf)
    ranks = 1 + (scores > pos_score[:, None]).sum(axis=1)
    ranks = np.where(covered, ranks, index.n_videos + 1)
    best_pos = best[q, np.where(covered, pos, 0)]
    return {"ranks": ranks, "covered": covered,
            "best_ts": index.ts[best_pos], "best_te": index.te[best_pos]}


def gallery_videos(subset_path, split: str = "val") -> set[str] | None:
    """Videos of ``subset_videos.json`` with ``split`` (``val`` = the main gallery); ``all`` -> None (no filter)."""
    if split == "all":
        return None
    blob = json.loads(Path(subset_path).read_text(encoding="utf-8"))
    return {r["video_id"] for r in blob["videos"] if r["split"] == split}


def load_index(cache_dir, partition: str, videos=None) -> SegmentIndex:
    """``SegmentIndex.load``, restricted to ``videos`` when given (never shrink the gallery via ``n_videos``)."""
    index = SegmentIndex.load(cache_dir, partition)
    return index if videos is None else index.subset(videos)


def evaluate_partition(index: SegmentIndex, text: dict, agg: str = "max", videos=None, **agg_kw) -> dict:
    if videos is not None:
        index = index.subset(videos)
    r = video_ranks(index, text, agg, **agg_kw)
    row = {"partition": index.name, **rank_metrics(r["ranks"]),
           "vec/video": float(np.mean(index.counts)), "videos": index.n_videos,
           "queries": int(len(r["ranks"])), "coverage": 100 * float(np.mean(r["covered"]))}
    hit = tiou_vec(r["best_ts"], r["best_te"], text["gt_ts"], text["gt_te"]) >= 0.5
    top1 = (r["ranks"] == 1) & r["covered"]
    pooled = index.name in POOLED   # one vector per video: no moment
    row["mom@.5|top1"] = float("nan") if pooled or not top1.any() else 100 * float(hit[top1].mean())
    row["VCMR R@1 (tIoU.5)"] = float("nan") if pooled else 100 * float((top1 & hit).mean())
    return row


def evaluate(cache_dir, partitions=DEFAULT_PARTITIONS, agg: str = "max", videos=None) -> list[dict]:
    """One row per partition. ``videos``: gallery restriction (``gallery_videos``), None = every video."""
    text = load_text(cache_dir)
    rows = []
    for name in partitions:
        try:
            index = load_index(cache_dir, name, videos)
        except KeyError as e:
            print(f"[evaluate] skip {name}: {e}")
            continue
        rows.append(evaluate_partition(index, text, agg))
    return rows


def paired_bootstrap(rank_a, rank_b, k: int = 1, n_boot: int = 1000, seed: int = 0, unit: str = "query",
                     groups=None) -> dict:
    """R@k(a) - R@k(b) in points over the same queries, with a 95 % percentile interval.

    ``unit='query'`` resamples queries. ``unit='video'`` resamples the source videos (``groups[i]`` = video of
    query i) with replacement and keeps every query of each drawn video: queries of one video share their
    positive and its segments, so they are not independent and the per-query interval is too narrow.

    ``rank_a`` may be ``[n_seeds, n_queries]`` (several runs scored on the same queries): the difference is then
    the mean R@k over the seeds minus R@k(b), and every resample draws the same queries for all seeds.
    """
    hit_a = (np.asarray(rank_a) <= k).astype(np.float64)
    d = (hit_a.mean(axis=0) if hit_a.ndim == 2 else hit_a) - (np.asarray(rank_b) <= k).astype(np.float64)
    rng = np.random.default_rng(seed)
    if unit == "query":
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        boots = 100 * d[idx].mean(axis=1)
        n_units = len(d)
    elif unit == "video":
        if groups is None or len(groups) != len(d):
            raise ValueError("unit='video' needs groups (the video of every query)")
        _, g = np.unique(np.asarray(groups), return_inverse=True)
        sums, counts = np.bincount(g, weights=d), np.bincount(g).astype(np.float64)
        idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
        boots = 100 * sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
        n_units = len(sums)
    else:
        raise ValueError(f"unit must be 'query' or 'video', got {unit!r}")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": 100 * float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p(diff<=0)": float((boots <= 0).mean()), "n_queries": len(d), "unit": unit, "n_units": n_units,
            "n_boot": n_boot}


def table(cache_dir, parts: dict[str, str], splits, pairs=(), n_boot: int = 1000, seed: int = 0):
    """Rows of ``evaluate_partition`` and paired bootstraps by video, the table every notebook builds.

    ``parts``: ``{partition: label}``; ``splits``: ``[(name, text, videos)]`` (``videos`` None = every video);
    ``pairs``: ``[(a, b)]`` partition names, R@1(a) - R@1(b). Returns ``(rows, boots)`` as lists of dicts.
    """
    rows, boots = [], []
    for split, text, videos in splits:
        ranks = {}
        for p, label in parts.items():
            idx = load_index(cache_dir, p, videos)
            rows.append({**evaluate_partition(idx, text), "split": split, "partition": label})
            ranks[p] = video_ranks(idx, text)["ranks"]
        for a, b in pairs:
            bs = paired_bootstrap(ranks[a], ranks[b], 1, n_boot, seed, "video", text["video_id"])
            boots.append({"split": split, "A": parts[a], "B": parts[b], "A - B": bs["diff"],
                          "CI (video)": f"[{bs['ci_low']:+.2f}, {bs['ci_high']:+.2f}]",
                          "ci_low": bs["ci_low"], "ci_high": bs["ci_high"], "p(diff<=0)": bs["p(diff<=0)"]})
    return rows, boots


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


def score_event_queries(index, q_emb, q_video, gt_ts, gt_te, pos, query_ids, query_texts,
                        block: int = QUERY_BLOCK) -> tuple[dict, list[dict]]:
    """Core of event-retrieval evaluation, shared by val captions and custom queries.

    ``pos[i]`` is the database row of query i's GT event. Correct = that row ranks first.
    Also reported: ``video@k`` (an event of the right video in the top k, any window) and
    ``tIoU.5@1`` (top-1 is the right video and overlaps the GT window by tIoU >= 0.5).
    Ranks are strict (``1 + #(score > positive)``). ``Q x N`` is computed per query block.
    """
    q_video, gt_ts, gt_te, pos = map(np.asarray, (q_video, gt_ts, gt_te, pos))
    n = len(pos)
    k5 = min(5, len(index))
    ranks = np.empty(n, dtype=np.int64)
    top5 = np.empty((n, k5), dtype=np.int64)
    best = np.empty(n, dtype=np.float32)
    for s in range(0, n, block):
        blk = slice(s, s + block)
        sim = index.scores(q_emb[blk])                                          # [b, N]
        q = np.arange(sim.shape[0])
        ranks[blk] = 1 + (sim > sim[q, pos[blk]][:, None]).sum(1)
        part = np.argpartition(-sim, k5 - 1, axis=1)[:, :k5]
        order = np.argsort(-np.take_along_axis(sim, part, 1), axis=1, kind="stable")
        top5[blk] = np.take_along_axis(part, order, 1)
        best[blk] = sim[q, top5[blk][:, 0]]

    top1 = top5[:, 0]
    right_video = index.video_id[top5] == q_video[:, None]                      # [n, 5]
    overlap = tiou_vec(index.ts[top1], index.te[top1], gt_ts, gt_te)
    summary = {
        **{f"event {k}": v for k, v in rank_metrics(ranks).items()},
        "event MRR": 100 * float(np.mean(1 / ranks)),
        "video@1": 100 * float(right_video[:, 0].mean()),
        "video@5": 100 * float(right_video.any(1).mean()),
        "tIoU.5@1": 100 * float((right_video[:, 0] & (overlap >= 0.5)).mean()),
        "queries": n, "events": len(index), "videos": len(set(index.video_id.tolist())),
    }
    per_query = [{
        "query_id": str(query_ids[i]), "query": str(query_texts[i]), "video_id": str(q_video[i]),
        "gt": f"[{gt_ts[i]:.1f}, {gt_te[i]:.1f}]", "gt_caption": str(index.caption[pos[i]]),
        "rank": int(ranks[i]), "correct": bool(ranks[i] == 1),
        "top1_video": str(index.video_id[top1[i]]),
        "top1": f"[{index.ts[top1[i]]:.1f}, {index.te[top1[i]]:.1f}]",
        "top1_caption": str(index.caption[top1[i]]), "top1_score": round(float(best[i]), 4),
    } for i in range(n)]
    return summary, per_query


def evaluate_events(cache_dir, block: int = QUERY_BLOCK) -> tuple[dict, list[dict]]:
    """Single-event retrieval with the GT queries: every val caption in ``text.npz`` searches
    every event in ``events.npz``; correct = the caption's own event. See ``score_event_queries``.
    """
    from .search import EventIndex

    index = EventIndex.load(cache_dir)
    text = load_text(cache_dir)
    pos = np.array([index.row_of.get(k, -1) for k in text["gt_seg_key"]])
    if (pos < 0).any():
        raise SanityCheckFailed(f"{int((pos < 0).sum())} captions have no event in events.npz")
    return score_event_queries(index, text["emb"], text["video_id"], text["gt_ts"], text["gt_te"],
                               pos, text["caption_id"], text["sentence"], block)


def text_to_clip(cache_dir, text_name: str = "text.npz") -> dict:
    """Text -> clip on the ``gt`` segments, two galleries.

    * ``subset``: every distinct GT clip of the subset (val + dev distractors).
    * ``eval``  : one clip per caption (diagonal), i.e. the matrix eval_youcookii
      scores. On the full val subset it is the same 3030 x 3030 problem.

    ``text_name``: ``text_dev.npz`` gives the same two numbers for the dev captions (plan P6 §A1).
    """
    text = load_text(cache_dir, text_name)
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


def text_to_clip_matched(cache_dir, n_videos: int, n_rep: int, seed: int = 0, text_name: str = "text.npz") -> list[dict]:
    """Text -> clip on ``n_videos`` random videos, ``n_rep`` times (plan ``UEMR_BUOC_TIEP_P6.md`` §3.B).

    Each repetition draws ``n_videos`` source videos without replacement; queries = every caption of those
    videos, gallery = their distinct GT clips (the other clips of the same videos are the hard negatives).
    This puts the trainsplit at the gallery size of the dev check (``text_to_clip`` on ``text_dev.npz``),
    so the two R@1 can be compared. One row per repetition.
    """
    text = load_text(cache_dir, text_name)
    keys, emb = _gt_clips(cache_dir)
    uniq, first = np.unique(keys, return_index=True)
    row_of = {k: i for i, k in enumerate(uniq)}
    G = emb[first]
    pos = np.array([row_of.get(k, -1) for k in text["gt_seg_key"]])
    if (pos < 0).any():
        raise SanityCheckFailed(f"{int((pos < 0).sum())} captions have no gt clip in {cache_dir}")
    videos = np.unique(text["video_id"])
    if n_videos > len(videos):
        raise ValueError(f"n_videos={n_videos} > {len(videos)} videos with captions")
    rng = np.random.default_rng(seed)
    out = []
    for rep in range(n_rep):
        q = np.flatnonzero(np.isin(text["video_id"], rng.choice(videos, size=n_videos, replace=False)))
        clips, own = np.unique(pos[q], return_inverse=True)          # own[i] = column of caption i's clip
        sim = text["emb"][q] @ G[clips].T
        ranks = 1 + (sim > sim[np.arange(len(q)), own][:, None]).sum(1)
        out.append(dict(rank_metrics(ranks), rep=rep, videos=n_videos, queries=int(len(q)), gallery=int(len(clips)),
                        MRR=100 * float(np.mean(1 / ranks))))
    return out


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
    p.add_argument("--split", default="val", choices=("val", "dev", "all"),
                   help="gallery: the videos of this split in subset_videos.json; 'all' = every video (498)")
    p.add_argument("--subset", default=None, help="subset_videos.json (default <cache_dir>/../common/)")
    a = p.parse_args(argv)
    sanity_check(a.cache_dir, a.ref, strict=not a.no_strict)
    subset = a.subset or Path(a.cache_dir).parent / "common" / "subset_videos.json"
    videos = gallery_videos(subset, a.split)
    print(f"[evaluate] gallery: {'all videos' if videos is None else f'{len(videos)} {a.split} videos'}")
    rows = evaluate(a.cache_dir, a.partitions, a.agg, videos)
    print(markdown(rows))
    if a.csv:
        save_csv(rows, a.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
