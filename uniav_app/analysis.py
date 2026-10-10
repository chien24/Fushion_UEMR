"""Per-video error analysis of the final segments against GT (pure numpy, no model).

Definitions (one video, segments S, GT steps G):
* best tIoU of a GT: max over S (not one-to-one); **missed** = best tIoU == 0 (no overlap at all).
* **matched@t**: one-to-one Hungarian pair with tIoU >= t (``match.hungarian_match``).
* **false positive (no overlap)**: a segment overlapping no GT; **unmatched@0.5**: not in a pair at 0.5.
* **merge**: a segment covering >= 2 GT steps, "covering" = at least ``cover`` (0.5) of the GT's length
  lies inside the segment.
* **fragmented**: a GT step cut into >= 2 segments, i.e. >= 2 segments with at least ``cover`` of their own
  length inside the GT. (Column ``fragmented``; ``split`` is the dataset split val/dev.)
* **boundary offset** (seconds, pairs matched at tIoU >= 0.3): left = ts_pred - ts_gt, right = te_pred - te_gt
  (negative = the prediction starts / ends early).
"""

from __future__ import annotations

import numpy as np

from .match import best_match, hungarian_match, seg_array

DURATION_BINS = (("<5s", 0, 5), ("5-10s", 5, 10), ("10-20s", 10, 20), ("20-40s", 20, 40),
                 ("40-80s", 40, 80), (">=80s", 80, np.inf))
BOUNDARY_TIOU = 0.3
COVER = 0.5


def duration_bin(d: float) -> str:
    for name, lo, hi in DURATION_BINS:
        if lo <= d < hi:
            return name
    return DURATION_BINS[-1][0]


def _inter(a, b) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def analyze_video(gts: list[dict], segs: list[dict], thrs=(0.3, 0.5, 0.7), cover: float = COVER) -> tuple[list[dict], list[dict]]:
    """``(gt_rows, seg_rows)``: one row per GT step and one per segment, with the flags above.

    ``gts``: dicts with ts, te, caption, event_id; ``segs``: dicts with t_s/t_e (or ts/te), conf, caption.
    """
    g = seg_array(gts)
    s = seg_array(segs)
    best = best_match(s, g)
    pairs = {t: {gi: (pi, iou) for pi, gi, iou in hungarian_match(s, g, t)} for t in set(thrs) | {BOUNDARY_TIOU}}
    gt_rows, seg_rows = [], []
    covers = [[_inter(sj, gi) / max(gi[1] - gi[0], 1e-9) >= cover for gi in g] for sj in s]   # seg j covers GT i
    inside = [[_inter(sj, gi) / max(sj[1] - sj[0], 1e-9) >= cover for gi in g] for sj in s]   # seg j lies in GT i
    for i, ev in enumerate(gts):
        bi, biou = best[i]
        row = {"event_id": ev.get("event_id", i), "ts": float(g[i, 0]), "te": float(g[i, 1]),
               "dur": float(g[i, 1] - g[i, 0]), "dur_bin": duration_bin(float(g[i, 1] - g[i, 0])),
               "caption": ev.get("caption", ""), "best_tiou": biou, "best_seg": bi, "missed": biou <= 0.0,
               "n_parts": int(sum(inside[j][i] for j in range(len(s))))}
        row["fragmented"] = row["n_parts"] >= 2   # not "split": that key is the dataset split (val/dev)
        row["in_merged_seg"] = bool(bi >= 0 and sum(covers[bi]) >= 2)
        for t in thrs:
            pi, iou = pairs[t].get(i, (-1, 0.0))
            row[f"match@{t}"] = pi
            row[f"tiou@{t}"] = iou
        pi, _ = pairs[BOUNDARY_TIOU].get(i, (-1, 0.0))
        row["left_offset"] = float(s[pi, 0] - g[i, 0]) if pi >= 0 else None
        row["right_offset"] = float(s[pi, 1] - g[i, 1]) if pi >= 0 else None
        gt_rows.append(row)
    matched05 = {pi for pi, _ in pairs.get(0.5, {}).values()}
    for j, sg in enumerate(segs):
        ious = [_inter(s[j], gi) / max((s[j, 1] - s[j, 0]) + (gi[1] - gi[0]) - _inter(s[j], gi), 1e-9) for gi in g]
        seg_rows.append({"seg": j, "ts": float(s[j, 0]), "te": float(s[j, 1]), "dur": float(s[j, 1] - s[j, 0]),
                         "conf": float(sg.get("conf", 0.0)), "caption": sg.get("caption", ""),
                         "max_tiou": float(max(ious)) if ious else 0.0,
                         "fp_no_overlap": (max(ious) if ious else 0.0) <= 0.0,
                         "unmatched@0.5": j not in matched05,
                         "n_gt_covered": int(sum(covers[j])), "merge": int(sum(covers[j])) >= 2})
    return gt_rows, seg_rows


def offset_stats(values) -> dict:
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if v.size == 0:
        return {"n": 0, "mean": None, "median": None, "mean_abs": None, "std": None}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "mean_abs": float(np.abs(v).mean()), "std": float(v.std())}


def count_stats(k_pred: dict[str, int], k_gt: dict[str, int]) -> dict:
    """Number of segments per video against the number of GT steps."""
    vids = [v for v in k_gt if v in k_pred]
    p = np.array([k_pred[v] for v in vids], dtype=np.float64)
    g = np.array([k_gt[v] for v in vids], dtype=np.float64)
    if not vids:
        return {}
    r = p / np.maximum(g, 1)
    return {"n_videos": len(vids), "mean_K_pred": float(p.mean()), "mean_K_GT": float(g.mean()),
            "median_K_pred": float(np.median(p)), "median_K_GT": float(np.median(g)),
            "mean_ratio": float(r.mean()), "mean_abs_diff": float(np.abs(p - g).mean()),
            "pct_fewer": float(100 * (p < g).mean()), "pct_equal": float(100 * (p == g).mean()),
            "pct_more": float(100 * (p > g).mean()), "pct_ratio_lt_0.5": float(100 * (r < 0.5).mean()),
            "pct_ratio_gt_2": float(100 * (r > 2).mean())}


def length_hist(pred_durs, gt_durs) -> list[dict]:
    pd_, gd = np.asarray(pred_durs, dtype=np.float64), np.asarray(gt_durs, dtype=np.float64)
    rows = []
    for name, lo, hi in DURATION_BINS:
        np_, ng = int(((pd_ >= lo) & (pd_ < hi)).sum()), int(((gd >= lo) & (gd < hi)).sum())
        rows.append({"bin": name, "n_pred": np_, "pct_pred": 100 * np_ / max(len(pd_), 1),
                     "n_gt": ng, "pct_gt": 100 * ng / max(len(gd), 1)})
    rows.append({"bin": "mean_s", "n_pred": float(pd_.mean()) if pd_.size else None,
                 "n_gt": float(gd.mean()) if gd.size else None})
    rows.append({"bin": "median_s", "n_pred": float(np.median(pd_)) if pd_.size else None,
                 "n_gt": float(np.median(gd)) if gd.size else None})
    return rows
