"""Post-processing of UniAV proposals into the R6 segments (moved unchanged from ``uniav_app/postprocess.py``).

Pure functions on lists of dicts with ``ts``, ``te`` (seconds) and ``conf``; other keys are carried along.

In this order:
  1. keep proposals with conf >= theta, at most K_max (highest conf first);
  2. if nothing is left, keep the highest-confidence proposal;
  3. merge proposals with tIoU > merge_tiou: replace the pair by its union [min ts, max te] with
     conf = max, repeated (most-overlapping pair first) until no pair is above the threshold;
  4. extend segments shorter than d_min symmetrically around their centre, shifted inwards at the
     video borders, never outside [0, duration];
  5. sort by time.
"""

from __future__ import annotations

def tiou(a, b) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return float(inter / union) if union > 0 else 0.0


def filter_theta(props: list[dict], theta: float) -> list[dict]:
    return [p for p in props if float(p["conf"]) >= theta]


def top_k(props: list[dict], k: int) -> list[dict]:
    return sorted(props, key=lambda p: -float(p["conf"]))[:k]


def merge_overlaps(segs: list[dict], thr: float = 0.7) -> list[dict]:
    segs = [dict(s) for s in segs]
    while len(segs) > 1:
        best, bi, bj = thr, -1, -1
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                o = tiou((segs[i]["ts"], segs[i]["te"]), (segs[j]["ts"], segs[j]["te"]))
                if o > best:
                    best, bi, bj = o, i, j
        if bi < 0:
            break
        a, b = segs[bi], segs[bj]
        keep = dict(a if float(a["conf"]) >= float(b["conf"]) else b)
        keep.update(ts=min(a["ts"], b["ts"]), te=max(a["te"], b["te"]), conf=max(float(a["conf"]), float(b["conf"])),
                    merged=int(a.get("merged", 1)) + int(b.get("merged", 1)))
        segs = [s for k, s in enumerate(segs) if k not in (bi, bj)] + [keep]
    return segs


def extend_short(segs: list[dict], d_min: float, duration: float | None) -> list[dict]:
    out = []
    for s in segs:
        s = dict(s)
        ts, te = float(s["ts"]), float(s["te"])
        if te - ts < d_min:
            c = 0.5 * (ts + te)
            ts, te = c - 0.5 * d_min, c + 0.5 * d_min
            if duration is not None:
                if d_min >= duration:
                    ts, te = 0.0, float(duration)
                elif ts < 0:
                    ts, te = 0.0, d_min
                elif te > duration:
                    ts, te = float(duration) - d_min, float(duration)
            s.update(ts=ts, te=te, extended=True)
        out.append(s)
    return out


def sort_by_time(segs: list[dict]) -> list[dict]:
    return sorted(segs, key=lambda s: (float(s["ts"]), float(s["te"])))


def postprocess_uemr(props: list[dict], duration: float | None, theta: float, k_max: int = 30,
                     d_min: float = 2.0, merge_tiou: float = 0.7) -> list[dict]:
    """Raw proposals (after soft-NMS) -> UEMR segments (see the module docstring)."""
    kept = top_k(filter_theta(props, theta), k_max)
    if not kept and props:
        kept = [max(props, key=lambda p: float(p["conf"]))]
    kept = merge_overlaps(kept, merge_tiou)
    kept = extend_short(kept, d_min, duration)
    return sort_by_time(kept)
