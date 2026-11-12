"""Zero-shot benchmark as a UEMR gallery (docs/uemr_chot/UEMR_CHOT_03 §1, §3): videos, queries, the rows that need no UniAV.

A benchmark is ``{video_id: {'duration', 'annotations': [{'segment': [s, e], 'sentence'}]}}`` (YouCook2: the ``validation``
videos of ``youcookii_annotations_trainval.json``). A query is the sentence of one GT step, verbatim: no recipe / activity
name is added (that would be a GT label). Captions have the fields ``uemr_htstep.data.queries`` gives (minus ``partial``),
so the scoring code of the dev runs reads them unchanged.
"""

from __future__ import annotations

from omni_retrieval.manifest import seg_key
from omni_retrieval.partitions_uemr import _round

from .data import normalize


def yc2(database: dict, subset: str = "validation") -> dict:
    return {v: x for v, x in database.items() if x["subset"] == subset}


def overlap(videos, **id_sets) -> dict:
    """``{name: sorted ids of ``videos`` also in that set}``, same YouTube id (plan §1)."""
    vids = set(videos)
    return {name: sorted(vids & set(ids)) for name, ids in id_sets.items()}


def funnel(bench: dict, have_mp4, with_audio, readable, in_training) -> tuple[list[str], dict]:
    """``(kept videos sorted, count after every filter of plan §1 in order)``."""
    steps = (("tải được", have_mp4), ("có tiếng", with_audio), ("đọc được (decord)", readable))
    keep, counts = set(bench), {"annotation": len(bench)}
    for name, ok in steps:
        keep &= set(ok)
        counts[name] = len(keep)
    keep -= set(in_training)
    counts["không trùng train / dev"] = len(keep)
    return sorted(keep), counts


def queries(bench: dict, videos, durations: dict) -> tuple[list[dict], list[str]]:
    """``(captions, dropped ids)``: one per GT step of ``videos`` in annotation order; the window is cut to the mp4
    (``durations``) and rounded to 0.01 s like every UEMR clip. A step with no window left in the mp4 is dropped and listed."""
    out, dropped = [], []
    for v in sorted(videos):
        for i, a in enumerate(bench[v]["annotations"]):
            ts, te = _round([[max(0.0, float(a["segment"][0])), float(a["segment"][1])]], durations[v])[0]
            if te <= ts:
                dropped.append(f"{v}_{i}")
                continue
            out.append({"caption_id": f"{v}_{i}", "video_id": v, "sentence": a["sentence"], "headline": normalize(a["sentence"]),
                        "gt_ts": ts, "gt_te": te, "gt_seg_key": seg_key(v, ts, te)})
    return out, dropped


def r0(durations: dict) -> dict:
    return {v: _round([[0.0, durations[v]]], durations[v]) for v in sorted(durations)}
