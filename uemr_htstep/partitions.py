"""The six rows of §4 C, in the YouCook2 code paths, with every parameter passed in from the protocol.

R6 is the chain of the YouCook2 runs (``uemr_yc2_bo_sung_*.ipynb`` ô 7 ``post``): ``postprocess_uemr`` on the raw
proposals -> 3 decimals (``segments_pred.jsonl``) -> ``partitions_uemr._round`` (2 decimals, never past the mp4).
Names are the YouCook2 partition names, so ``SegmentIndex`` and the tables read them unchanged.
"""

from __future__ import annotations

from omni_retrieval.bo_sung import fill_gaps
from omni_retrieval.partitions import clamp, uniform
from omni_retrieval.partitions_uemr import _round, uni_chunk
from uniav_app.postprocess import postprocess_uemr

# partition name -> row of the plan, and the file name under segments/
ROWS = {"global": "R0", "uni_chunk": "R2", "uni_M": "R3", "uniav_pred": "R6", "gt": "R7", "r8": "R8"}
FILES = {"global": "r0", "uni_chunk": "r2", "uni_M": "r3", "uniav_pred": "athena961f/r6", "gt": "r7",
         "r8": "athena961f/r8"}


def r6(proposals: list[dict], duration: float, theta: float, k_max: int, d_min: float, merge_tiou: float):
    segs = postprocess_uemr(proposals, duration, theta, k_max, d_min, merge_tiou)
    return _round([[round(float(s["ts"]), 3), round(float(s["te"]), 3)] for s in segs], duration)


def build(r6_parts: dict, gt: dict, durations: dict, chunk_sec: float, d_min: float, l_min: float,
          l_fill: float) -> dict[str, dict]:
    """``{name: {video_id: [[ts, te], ...]}}`` for every video of ``durations``. R3 has ``K_pred`` = the number of
    R6 segments of the video; R2 chunks of ``chunk_sec``; R8 = R6 + gap windows (``fill_gaps``)."""
    vids = sorted(durations)
    missing = [v for v in vids if v not in r6_parts or v not in gt]
    if missing:
        raise ValueError(f"{len(missing)} videos have no R6 or no GT: {missing[:5]}")
    return {
        "global": {v: clamp([[0.0, durations[v]]], durations[v]) for v in vids},
        "uni_chunk": {v: _round(uni_chunk(durations[v], chunk_sec, d_min), durations[v]) for v in vids},
        "uni_M": {v: _round(uniform(durations[v], len(r6_parts[v])), durations[v]) for v in vids},
        "uniav_pred": {v: r6_parts[v] for v in vids},
        "gt": {v: gt[v] for v in vids},
        "r8": {v: _round(fill_gaps(r6_parts[v], durations[v], l_min, l_fill)[0], durations[v]) for v in vids},
    }
