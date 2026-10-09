"""Ways of cutting each video into segments (UEMR §5).

A partition is ``{video_id: [[ts, te], ...]}`` in seconds, sorted by time. Built in:

* ``global``   -- ``[[0, duration]]`` (R0)
* ``gt``       -- the YouCook2 GT events (R7, oracle)
* ``uni_M_gt`` -- M equal segments with M = number of GT events (count control for R7)

Anything else (``uniav_pred``, ``uni_M``, ``rand_M``, ``kmedoids_M``, ...) is a JSON file
of the same shape, passed through :func:`load_partition_file`. Every partition goes
through :func:`clamp` against the mp4's real duration, with the rule
``convert_youcookii.py`` used for the GT manifests.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import BASE_PARTITIONS
from .subset import video_id

# Same floor as scripts/convert_youcookii.py: a segment must start at least this long
# before the file ends, or the frame window and the audio cut come back empty.
MIN_SEG_SEC = 1.0


def clamp(segments, duration: float) -> list[list[float]]:
    """Clamp to ``[0, duration]``, drop empty / past-the-end segments, sort by time."""
    out = []
    for ts, te in segments:
        ts, te = max(0.0, float(ts)), min(float(te), float(duration))
        if te <= ts or ts > duration - MIN_SEG_SEC:
            continue
        out.append([round(ts, 3), round(te, 3)])
    return sorted(out)


def uniform(duration: float, m: int) -> list[list[float]]:
    edges = np.linspace(0.0, float(duration), int(m) + 1)
    return [[round(float(a), 3), round(float(b), 3)] for a, b in zip(edges[:-1], edges[1:])]


def gt_segments(events: dict[str, list[dict]], videos: set[str]) -> dict[str, list[list[float]]]:
    out = defaultdict(list)
    for split in ("val", "dev"):
        for r in events[split]:
            v = video_id(r)
            if v in videos:
                out[v].append([float(r["timestamps"][0]), float(r["timestamps"][1])])
    return out


def build_base(subset: list[dict], events: dict[str, list[dict]],
               names=BASE_PARTITIONS) -> dict[str, dict]:
    """The built-in partitions in ``names`` (``global``, ``gt``, ``uni_M_gt``) for every video.

    Event retrieval (one vector per GT event) only needs ``names=("gt",)``.
    """
    unknown = set(names) - set(BASE_PARTITIONS)
    if unknown:
        raise ValueError(f"unknown built-in partitions {sorted(unknown)}; have {BASE_PARTITIONS}")
    durations = {r["video_id"]: r["duration"] for r in subset}
    gt = gt_segments(events, set(durations))
    parts = {name: {} for name in names}
    for v, dur in durations.items():
        gt_v = clamp(gt.get(v, []), dur)
        if "global" in parts:
            parts["global"][v] = clamp([[0.0, dur]], dur)
        if "gt" in parts:
            parts["gt"][v] = gt_v
        if "uni_M_gt" in parts:
            parts["uni_M_gt"][v] = clamp(uniform(dur, len(gt_v)), dur) if gt_v else []
    return parts


def load_partition_file(path, durations: dict[str, float]) -> dict[str, list[list[float]]]:
    """Read ``video_id -> [[ts, te], ...]`` (extra entries such as a confidence are ignored).

    Videos of the subset missing from the file are reported; they simply have no
    segments in that partition (``evaluate`` reports the coverage).
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for v, dur in durations.items():
        segs = raw.get(v) or raw.get(f"{v}.mp4")
        if segs:
            out[v] = clamp([s[:2] for s in segs], dur)
    missing = len(durations) - len(out)
    if missing:
        print(f"[partitions] {Path(path).name}: {missing} of {len(durations)} subset videos have no segments")
    return out


def build_all(subset: list[dict], events: dict[str, list[dict]], extra: dict | None = None,
              base=BASE_PARTITIONS) -> dict[str, dict]:
    parts = build_base(subset, events, base)
    durations = {r["video_id"]: r["duration"] for r in subset}
    for name, path in (extra or {}).items():
        parts[name] = load_partition_file(path, durations)
    return parts


def save_partitions(parts: dict[str, dict], directory) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, segs in parts.items():
        (directory / f"{name}.json").write_text(json.dumps(segs), encoding="utf-8")


def summary(parts: dict[str, dict]) -> list[dict]:
    rows = []
    for name, segs in parts.items():
        counts = [len(s) for s in segs.values()]
        lengths = [te - ts for s in segs.values() for ts, te in s]
        rows.append({
            "partition": name, "videos": sum(c > 0 for c in counts), "segments": int(sum(counts)),
            "vec/video": round(float(np.mean(counts)), 2) if counts else 0.0,
            "median len (s)": round(float(np.median(lengths)), 1) if lengths else 0.0,
        })
    return rows
