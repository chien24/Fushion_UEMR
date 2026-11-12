"""Teacher vectors of UniAV from Omni ``best`` (docs/uemr_chot/UEMR_CHOT_01 §2-4): which GT segments, from which source,
and the files ``libs/datasets/htstep.py`` of HanLi 77fc4fa reads.

Format of the teacher set (read from ``libs/datasets/helper.py::load_omni_segments`` and the old files): one
``<folder>/{train,val}/<video_id>.npz`` per video, ``emb`` (k, 3584) fp16, ``seg_idx`` (k,) = index of the segment in the
video's ``annotations`` list of the official json (the key the loader uses), plus ``step_id``, ``segment`` (seconds, as
annotated) and ``partial`` for the record. The loader L2-normalises, skips a ``seg_idx`` it dropped and leaves every
segment without a row out of both distillation losses (``omni_ok``), so a missing vector needs no change of HanLi.

The list is every annotation of the video. The dataset drops the segments that start after the features end (a few);
their rows are never read, so the list is a superset of what the dataset uses and no feature file is needed to build it.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

from omni_retrieval.manifest import PROMPT, seg_key

TOL = 0.01      # s: a pre-cut clip / a cached vector stands for a segment when its window is this close


def seg_id(video: str, idx: int) -> str:
    return f"{video}#a{idx}"


def gt_segments(annotations: dict, videos) -> list[dict]:
    """Every annotation of ``videos`` in json order: ``{id, video_id, seg_idx, segment, step_id, partial}``."""
    return [{"id": seg_id(v, i), "video_id": v, "seg_idx": i, "segment": [float(a["segment"][0]), float(a["segment"][1])],
             "step_id": int(a["id"]), "partial": a["partial"]}
            for v in sorted(videos) for i, a in enumerate(annotations[v]["annotations"])]


def _close(a, b) -> bool:
    return abs(a[0] - b[0]) <= TOL and abs(a[1] - b[1]) <= TOL


def assign(segs: list[dict], clip_rows: list[dict], cached: set, mp4_videos: set) -> list[dict]:
    """``source`` of every segment, first that applies (plan §2):

    ``cache``   a manifest row of the same segment whose window (rounded to 0.01 s, the key of the dev-1 store) is in ``cached``;
    ``clip``    a pre-cut clip of the same segment whose window is the segment (<= 0.01 s);
    ``mp4``     the video's mp4 can be fetched: ``timestamps`` = the annotated segment;
    ``missing`` end <= start, or no mp4.
    ``clip_rows``: records of ``train/val_omni_htstep.jsonl`` (only clips that were cut are in them).
    """
    rows = {}
    for r in clip_rows:
        rows.setdefault(r["video_id"], []).append(r)
    out = []
    for s in segs:
        hit = next((r for r in rows.get(s["video_id"], []) if _close(r["segment"], s["segment"])), None)
        key = seg_key(s["video_id"], round(float(hit["window"][0]), 2), round(float(hit["window"][1]), 2)) if hit else None
        if s["segment"][1] <= s["segment"][0]:
            src = {"source": "missing", "why": "end <= start"}
        elif key in cached:
            src = {"source": "cache", "key": key}
        elif hit and _close(hit["window"], s["segment"]):
            src = {"source": "clip", "clip": hit["clip"]}
        elif s["video_id"] in mp4_videos:
            src = {"source": "mp4"}
        else:
            src = {"source": "missing", "why": "no mp4"}
        out.append({**s, **src})
    return out


def record(s: dict) -> dict:
    """The encoder record of a ``clip`` / ``mp4`` segment (no caption turn: the forward returns the AV vector)."""
    media = {"clip": s["clip"]} if s["source"] == "clip" else {"video": f"{s['video_id']}.mp4", "timestamps": s["segment"]}
    return {"id": s["id"], "type": "retrieval", "conversations": [{"from": "human", "value": PROMPT}], **media}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def write(segs: list[dict], vectors: dict, folder) -> dict:
    """One npz per video with the segments that have a vector (``vectors``: ``{segment id: (3584,)}``), L2-normalised fp16.
    ``{'files': {name: sha256}, 'segments': n, 'missing': [ids]}``; a video with no vector gets no file (the loader's own case)."""
    os.makedirs(folder, exist_ok=True)
    by_video, files, missing = {}, {}, []
    for s in segs:
        if s["id"] in vectors:
            by_video.setdefault(s["video_id"], []).append(s)
        else:
            missing.append(s["id"])
    for v, rows in sorted(by_video.items()):
        emb = np.stack([np.asarray(vectors[s["id"]], np.float32).reshape(-1) for s in rows])
        emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        path = os.path.join(folder, f"{v}.npz")
        np.savez(path, emb=emb.astype(np.float16), seg_idx=np.array([s["seg_idx"] for s in rows], np.int64),
                 step_id=np.array([s["step_id"] for s in rows], np.int64), segment=np.array([s["segment"] for s in rows], np.float64),
                 partial=np.array([s["partial"] for s in rows]))
        files[f"{v}.npz"] = sha256(path)
    return {"files": files, "segments": sum(len(r) for r in by_video.values()), "missing": missing}
