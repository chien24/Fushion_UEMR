"""HT-Step val_seen as a UEMR gallery (plan HTSTEP_UEMR_OMNI_UNIAV.md §1, §4).

Inputs are the files of the HT-Step and Omni repos as they are: ``annotations.json`` (official), the manifests
``train_omni_htstep.jsonl`` / ``val_omni_htstep.jsonl`` of ``scripts/convert_htstep.py``, and the ids of the UniAV
feature files. A query is one record of ``val_omni_htstep.jsonl``: Full and Partial Match, text ``"<activity>: <headline>"``.
Its window is the record's ``window`` (the step cut to the mp4 at precut), rounded to 0.01 s like every UEMR clip.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from omni_retrieval.manifest import seg_key

VAL_SUBSET = "val_seen"
MAX_STEPS = 48        # scripts/convert_htstep.py MAX_STEPS = UniAV num_classes


def normalize(text: str) -> str:
    """Key of the multi-positive scoring: lower case, punctuation removed, whitespace collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(text).lower())).strip()


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def video_steps(annotations: list[dict]) -> list[dict]:
    """The step filter of ``scripts/convert_htstep.py::video_steps``: end > start, then the first 48."""
    return [a for a in annotations if a["segment"][1] > a["segment"][0]][:MAX_STEPS]


def train_steps_per_video(annotations: dict) -> dict:
    """Mean number of steps per HT-Step train video (the target of the θ rule, plan §4.2)."""
    counts = [len(video_steps(v["annotations"])) for v in annotations.values() if v["subset"].lower() == "train"]
    counts = [c for c in counts if c > 0]
    return {"train_videos": len(counts), "train_steps": int(sum(counts)), "mean": float(np.mean(counts))}


def select_gallery(annotations: dict, feat_ids, val_rows: list[dict], uniav_train, omni_train) -> tuple[list[str], dict]:
    """Rule of plan §4.1, ``(sorted video ids, counts per filter)``: val_seen, UniAV features, in the val manifest, and not
    a train video of either model. ``counts['in train of a model']`` must be 0 (plan §1 stops the run otherwise)."""
    val = {v for v, x in annotations.items() if x["subset"].lower() == VAL_SUBSET}
    with_feat = val & set(feat_ids)
    with_mp4 = with_feat & {r["video_id"] for r in val_rows}
    train = set(uniav_train) | set(omni_train)
    keep = sorted(with_mp4 - train)
    return keep, {"val_seen": len(val), "with UniAV features": len(with_feat), "in val_omni_htstep.jsonl": len(with_mp4),
                  "in train of a model": len(with_mp4 & train), "kept": len(keep)}


def queries(val_rows: list[dict], videos) -> list[dict]:
    """The ``text.npz`` fields (``caption_id, video_id, sentence, gt_ts, gt_te, gt_seg_key``), ``headline`` (multi-positive
    key) and ``partial``, one per manifest record of a kept video, in manifest order."""
    keep = set(videos)
    out = []
    for r in val_rows:
        v = r["video_id"]
        if v in keep:
            ts, te = round(float(r["window"][0]), 2), round(float(r["window"][1]), 2)
            out.append({"caption_id": r["id"], "video_id": v, "sentence": r["text"], "headline": normalize(r["text"]),
                        "partial": r["partial"], "gt_ts": ts, "gt_te": te, "gt_seg_key": seg_key(v, ts, te)})
    return out


def gt_partition(captions: list[dict]) -> dict[str, list[list[float]]]:
    """R7: the distinct GT windows of every video."""
    out = defaultdict(set)
    for c in captions:
        out[c["video_id"]].add((c["gt_ts"], c["gt_te"]))
    return {v: sorted([list(s) for s in segs]) for v, segs in out.items()}


def headline_stats(captions: list[dict], top: int = 20) -> dict:
    """How often a text is shared between videos of the gallery (why strict scoring is too harsh)."""
    vids = defaultdict(set)
    for c in captions:
        vids[c["headline"]].add(c["video_id"])
    shared = [len(vids[c["headline"]]) >= 2 for c in captions]
    return {"queries": len(captions), "distinct texts": len(vids),
            "% queries whose text is in >= 2 videos": 100 * float(np.mean(shared)) if captions else 0.0,
            "mean videos per text": float(np.mean([len(s) for s in vids.values()])) if vids else 0.0,
            "top": [{"text": h, "videos": len(s)} for h, s in sorted(vids.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:top]]}


def teacher_vectors(folder) -> dict:
    """``htstep-val-seen-feats/omni_av/``: ``{(video, start, end rounded to 1 ms): {npz, row, segment, emb}}`` of every GT
    segment that has a teacher vector."""
    out = {}
    for p in sorted(Path(folder).glob("*.npz")):
        with np.load(p) as z:
            for row, (seg, emb) in enumerate(zip(z["segment"], z["emb"])):
                out[(p.stem, round(float(seg[0]), 3), round(float(seg[1]), 3))] = {"npz": str(p), "row": row,
                                                                                    "segment": [float(seg[0]), float(seg[1])],
                                                                                    "emb": emb.astype(np.float32)}
    return out


def teacher_queries(captions: list[dict], val_rows: list[dict], teacher: dict) -> tuple[list[int], dict]:
    """``(indices of the queries whose GT step has a teacher vector, {gt_seg_key: teacher vector})``; the manifest's
    ``segment`` is the annotated window the teacher was encoded on."""
    segment = {r["id"]: r["segment"] for r in val_rows}
    keep, by_clip = [], {}
    for i, c in enumerate(captions):
        s = segment[c["caption_id"]]
        hit = teacher.get((c["video_id"], round(float(s[0]), 3), round(float(s[1]), 3)))
        if hit is not None:
            keep.append(i)
            by_clip[c["gt_seg_key"]] = hit["emb"]
    return keep, by_clip


def smoke_videos(teacher: dict, durations: dict, av1, n_teacher: int = 4, min_clips: int = 5) -> list[str]:
    """The videos of the smoke run: the ``n_teacher`` shortest with at least ``min_clips`` teacher clips (the teacher and
    encode checks need 20 clips) and the shortest AV1 video (the transcode path)."""
    rows = defaultdict(int)
    for v, _, _ in teacher:
        rows[v] += 1
    cand = sorted((v for v, n in rows.items() if n >= min_clips and v in durations), key=lambda v: (durations[v], v))
    return sorted(set(cand[:n_teacher]) | set(sorted((v for v in av1 if v in durations), key=lambda v: (durations[v], v))[:1]))


def write_ids(path, ids) -> None:
    Path(path).write_text("".join(f"{v}\n" for v in ids), encoding="utf-8", newline="\n")
