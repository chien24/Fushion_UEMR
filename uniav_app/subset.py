"""Gallery videos and YouCook2 GT.

Gallery = the Omni gallery (394 val + 104 dev = 498 videos), read from
``MyDrive/uemr/omni_cache/common/subset_videos.json`` (read only). When that file is missing it is
rebuilt with the same rule as ``omni_retrieval/subset.py`` (all val videos with an mp4, dev videos
seeded with seed 0 up to 500 in total, trainsplit never used) and written to ``uniav_cache/common``.
Durations: from the Omni file (mp4 header, shorter of the video/audio streams); a rebuilt gallery
uses ``ffmpeg -i`` (athena.encoders.media.probe, container duration), which can differ slightly.

GT for the main evaluation = the Omni manifests (val 3030 + dev 773 = 3803 events, already clamped
to the mp4), so the event set is the one Omni was evaluated on. The sanity check against Athena's
``final_eval`` uses ``youcookii_annotations_trainval.json`` (``annotations[:16]``, 3031 val events).
"""

from __future__ import annotations

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import Config
from .io_utils import read_json, read_jsonl, write_json

VAL_MANIFEST = "val_omni_video.jsonl"
DEV_MANIFEST = "train_omni_video.dev.jsonl"
TRAIN_MANIFEST = "train_omni_video.trainsplit.jsonl"
NMAX = 16   # Athena keeps the first 16 annotations of a video (libs/datasets/youcook2_cap.py)


def manifest_video_id(record: dict) -> str:
    return os.path.splitext(os.path.basename(record["video"]))[0]


def manifest_event_id(record: dict) -> int:
    """``"<video_id>_<i>"`` -> ``i`` (video ids may contain underscores, so split from the right)."""
    return int(str(record["id"]).rsplit("_", 1)[1])


# --------------------------------------------------------------------------- #
# GT                                                                          #
# --------------------------------------------------------------------------- #

def parse_manifest(records: list[dict], split: str) -> dict[str, list[dict]]:
    """Omni manifest records -> ``{video_id: [{event_id, ts, te, caption, split}]}`` sorted by event id."""
    out: dict[str, list[dict]] = {}
    for r in records:
        ts, te = (float(x) for x in r["timestamps"])
        out.setdefault(manifest_video_id(r), []).append(
            {"event_id": manifest_event_id(r), "ts": ts, "te": te, "caption": str(r["text"]), "split": split})
    for v in out.values():
        v.sort(key=lambda e: e["event_id"])
    return out


def load_gt_manifest(meta_dir) -> dict[str, list[dict]]:
    meta = Path(meta_dir)
    gt = parse_manifest(read_jsonl(meta / VAL_MANIFEST), "val")
    for vid, evs in parse_manifest(read_jsonl(meta / DEV_MANIFEST), "dev").items():
        if vid in gt:
            raise ValueError(f"{vid} is in both the val and the dev manifest")
        gt[vid] = evs
    return gt


def load_gt_annotations(path, subset: str = "validation", nmax: int = NMAX) -> dict[str, dict]:
    """``youcookii_annotations_trainval.json`` -> ``{video_id: {duration, events}}`` as Athena scores it."""
    with open(path, encoding="utf-8") as f:
        db = json.load(f)["database"]
    out = {}
    for vid, x in db.items():
        if subset and x["subset"] != subset:
            continue
        evs = [{"event_id": i, "ts": float(a["segment"][0]), "te": float(a["segment"][1]),
                "caption": a["sentence"], "split": "val" if x["subset"] == "validation" else "train"}
               for i, a in enumerate(x["annotations"][:nmax])]
        if evs:
            out[vid] = {"duration": float(x["duration"]), "events": evs}
    return out


# --------------------------------------------------------------------------- #
# Gallery                                                                     #
# --------------------------------------------------------------------------- #

def select_gallery(val_ids: set[str], dev_ids: set[str], available: set[str], n_videos: int = 500,
                   seed: int = 0, exclude: set[str] = frozenset()) -> list[dict]:
    """Same rule as ``omni_retrieval.subset.select_subset`` (non-smoke): all val, then seeded dev."""
    val = sorted(val_ids & available - exclude)
    dev = sorted(dev_ids & available - exclude - set(val))
    rng = random.Random(seed)
    chosen_val = val if n_videos >= len(val) else rng.sample(val, n_videos)
    chosen_dev = rng.sample(dev, min(max(n_videos - len(chosen_val), 0), len(dev)))
    rows = [{"video_id": v, "split": "val"} for v in chosen_val] + [{"video_id": v, "split": "dev"} for v in chosen_dev]
    return sorted(rows, key=lambda r: r["video_id"])


def _rebuild_gallery(cfg: Config) -> list[dict]:
    from .athena_api import probe
    meta = Path(cfg.meta_dir)
    val = {manifest_video_id(r) for r in read_jsonl(meta / VAL_MANIFEST)}
    dev = {manifest_video_id(r) for r in read_jsonl(meta / DEV_MANIFEST)}
    train = {manifest_video_id(r) for r in read_jsonl(meta / TRAIN_MANIFEST)}
    available = {f[:-4] for f in os.listdir(cfg.video_src) if f.endswith(".mp4")}
    rows = select_gallery(val, dev, available, 500, cfg.seed, exclude=train)
    with ThreadPoolExecutor(8) as pool:
        durs = list(pool.map(lambda r: probe(os.path.join(cfg.video_src, r["video_id"] + ".mp4"))["duration"], rows))
    bad = [r["video_id"] for r, d in zip(rows, durs) if not d]
    if bad:
        print(f"[gallery] dropping {len(bad)} unreadable mp4: {bad[:5]}")
    return [dict(r, duration=round(float(d), 3)) for r, d in zip(rows, durs) if d]


def load_gallery(cfg: Config) -> list[dict]:
    """``[{video_id, split ('val'|'dev'), duration}]`` sorted by id. SMOKE: the shipped samples."""
    omni = read_json(cfg.omni_subset) if cfg.omni_subset else None
    omni_rows = {r["video_id"]: r for r in omni["videos"]} if omni else {}
    if cfg.smoke:
        import numpy as np
        rows = []
        for vid in cfg.smoke_ids:
            if vid in omni_rows:
                dur = float(omni_rows[vid]["duration"])
            else:
                dur = float(np.load(os.path.join(cfg.samples_dir, vid + ".npz"))["duration"])
            rows.append({"video_id": vid, "split": "val", "duration": round(dur, 3)})
        return rows
    if omni_rows:
        print(f"[gallery] {len(omni_rows)} videos from {cfg.omni_subset} (read only)")
        return sorted(({"video_id": r["video_id"], "split": r["split"], "duration": float(r["duration"])}
                       for r in omni_rows.values()), key=lambda r: r["video_id"])
    own = cfg.common_dir / "subset_videos.json"
    saved = read_json(own)
    if saved:
        print(f"[gallery] {len(saved['videos'])} videos from {own}")
        return saved["videos"]
    print(f"[gallery] {cfg.omni_subset} not found -> rebuilding with the Omni rule (seed {cfg.seed})")
    rows = _rebuild_gallery(cfg)
    write_json(own, {"settings": {"n_videos": 500, "seed": cfg.seed, "rule": "omni_retrieval.subset"},
                     "videos": rows})
    return rows


def gt_for_gallery(gallery: list[dict], gt_manifest: dict[str, list[dict]],
                   annotations: dict[str, dict] | None = None) -> dict[str, list[dict]]:
    """GT events of every gallery video: the Omni manifest, else (with a warning) the annotation json."""
    out, fallback, missing = {}, [], []
    for r in gallery:
        vid = r["video_id"]
        if vid in gt_manifest:
            out[vid] = gt_manifest[vid]
        elif annotations and vid in annotations:
            out[vid] = [dict(e, split=r["split"]) for e in annotations[vid]["events"]]
            fallback.append(vid)
        else:
            missing.append(vid)
    if fallback:
        print(f"[gt] {len(fallback)} videos not in the Omni manifests, GT from the annotation json: {fallback[:5]}")
    if missing:
        print(f"[gt] WARNING: no GT for {len(missing)} videos: {missing[:5]}")
    return out
