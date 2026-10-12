"""Train-side cache for the fusion (plan P6 §B1-B3): video list, durations, partitions, Omni work lists.

The gallery code (``omni_retrieval.manifest.prepare``) refuses train videos on purpose, so the train cache
reuses its pieces directly:

* B1  ``train_events`` / ``train_videos`` from ``train_omni_video.trainsplit.jsonl``; ``probe_durations`` writes
      ``durations.json`` (mp4 header, shorter of the video and audio streams, as ``subset_videos.json``);
      ``write_uniav_source`` records the Athena run that produced ``segments_pred.jsonl`` / ``Z``.
* B2  ``build_train_partitions``: ``uniav_pred`` (R6), ``uni_M`` (R3), ``gt`` (R7), ``global`` (R0), with the
      functions of ``partitions_uemr`` / ``partitions`` and their checks; ``save_train_partitions``.
* B3  ``train_captions`` + ``work_lists`` (``segs_todo`` / ``caps_todo`` for ``omni_retrieval.encode``) and
      ``collect_train`` (``segments.npz``, ``text.npz``, ``meta.json`` in ``omni_cache_train/<tag>/``).

Numpy and the standard library only: B1 runs in the UniAV runtime, B3 in the Omni one.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from omni_retrieval.manifest import (caption_records, segment_records, segment_table, seg_key, store_keys,
                                     unique_segments, write_jsonl)
from omni_retrieval.partitions import clamp, uniform
from omni_retrieval.partitions_uemr import _git_commit, _round, check, read_segments, sha256
from omni_retrieval.subset import TRAIN_MANIFEST, probe_durations as _probe, read_jsonl, video_id

TRAIN_PARTITIONS = ("uniav_pred", "uni_M", "gt", "global")
# encode order of plan §B3: R6 and R7 are enough for the first run
TRAIN_PASSES = {1: ("uniav_pred", "gt"), 2: ("uni_M", "global")}


# --------------------------------------------------------------------------- #
# B1                                                                          #
# --------------------------------------------------------------------------- #

def train_events(meta_dir) -> list[dict]:
    """Records of ``train_omni_video.trainsplit.jsonl`` (one per caption, timestamps already clamped to the mp4)."""
    return read_jsonl(Path(meta_dir) / TRAIN_MANIFEST)


def train_videos(meta_dir) -> list[str]:
    return sorted({video_id(r) for r in train_events(meta_dir)})


def probe_durations(video_dir, video_ids, out_path, workers: int = 8) -> dict[str, float]:
    """``{video_id: seconds}`` from the mp4 headers; reuses ``out_path`` and only probes the videos it lacks.

    Unreadable or audio-less files are left out (printed); the caller drops those videos.
    """
    out_path = Path(out_path)
    have = json.loads(out_path.read_text(encoding="utf-8")) if out_path.is_file() else {}
    todo = [v for v in video_ids if v not in have]
    if todo:
        for v, d in _probe(video_dir, todo, workers).items():
            if d:
                have[v] = round(float(d), 3)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(dict(sorted(have.items())), indent=0), encoding="utf-8")
    missing = [v for v in video_ids if v not in have]
    print(f"[durations] {len(video_ids)} videos: {len(video_ids) - len(missing)} with a duration "
          f"({len(todo)} probed now), {len(missing)} unreadable / no audio: {missing[:5]}")
    return {v: have[v] for v in video_ids if v in have}


def write_uniav_source(out_dir, checkpoint: dict, params: dict, durations_path) -> dict:
    """``SOURCE.json`` next to ``segments_pred.jsonl``: checkpoint, theta / K_max / d_min, sha256 of the outputs."""
    out = Path(out_dir)
    files = {}
    for name in ("segments_pred.jsonl", "raw_proposals.jsonl", "predictions.jsonl"):
        if (out / name).is_file():
            files[name] = sha256(out / name)
    z = sorted((out / "Z").glob("*.npz")) if (out / "Z").is_dir() else []
    blob = {"checkpoint": checkpoint, "params": params, "durations": {"path": str(durations_path),
                                                                     "sha256": sha256(durations_path)},
            "files": files, "z_files": len(z), "commit": _git_commit(), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "SOURCE.json").write_text(json.dumps(blob, indent=1), encoding="utf-8")
    return blob


# --------------------------------------------------------------------------- #
# B2                                                                          #
# --------------------------------------------------------------------------- #

def gt_by_video(events: list[dict]) -> dict[str, list[list[float]]]:
    out = defaultdict(list)
    for r in events:
        out[video_id(r)].append([float(r["timestamps"][0]), float(r["timestamps"][1])])
    return dict(out)


def build_train_partitions(segments_path, durations: dict[str, float], events: list[dict]) -> tuple[dict, list[dict]]:
    """R6 / R3 / R7 / R0 for every video of ``durations`` (same rules as the gallery: ``partitions_uemr.build``
    and ``partitions.build_base``), and the checks of ``partitions_uemr.check``."""
    segments = read_segments(segments_path)
    missing = sorted(set(durations) - set(segments))
    if missing:
        raise ValueError(f"{len(missing)} train videos have no line in {segments_path}: {missing[:5]}")
    gt = gt_by_video(events)
    parts = {name: {} for name in TRAIN_PARTITIONS}
    for vid in sorted(durations):
        dur, k = durations[vid], len(segments[vid])
        parts["uniav_pred"][vid] = _round(segments[vid], dur)
        parts["uni_M"][vid] = _round(uniform(dur, k), dur)
        parts["gt"][vid] = clamp(gt.get(vid, []), dur)
        parts["global"][vid] = clamp([[0.0, dur]], dur)
    k_pred = {v: len(segments[v]) for v in durations}
    rows = [check(name, parts[name], durations, k_pred) for name in TRAIN_PARTITIONS]
    return parts, rows


def save_train_partitions(parts: dict, out_dir, segments_path, durations_path, rows: list[dict]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, part in parts.items():
        path = out / f"{name}.json"
        path.write_text(json.dumps(part), encoding="utf-8")
        files[name] = {"file": path.name, "sha256": sha256(path)}
    (out / "SOURCE.json").write_text(json.dumps({
        "segments_pred": {"path": str(segments_path), "sha256": sha256(segments_path)},
        "durations": {"path": str(durations_path), "sha256": sha256(durations_path)},
        "partitions": files, "summary": rows, "commit": _git_commit(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1), encoding="utf-8")


def load_train_partitions(parts_dir, names=TRAIN_PARTITIONS) -> dict[str, dict]:
    return {n: json.loads((Path(parts_dir) / f"{n}.json").read_text(encoding="utf-8")) for n in names}


# --------------------------------------------------------------------------- #
# B3                                                                          #
# --------------------------------------------------------------------------- #

def train_captions(events: list[dict], durations: dict[str, float]) -> list[dict]:
    """Trainsplit captions in the ``text.npz`` layout; ``gt_seg_key`` matches the ``gt`` partition's key."""
    out = []
    for r in events:
        v = video_id(r)
        if v not in durations:
            continue
        ts, te = float(r["timestamps"][0]), float(r["timestamps"][1])
        seg = clamp([[ts, te]], durations[v])
        out.append({"caption_id": r["id"], "video_id": v, "sentence": r["text"], "gt_ts": ts, "gt_te": te,
                    "gt_seg_key": seg_key(v, *seg[0]) if seg else ""})
    return out


def work_lists(parts: dict, captions: list[dict], av_store, text_store, segs_todo, caps_todo) -> dict:
    """Write the encoder work lists (only what the stores lack); return the segment rows and counts."""
    rows = segment_table(parts)
    segments = unique_segments(rows)
    segs = segment_records(segments, store_keys(av_store, "av"))
    caps = caption_records(captions, store_keys(text_store, "text"))
    write_jsonl(segs_todo, segs)
    write_jsonl(caps_todo, caps)
    print(f"[train] partitions {list(parts)} | rows {len(rows)} | unique clips {len(segments)} | "
          f"to encode {len(segs)} | captions {len(captions)}, to encode {len(caps)}")
    return {"rows": rows, "unique": len(segments), "segs_todo": len(segs), "caps_todo": len(caps)}


def collect_train(cfg, parts: dict, captions: list[dict], durations: dict, allow_missing: bool = False) -> dict:
    """``segments.npz``, ``text.npz`` and ``meta.json`` in ``cfg.out_dir`` (an ``omni_retrieval.Config`` whose
    ``cache_root`` is ``omni_cache_train``)."""
    from omni_retrieval.collect import _read_json, _read_lines, collect_segments, collect_text

    out = Path(cfg.out_dir)
    seg = collect_segments(segment_table(parts), cfg.av_store, out / "segments.npz", allow_missing)
    txt = collect_text(captions, cfg.text_store, out / "text.npz", allow_missing)
    meta = {"tag": cfg.tag, "adapter": cfg.adapter, "encoder": _read_json(Path(cfg.av_store) / "encoder_meta.json"),
            "text_encoder": _read_json(Path(cfg.text_store) / "encoder_meta.json"),
            "encode_runs": {m: _read_lines(Path(s) / "runs.jsonl") for m, s in (("av", cfg.av_store),
                                                                              ("text", cfg.text_store))},
            "partitions": {n: sum(len(s) for s in p.values()) for n, p in parts.items()},
            "segments": seg, "text": txt, "n_videos": len(durations), "split": "trainsplit",
            "collected_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[collect train] segments: {seg} | text: {txt} -> {out}")
    return meta

