"""HT-Step val_seen as a UEMR gallery (docs/HTSTEP_0SHOT_UEMR.md §2): video choice, queries, headline statistics.

Inputs are the files of the HF repos, read as they are:

* ``htstep-feats/anno/htstep.json``: ``{video_id: {role, duration, activity, segments, captions}}``, only the
  ``Full Match`` steps (``captions[i]`` = headline of ``segments[i]``);
* ``htstep-videos/manifest.json``: ``{"shards": {tar: {"size", "videos": [video_id, ...]}}}``;
* YouCook2 ids: ``youcookii_annotations_trainval.json`` (``{"database": {id: ...}}``).

A query is one ``Full Match`` step of a kept video; its text is the headline and its id ``<video_id>#<i>`` (index in
``htstep.json``). Times are clamped to the mp4 like every YouCook2 partition (``omni_retrieval.partitions.clamp``),
so ``gt_seg_key`` is the key of the step's clip in the ``gt`` partition.
"""

from __future__ import annotations

import json
import os
import random
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from omni_retrieval.manifest import seg_key
from omni_retrieval.partitions import clamp

VAL_ROLE = "val_seen"


def normalize(text: str) -> str:
    """Headline key of the multi-positive scoring: lower case, punctuation removed, whitespace collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(text).lower())).strip()


def yc2_video_ids(trainval_json) -> set[str]:
    return set(json.loads(Path(trainval_json).read_text(encoding="utf-8"))["database"])


def video_tars(manifest: dict) -> dict[str, str]:
    """``video_id -> tar`` from ``htstep-videos/manifest.json``; a video listed in two tars is an error."""
    out = {}
    for tar, shard in manifest["shards"].items():
        for v in shard["videos"]:
            if v in out and out[v] != tar:
                raise ValueError(f"{v} is in {out[v]} and {tar}")
            out[v] = tar
    return out


def select_videos(htstep: dict, tars: dict[str, str], yc2_ids: set[str]) -> tuple[list[str], dict]:
    """The rule of §2.2: ``val_seen``, an mp4 in the video repo, at least one ``Full Match`` step, not a YouCook2
    video. Returns ``(sorted ids, counts)``; every count is over ``val_seen`` videos."""
    val = {v: x for v, x in htstep.items() if x["role"] == VAL_ROLE}
    has_mp4 = {v for v in val if v in tars}
    has_step = {v for v in val if len(val[v]["segments"])}
    in_yc2 = {v for v in val if v in yc2_ids}
    keep = sorted(has_mp4 & has_step - in_yc2)
    counts = {"val_seen": len(val), "with mp4": len(has_mp4), "with >= 1 Full Match step": len(has_step),
              "YouCook2 id": len(in_yc2), "YouCook2 id among mp4 + step": len(has_mp4 & has_step & in_yc2),
              "kept": len(keep)}
    return keep, counts


def steps(htstep: dict, videos) -> list[dict]:
    """Every ``Full Match`` step of ``videos``: ``{caption_id, video_id, sentence, headline, ts, te}`` (unclamped)."""
    out = []
    for v in sorted(videos):
        x = htstep[v]
        if len(x["segments"]) != len(x["captions"]):
            raise ValueError(f"{v}: {len(x['segments'])} segments, {len(x['captions'])} captions")
        for i, ((ts, te), cap) in enumerate(zip(x["segments"], x["captions"])):
            out.append({"caption_id": f"{v}#{i}", "video_id": v, "sentence": cap, "headline": normalize(cap),
                        "ts": float(ts), "te": float(te)})
    return out


def queries(step_rows: list[dict], durations: dict[str, float]) -> tuple[list[dict], list[dict]]:
    """``(captions, dropped)``. A caption has the ``text.npz`` fields (``caption_id, video_id, sentence, gt_ts,
    gt_te, gt_seg_key``) plus ``headline``; ``gt_ts / gt_te`` are the clamped step. A step left empty by the clamp
    (it starts in the last second of the mp4 or after it) is dropped: it has no clip to find."""
    out, dropped = [], []
    for s in step_rows:
        seg = clamp([[s["ts"], s["te"]]], durations[s["video_id"]])
        if not seg:
            dropped.append(s)
            continue
        ts, te = seg[0]
        out.append({"caption_id": s["caption_id"], "video_id": s["video_id"], "sentence": s["sentence"],
                    "headline": s["headline"], "gt_ts": ts, "gt_te": te, "gt_seg_key": seg_key(s["video_id"], ts, te)})
    return out, dropped


def gt_partition(captions: list[dict]) -> dict[str, list[list[float]]]:
    """R7: the clamped steps of every video (identical windows kept once)."""
    out = defaultdict(set)
    for c in captions:
        out[c["video_id"]].add((c["gt_ts"], c["gt_te"]))
    return {v: sorted([list(s) for s in segs]) for v, segs in out.items()}


def headline_stats(captions: list[dict], top: int = 20) -> dict:
    """How often a headline is shared between videos of the gallery (why strict scoring is too harsh)."""
    vids = defaultdict(set)
    for c in captions:
        vids[c["headline"]].add(c["video_id"])
    shared = [len(vids[c["headline"]]) >= 2 for c in captions]
    return {"queries": len(captions), "distinct headlines": len(vids),
            "% queries whose headline is in >= 2 videos": 100 * float(np.mean(shared)) if captions else 0.0,
            "mean videos per headline": float(np.mean([len(s) for s in vids.values()])) if vids else 0.0,
            "top": [{"headline": h, "videos": len(s)} for h, s in
                    sorted(vids.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:top]]}


def cross_check(step_rows: list[dict], official: dict, headline_of: dict[int, str], n: int = 50,
                seed: int = 0) -> tuple[list[dict], list[dict]]:
    """``n`` random steps (seed) against the HT-Step repo: the same video must have an official ``Full Match`` step
    with the same headline and the same times (1e-3 s). ``official`` = ``data/annotations.json``, ``headline_of`` =
    ``global_step_index -> headline`` of ``taxonomy.csv``. Returns ``(sampled, mismatches)``."""
    sample = random.Random(seed).sample(step_rows, min(n, len(step_rows)))
    bad = []
    for s in sample:
        ann = official.get(s["video_id"])
        ok = ann is not None and any(
            a["partial"] == "Full Match" and headline_of.get(int(a["id"])) == s["sentence"]
            and abs(float(a["segment"][0]) - s["ts"]) < 1e-3 and abs(float(a["segment"][1]) - s["te"]) < 1e-3
            for a in ann["annotations"])
        if not ok:
            bad.append(s)
    return sample, bad


def extract_members(tar_path, wanted: dict[str, str]) -> list[str]:
    """Copy the tar members whose file name is a key of ``wanted`` to ``wanted[name]`` (written to ``.part``, then
    renamed, so an interrupted copy never looks finished). Nothing else of the tar is written. Returns the names
    found."""
    found = []
    with tarfile.open(tar_path) as t:
        for m in t:
            name = os.path.basename(m.name)
            if not m.isfile() or name not in wanted:
                continue
            dst = Path(wanted[name])
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            with t.extractfile(m) as src, open(tmp, "wb") as out:
                while chunk := src.read(1 << 24):
                    out.write(chunk)
            os.replace(tmp, dst)
            found.append(name)
    return found


def feature_format(z) -> dict:
    """``{key: (shape, dtype)}`` of a feature npz, to compare HT-Step with the YouCook2 files Athena was run on."""
    return {k: (tuple(z[k].shape), str(z[k].dtype)) for k in sorted(z.files)}
