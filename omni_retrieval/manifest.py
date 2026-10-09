"""Segment table, deduplication, and the encoder work lists.

Every (partition, video, k) is one row of the segment table, named
``<vid>__<partition>__<k>``. Two rows of the same video whose ``(ts, te)`` agree to
0.01 s share one *segment key* ``<vid>__<ts>_<te>`` and are encoded once; the store
holds one embedding per segment key (``<seg_key>__av``), and ``collect`` maps rows back
onto it.

``segs_todo.jsonl`` lists only the segment keys the store does not have yet, so adding
a partition later encodes its new segments and nothing else. Its records are in the
video-only training layout ``LazySupervisedDataset`` reads -- the same as
``val_omni_video.jsonl`` minus the caption turn, so the forward pass returns the
AV embedding and skips the text branch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import Config
from .partitions import build_all, clamp, save_partitions, summary
from .subset import build_subset, load_events, video_id

# The human turn of every Omni YouCook2 record (scripts/convert_youcookii.py::PROMPT).
PROMPT = "<video>\nPlease describe the video."


def seg_key(vid: str, ts: float, te: float) -> str:
    return f"{vid}__{ts:.2f}_{te:.2f}"


def segment_table(parts: dict[str, dict]) -> list[dict]:
    rows = []
    for name, by_video in parts.items():
        for vid in sorted(by_video):
            for k, (ts, te) in enumerate(by_video[vid]):
                rows.append({"key": f"{vid}__{name}__{k}", "video_id": vid, "partition": name,
                             "k": k, "ts": ts, "te": te, "seg_key": seg_key(vid, ts, te)})
    return rows


def unique_segments(rows: list[dict]) -> dict[str, dict]:
    """``seg_key -> {video_id, ts, te, refs}``; the window is the 0.01 s-rounded one."""
    out: dict[str, dict] = {}
    for r in rows:
        u = out.setdefault(r["seg_key"], {"video_id": r["video_id"], "ts": round(r["ts"], 2),
                                          "te": round(r["te"], 2), "refs": []})
        u["refs"].append(r["key"])
    return out


def caption_table(events: dict[str, list[dict]], subset: list[dict]) -> list[dict]:
    """Queries: every val caption of a query-source video in the subset.

    ``gt_seg_key`` is the key of the caption's own clip in the ``gt`` partition, built
    with the same clamp/rounding, so the text->clip sanity check can find it.
    """
    durations = {r["video_id"]: r["duration"] for r in subset if r["is_query_source"]}
    out = []
    for r in events["val"]:
        v = video_id(r)
        if v not in durations:
            continue
        ts, te = float(r["timestamps"][0]), float(r["timestamps"][1])
        seg = clamp([[ts, te]], durations[v])
        out.append({"caption_id": r["id"], "video_id": v, "sentence": r["text"], "gt_ts": ts,
                    "gt_te": te, "gt_seg_key": seg_key(v, *seg[0]) if seg else ""})
    return out


# --------------------------------------------------------------------------- #
# Store: what the encoder has already written                                 #
# --------------------------------------------------------------------------- #

def store_files(store) -> list[Path]:
    """The ``.npz`` files of a store: a chunk directory, or one CLI-style file."""
    store = Path(store)
    if store.is_dir():
        return sorted(p for p in store.glob("chunk_*.npz"))
    return [store] if store.is_file() else []


def store_keys(store, modality: str) -> set[str]:
    """Ids already encoded (``<id>__<modality>`` keys, suffix stripped)."""
    suffix = f"__{modality}"
    done = set()
    for path in store_files(store):
        with np.load(path) as blob:
            done.update(k[: -len(suffix)] for k in blob.files if k.endswith(suffix))
    return done


def read_store(store, modality: str) -> dict[str, np.ndarray]:
    suffix = f"__{modality}"
    out = {}
    for path in store_files(store):
        with np.load(path) as blob:
            for k in blob.files:
                if k.endswith(suffix):
                    out[k[: -len(suffix)]] = blob[k]
    return out


# --------------------------------------------------------------------------- #
# Work lists                                                                  #
# --------------------------------------------------------------------------- #

def write_jsonl(path, records) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def segment_records(segments: dict[str, dict], skip: set[str]) -> list[dict]:
    todo = [(u["video_id"], u["ts"], u["te"], k, u) for k, u in segments.items() if k not in skip]
    return [{"id": k, "type": "retrieval", "conversations": [{"from": "human", "value": PROMPT}],
             "video": f"{vid}.mp4", "timestamps": [ts, te], "refs": u["refs"]}
            for vid, ts, te, k, u in sorted(todo, key=lambda t: t[:3])]


def caption_records(captions: list[dict], skip: set[str]) -> list[dict]:
    return [{"id": c["caption_id"], "text": c["sentence"]} for c in captions if c["caption_id"] not in skip]


def prepare(cfg: Config, write: bool = True, verbose: bool = True) -> dict:
    """Subset -> partitions -> segment table -> todo lists. Safe to rerun."""
    subset = build_subset(cfg, write=write)
    events = load_events(cfg.meta_dir)
    parts = build_all(subset, events, cfg.extra_partitions)
    rows = segment_table(parts)
    segments = unique_segments(rows)
    captions = caption_table(events, subset)

    done_av = store_keys(cfg.av_store, "av")
    done_text = store_keys(cfg.text_store, "text")
    segs_todo = segment_records(segments, done_av)
    caps_todo = caption_records(captions, done_text)
    if write:
        save_partitions(parts, cfg.partitions_dir)
        write_jsonl(cfg.segs_todo, segs_todo)
        write_jsonl(cfg.caps_todo, caps_todo)

    if verbose:
        n_val = sum(r["is_query_source"] for r in subset)
        print(f"videos: {len(subset)} (val {n_val}, dev distractors {len(subset) - n_val})")
        print(f"{'partition':<14}{'videos':>8}{'segments':>10}{'vec/video':>11}{'median len (s)':>16}")
        for s in summary(parts):
            print(f"{s['partition']:<14}{s['videos']:>8}{s['segments']:>10}{s['vec/video']:>11}"
                  f"{s['median len (s)']:>16}")
        print(f"segment rows: {len(rows)} | unique clips after dedup: {len(segments)} "
              f"| already encoded: {len(segments) - len(segs_todo)} | to encode: {len(segs_todo)}")
        print(f"captions (queries): {len(captions)} | to encode: {len(caps_todo)}")
    return {"subset": subset, "parts": parts, "rows": rows, "segments": segments,
            "captions": captions, "segs_todo": segs_todo, "caps_todo": caps_todo}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build segs_todo.jsonl / caps_todo.jsonl.")
    p.add_argument("--dry-run", action="store_true", help="print counts and samples, write nothing")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--tag", default="ft")
    p.add_argument("--n-videos", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--extra", nargs="*", default=[], metavar="NAME=JSON",
                   help="extra partitions, e.g. uniav_pred=/path/uniav_pred.json")
    a = p.parse_args(argv)
    extra = dict(item.split("=", 1) for item in a.extra)
    cfg = Config.colab(
        smoke=a.smoke, tag=a.tag, n_videos=a.n_videos, seed=a.seed, extra_partitions=extra)
    out = prepare(cfg, write=not a.dry_run)
    print("\nsample segs_todo.jsonl:")
    for r in out["segs_todo"][:3]:
        print(" ", json.dumps(r, ensure_ascii=False))
    print("sample caps_todo.jsonl:")
    for r in out["caps_todo"][:2]:
        print(" ", json.dumps(r, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
