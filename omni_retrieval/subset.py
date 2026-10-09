"""Choose the gallery videos and read their real durations.

Queries are always val captions (val = test in UEMR). The gallery is every val video
with an mp4, topped up with dev videos as distractors (their captions are never
queries). Trainsplit videos -- what the adapter was fine-tuned on -- are never used.

Durations come from the mp4 header (``probe_duration``, copied from Omni-fix
``convert_youcookii.py``: shorter of the video and audio streams), not from the metadata: some
downloads are shorter than the original upload.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import Config

VAL_MANIFEST = "val_omni_video.jsonl"
DEV_MANIFEST = "train_omni_video.dev.jsonl"
TRAIN_MANIFEST = "train_omni_video.trainsplit.jsonl"


def video_id(record: dict) -> str:
    return os.path.splitext(os.path.basename(record["video"]))[0]


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_events(meta_dir) -> dict[str, list[dict]]:
    """Event records of val and dev (one per caption), from the Omni manifests.

    These manifests were already clamped to the mp4 on disk by ``convert_youcookii.py``,
    so ``gt`` segments are exactly the clips the fine-tune eval scored.
    """
    meta = Path(meta_dir)
    return {"val": read_jsonl(meta / VAL_MANIFEST), "dev": read_jsonl(meta / DEV_MANIFEST)}


def train_videos(meta_dir) -> set[str]:
    path = Path(meta_dir) / TRAIN_MANIFEST
    return {video_id(r) for r in read_jsonl(path)} if path.is_file() else set()


def probe_duration(path: str) -> float | None:
    """Seconds both streams cover, from the header only (copy of Omni-fix
    ``scripts/convert_youcookii.py::probe_duration``).

    The shorter of the video and audio stream: a frame window past the audio's end would
    pair with a silent tail. ``None`` when the file cannot be opened or has no audio stream.
    The container is closed as soon as the header is read.
    """
    import av
    try:
        with av.open(path) as container:
            if not container.streams.video or not container.streams.audio:
                return None
            spans = [
                float(s.duration * s.time_base)
                for s in (container.streams.video[0], container.streams.audio[0])
                if s.duration and s.time_base
            ]
            if not spans and container.duration:
                spans = [container.duration / 1e6]
            return min(spans) if spans else None
    except Exception:  # noqa: BLE001 - PyAV raises several error types
        return None


def probe_durations(video_dir, video_ids, workers: int = 8) -> dict[str, float | None]:
    ids = sorted(video_ids)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        spans = pool.map(lambda v: probe_duration(os.path.join(video_dir, f"{v}.mp4")), ids)
        return dict(zip(ids, spans))


def select_subset(events: dict[str, list[dict]], available: set[str], n_videos: int, seed: int,
                  smoke: bool = False, smoke_videos: int = 5, exclude: set[str] = frozenset()) -> list[dict]:
    """Pick the gallery. Returns ``[{video_id, split, is_query_source}]`` sorted by id.

    Val first (all of it, or a seeded sample when ``n_videos`` is smaller), then dev
    distractors, seeded, to fill up to ``n_videos``. Smoke mode keeps the same rule at
    ``smoke_videos`` but forces one dev video in so the distractor path is exercised.
    """
    val = sorted({video_id(r) for r in events["val"]} & available - exclude)
    dev = sorted({video_id(r) for r in events["dev"]} & available - exclude - set(val))
    rng = random.Random(seed)
    if smoke:
        n_dev = 1 if dev and smoke_videos > 1 else 0
        chosen_val = rng.sample(val, min(smoke_videos - n_dev, len(val)))
        chosen_dev = rng.sample(dev, n_dev)
    else:
        chosen_val = val if n_videos >= len(val) else rng.sample(val, n_videos)
        chosen_dev = rng.sample(dev, min(max(n_videos - len(chosen_val), 0), len(dev)))
        if len(chosen_val) + len(chosen_dev) < n_videos:
            print(f"[subset] WARNING: N_VIDEOS={n_videos} but only {len(chosen_val)} val + "
                  f"{len(chosen_dev)} dev videos are available -> gallery of "
                  f"{len(chosen_val) + len(chosen_dev)} (train videos are never used).")
    rows = [{"video_id": v, "split": "val", "is_query_source": True} for v in chosen_val]
    rows += [{"video_id": v, "split": "dev", "is_query_source": False} for v in chosen_dev]
    return sorted(rows, key=lambda r: r["video_id"])


def build_subset(cfg: Config, workers: int = 8, write: bool = True) -> list[dict]:
    """Select the gallery, attach header durations, and save ``subset_videos.json``.

    Reuses the saved file when its settings match, so rerunning the cell is free and
    never re-probes Drive.
    """
    path = cfg.subset_path
    settings = {"n_videos": cfg.n_videos, "seed": cfg.seed, "smoke": cfg.smoke,
                "smoke_videos": cfg.smoke_videos}
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("settings") == settings:
            return saved["videos"]

    events = load_events(cfg.meta_dir)
    available = {f[:-4] for f in os.listdir(cfg.video_src) if f.endswith(".mp4")}
    excluded = train_videos(cfg.meta_dir)
    rows = select_subset(events, available, cfg.n_videos, cfg.seed, cfg.smoke,
                         cfg.smoke_videos, exclude=excluded)
    assert not {r["video_id"] for r in rows} & excluded, "a train video slipped into the gallery"

    durations = probe_durations(cfg.video_src, [r["video_id"] for r in rows], workers)
    unreadable = [v for v, d in durations.items() if not d]
    if unreadable:
        print(f"[subset] dropping {len(unreadable)} unreadable / audio-less mp4: {unreadable[:5]}")
    rows = [dict(r, duration=round(durations[r["video_id"]], 3)) for r in rows if durations[r["video_id"]]]

    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"settings": settings, "videos": rows}, indent=1), encoding="utf-8")
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Select the gallery videos (prints a summary).")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--n-videos", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-write", action="store_true")
    a = p.parse_args(argv)
    cfg = Config.colab(smoke=a.smoke, n_videos=a.n_videos, seed=a.seed)
    rows = build_subset(cfg, write=not a.no_write)
    n_val = sum(r["is_query_source"] for r in rows)
    print(f"videos: {len(rows)} (val {n_val}, dev distractors {len(rows) - n_val})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
