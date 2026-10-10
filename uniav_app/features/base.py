"""Common interface of the feature sources.

Every source ends with the same files, so the steps after it never know where features came from:

    <cache_root>/feats/<source>/<video_id>.npz     v768 (T, 768), v512 (T, 512), a768 (T, 768), float16
    <cache_root>/feats/<source>/feats_meta.jsonl   {video_id, n, duration, source, ...}
    <cache_root>/feats/<source>/failed.jsonl       {video_id, error, source}

(``samples`` reads the npz shipped in third_party/athena/athena/samples in place.) ``prepare`` skips
videos that already have a feature file, so a cell interrupted by a Colab disconnect is simply rerun.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import Config
from ..io_utils import append_jsonl, latest_by, read_jsonl

KEYS = ("v768", "v512", "a768")


@dataclass
class FeatureRecord:
    video_id: str
    path: str
    n: int                       # feature rows (min of visual and audio rows) = Athena's n
    duration: float | None       # seconds (gallery / mp4 header), only used to clip times
    source: str
    extra: dict = field(default_factory=dict)


def save_feature_npz(path, v768, v512, a768) -> None:
    """Write the repo's feature format (float16) atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:   # a file object: np.savez would otherwise append ".npz" to the name
        np.savez(f, v768=np.asarray(v768, np.float16), v512=np.asarray(v512, np.float16),
                 a768=np.asarray(a768, np.float16))
    os.replace(tmp, path)


def feature_rows(path) -> int:
    with np.load(path) as z:
        return int(min(z["v768"].shape[0], z["a768"].shape[0]))


def load_arrays(record: FeatureRecord, spec):
    """(visual, audio) in the checkpoint's format, through ``athena.features.FeatureSpec.from_npz``."""
    with np.load(record.path) as z:
        return spec.from_npz(z)


class FeatureSource:
    name = "base"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dir = cfg.feats_dir_for(self.name)

    # --- files -------------------------------------------------------------
    @property
    def meta_path(self) -> Path:
        return self.dir / "feats_meta.jsonl"

    @property
    def failed_path(self) -> Path:
        return self.dir / "failed.jsonl"

    def npz_path(self, video_id: str) -> Path:
        return self.dir / f"{video_id}.npz"

    def records(self) -> dict[str, FeatureRecord]:
        out = {}
        for vid, r in latest_by(read_jsonl(self.meta_path)).items():
            path = r.get("path") or str(self.npz_path(vid))
            if os.path.isfile(path):
                extra = {k: v for k, v in r.items() if k not in ("video_id", "path", "n", "duration", "source")}
                out[vid] = FeatureRecord(vid, path, int(r["n"]), r.get("duration"), self.name, extra)
        return out

    def failed(self) -> dict[str, dict]:
        return latest_by(read_jsonl(self.failed_path))

    # --- bookkeeping ---------------------------------------------------------
    def _add(self, video_id: str, path, duration: float | None, **extra) -> FeatureRecord:
        n = feature_rows(path)
        rec = {"video_id": video_id, "n": n, "duration": None if duration is None else round(float(duration), 3),
               "source": self.name, **extra}
        if Path(path) != self.npz_path(video_id):
            rec["path"] = str(path)
        append_jsonl(self.meta_path, [rec])
        return FeatureRecord(video_id, str(path), n, rec["duration"], self.name, extra)

    def _fail(self, video_id: str, error) -> None:
        print(f"[{self.name}] FAIL {video_id}: {error}")
        append_jsonl(self.failed_path, [{"video_id": video_id, "error": str(error)[:500], "source": self.name,
                                         "time": time.strftime("%Y-%m-%d %H:%M:%S")}])

    # --- main entry ----------------------------------------------------------
    def prepare(self, video_ids, durations: dict[str, float] | None = None) -> dict[str, FeatureRecord]:
        """Features of ``video_ids`` (prepared if missing) -> ``{video_id: FeatureRecord}``."""
        video_ids = list(dict.fromkeys(video_ids))
        durations = durations or {}
        self.dir.mkdir(parents=True, exist_ok=True)
        have = self.records()
        todo = [v for v in video_ids if v not in have]
        print(f"[{self.name}] {len(video_ids)} videos: {len(video_ids) - len(todo)} ready, {len(todo)} to prepare "
              f"-> {self.dir}")
        if todo:
            self._prepare(todo, durations)
        recs = self.records()
        missing = [v for v in video_ids if v not in recs]
        if missing:
            print(f"[{self.name}] WARNING: {len(missing)} videos without features (see {self.failed_path}): {missing[:5]}")
        out = {}
        for v in video_ids:
            if v in recs:
                r = recs[v]
                if v in durations:   # the gallery duration wins (Q8); the stored one is kept in meta
                    r.duration = float(durations[v])
                out[v] = r
        return out

    def _prepare(self, todo: list[str], durations: dict[str, float]) -> None:
        raise NotImplementedError


def check_n_vs_duration(records: dict[str, FeatureRecord], limit: float = 2.0) -> list[dict]:
    """Videos whose feature rows ``n`` and duration differ by more than ``limit`` seconds (Q8)."""
    bad = [{"video_id": v, "n": r.n, "duration": r.duration, "diff": round(r.n - r.duration, 2)}
           for v, r in records.items() if r.duration is not None and abs(r.n - r.duration) > limit]
    print(f"[feats] |n - duration| > {limit} s: {len(bad)} / {len(records)} videos")
    for b in bad[:10]:
        print("   ", b)
    return bad
