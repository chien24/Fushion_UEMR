"""FEATURE_SOURCE 'extract': the path from raw video, with Athena's own extractor.

mp4 -> ``athena.encoders.internvideo2.InternVideo2AVEncoder.encode_raw`` (which reuses the decoding,
windows and model builders of ``tools/extract_internvideo2.py``, the script that made the training
features): InternVideo2-1B ``v768`` / ``v512`` (2 fps, 224x224, 4-frame window per second) + BEATs
``a768`` (16 kHz, 3 s window per second); duration from ``athena.encoders.media.probe``. Videos
without audio get zero audio rows, as in the repo. Nothing here is reimplemented.

The encoders load on the first video, never at import time.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from ..athena_api import make_encoder
from .base import FeatureRecord, FeatureSource, save_feature_npz


class ExtractSource(FeatureSource):
    name = "extract"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._encoder = None
        self.timings: list[dict] = []

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = make_encoder(self.cfg)
        return self._encoder

    def local_video(self, video_id: str) -> str:
        """Local copy of ``<video_src>/<id>.mp4`` (Drive reads are slow and flaky for ffmpeg)."""
        src = os.path.join(self.cfg.video_src, video_id + ".mp4")
        dst = os.path.join(self.cfg.video_root, video_id + ".mp4")
        if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)):
            os.makedirs(self.cfg.video_root, exist_ok=True)
            shutil.copy2(src, dst + ".part")
            os.replace(dst + ".part", dst)
        return dst

    def encode_file(self, path: str, video_id: str | None = None, duration: float | None = None) -> FeatureRecord:
        """Any video file -> feature npz + meta record (cached: an existing npz is reused)."""
        video_id = video_id or Path(path).stem
        rec = self.records().get(video_id)
        if rec is not None:
            return rec
        t0 = time.time()
        raw = self.encoder.encode_raw(path)
        dt = time.time() - t0
        save_feature_npz(self.npz_path(video_id), raw["v768"], raw["v512"], raw["a768"])
        self.timings.append({"video_id": video_id, "seconds": dt, "rows": len(raw["v768"])})
        return self._add(video_id, self.npz_path(video_id), duration if duration is not None else raw["duration"],
                         probe_duration=raw["duration"], has_audio=bool(raw["has_audio"]), encode_s=round(dt, 2),
                         video=str(path))

    def _prepare(self, todo, durations):
        no_audio, total_rows, total_s = [], 0, 0.0
        for i, v in enumerate(todo, 1):
            try:
                rec = self.encode_file(self.local_video(v), v, durations.get(v))
            except Exception as e:   # noqa: BLE001 - one bad video must not stop the others
                self._fail(v, repr(e))
                continue
            if not rec.extra.get("has_audio", True):
                no_audio.append(v)
            t = self.timings[-1] if self.timings and self.timings[-1]["video_id"] == v else None
            if t:
                total_rows += t["rows"]; total_s += t["seconds"]
            if t and (i <= 3 or i % 20 == 0 or i == len(todo)):
                speed = total_rows / max(total_s, 1e-6)   # video seconds per wall second
                left = sum(float(durations.get(x, 300.0)) for x in todo[i:])
                print(f"[extract] {i}/{len(todo)} {v}: {t['rows']} s video in {t['seconds']:.1f} s "
                      f"| {speed:.1f} video-s/s | ETA {left / speed / 60:.1f} min", flush=True)
                if i == 3 and durations:
                    print(f"[extract] estimate for all {len(durations)} videos with a duration "
                          f"({sum(durations.values()) / 3600:.1f} h of video): "
                          f"{sum(durations.values()) / speed / 3600:.2f} h", flush=True)
        print(f"[extract] videos without audio (zero a768 rows, as the repo does): {len(no_audio)} {no_audio[:5]}")
