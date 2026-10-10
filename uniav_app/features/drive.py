"""FEATURE_SOURCE 'drive': a folder of ``<video_id>.npz`` already on Drive (repo format), copied into
``feats/drive/`` so the rest of the pipeline reads the same layout as for the other sources."""

from __future__ import annotations

import os
import shutil

from .base import KEYS, FeatureSource


class DriveSource(FeatureSource):
    name = "drive"

    def _prepare(self, todo, durations):
        src_dir = self.cfg.drive_feats_src
        if not src_dir or not os.path.isdir(src_dir):
            raise FileNotFoundError(f"drive_feats_src {src_dir!r} is not a folder")
        import numpy as np
        for v in todo:
            src = os.path.join(src_dir, v + ".npz")
            if not os.path.isfile(src):
                self._fail(v, f"{src} not found")
                continue
            try:
                with np.load(src) as z:
                    lacking = [k for k in KEYS if k not in z.files]
                if lacking:
                    raise KeyError(f"missing keys {lacking}")
                dst = self.npz_path(v)
                shutil.copy2(src, str(dst) + ".part")
                os.replace(str(dst) + ".part", dst)
                self._add(v, dst, durations.get(v), copied_from=src)
            except Exception as e:   # noqa: BLE001
                self._fail(v, repr(e))
