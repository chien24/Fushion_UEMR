"""SMOKE source: the YouCook2 val features shipped with Athena (third_party/athena/athena/samples/<id>.npz,
keys v768, v512, a768, duration). Read in place, no HF token needed."""

from __future__ import annotations

import os

import numpy as np

from .base import FeatureSource


class SamplesSource(FeatureSource):
    name = "samples"

    def _prepare(self, todo, durations):
        for v in todo:
            path = os.path.join(self.cfg.samples_dir, v + ".npz")
            if not os.path.isfile(path):
                self._fail(v, f"{path} not found (only the shipped samples are available in SMOKE)")
                continue
            with np.load(path) as z:
                dur = float(z["duration"]) if "duration" in z.files else None
            self._add(v, path, durations.get(v, dur), sample_duration=dur)
