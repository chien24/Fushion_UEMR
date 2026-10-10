"""Chuỗi Z for Context Fusion: level-0 backbone features of ``seg_model``.

``EventCaptionModel.forward`` stores ``self._last['feats']``: one ``(1, 1024, 256 / 2**l)`` tensor per
pyramid level (``cat(V_l, A_l)``, 512 + 512 channels). Level 0 has 256 steps for every video, at
``t_sec[j] = (j + 0.5) * n / 256`` (``timegrid.level_times``), so the step in seconds depends on the
video length. Saved per video as ``Z/<video_id>.npz``:

    z      (256, 1024) float16
    t_sec  (256,)      float32, centre time of each step
    meta   JSON string {level, n, duration, formula, source: 'seg_model', feature_source, checkpoint_run}
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .timegrid import FORMULA, level_times

LEVEL = 0


def level_features(model, level: int = LEVEL) -> np.ndarray:
    """``(T_l, 1024)`` features of the last forward pass of an ``EventCaptionModel``."""
    f = model._last["feats"][level][0]          # (1024, T_l)
    return f.float().cpu().numpy().T


def save_z(path, z: np.ndarray, n: int, duration: float | None, feature_source: str, run: str = "",
           time_params: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"level": LEVEL, "n": int(n), "duration": duration, "formula": FORMULA, "source": "seg_model",
            "feature_source": feature_source, "checkpoint_run": run, "shape": list(z.shape)}
    t = level_times(int(n), LEVEL, **(time_params or {}))
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        np.savez(f, z=z.astype(np.float16), t_sec=t.astype(np.float32), meta=np.array(json.dumps(meta)))
    os.replace(tmp, path)


def load_z(path) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(path) as d:
        return d["z"], d["t_sec"], json.loads(str(d["meta"]))


def z_ok(path, feature_source: str) -> bool:
    """An existing Z file made from the same feature source (another source is redone)."""
    if not os.path.isfile(path):
        return False
    try:
        return load_z(path)[2].get("feature_source") == feature_source
    except Exception:   # noqa: BLE001 - truncated file: redo
        return False


def export_z(pipe, records: dict, out_dir, overwrite: bool = False) -> int:
    """Z for any list of videos (e.g. the train videos later, for Context Fusion), without detection
    outputs. ``records``: ``{video_id: FeatureRecord}`` from a feature source."""
    import torch

    from .features import load_arrays
    from .timegrid import spec_params
    done = 0
    for vid, rec in records.items():
        path = Path(out_dir) / f"{vid}.npz"
        if not overwrite and z_ok(path, rec.source):
            continue
        visual, audio = load_arrays(rec, pipe.spec)
        fv, fa, n = pipe.spec.prepare(visual, audio)
        with torch.no_grad():
            pipe.seg_model(fv.to(pipe.device), fa.to(pipe.device))
        save_z(path, level_features(pipe.seg_model), n, rec.duration, rec.source, pipe.run, spec_params(pipe.spec))
        done += 1
    print(f"[Z] wrote {done} files -> {out_dir}")
    return done
