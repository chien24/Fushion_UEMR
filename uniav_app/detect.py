"""Run Athena's segmentation on features and keep the raw proposals (after soft-NMS, before any selection).

Per video (``AthenaPipeline.process_features`` without its selection and captioning):
  ``fv, fa, n = pipe.spec.prepare(visual, audio)``          resample to 256 steps
  ``segs, scores, _ = pipe.seg_model(fv, fa)``              best_seg weights; <= 100 proposals, grid units,
                                                            Gaussian soft-NMS, best first
  ``secs = pipe.spec.to_seconds(segs, n, duration)``        t = (g + 0.5) * n / 256, clipped
and ``seg_model._last['feats'][0]`` for Z.

``raw_proposals.jsonl``: one line per video
``{video_id, split, duration, n, feature_source, run, time_params, proposals: [{g_s, g_e, ts, te, conf}]}``.
"""

from __future__ import annotations

import time

import torch

from .config import Config
from .features import FeatureRecord, load_arrays
from .io_utils import append_jsonl, latest_by, read_jsonl
from .timegrid import spec_params
from .z_export import level_features, save_z, z_ok


@torch.no_grad()
def detect_features(pipe, visual, audio, duration: float | None) -> dict:
    """Raw proposals (+ level-0 Z) of one video from its features."""
    fv, fa, n = pipe.spec.prepare(visual, audio)
    segs, scores, _ = pipe.seg_model(fv.to(pipe.device), fa.to(pipe.device))
    z0 = level_features(pipe.seg_model)
    dur = float(duration) if duration is not None else float(n)
    secs = pipe.spec.to_seconds(segs, n, dur)
    props = [{"g_s": round(float(g[0]), 4), "g_e": round(float(g[1]), 4), "ts": round(float(s[0]), 3),
              "te": round(float(s[1]), 3), "conf": round(float(c), 5)} for g, s, c in zip(segs, secs, scores)]
    return {"n": int(n), "duration": dur, "proposals": props, "z0": z0}


def run_detection(cfg: Config, pipe, records: dict[str, FeatureRecord], splits: dict[str, str],
                  redo: bool = False) -> dict[str, dict]:
    """Detect every video with features; append to ``raw_proposals.jsonl`` and write ``Z/`` (SAVE_Z).

    Videos already in the file (and with a Z of the same feature source) are skipped unless ``redo``.
    Returns ``{video_id: raw record}`` for the requested videos.
    """
    done = {} if redo else latest_by(read_jsonl(cfg.raw_path))
    todo = [v for v in records if v not in done or done[v].get("feature_source") != records[v].source
            or (cfg.save_z and not z_ok(cfg.z_dir / f"{v}.npz", records[v].source))]
    print(f"[detect] {len(records)} videos: {len(records) - len(todo)} done, {len(todo)} to run "
          f"| seg weights: {'best_seg' if pipe.seg_model is not pipe.model else 'main'} | device {pipe.device}")
    tp = spec_params(pipe.spec)
    t0 = time.time()
    for i, vid in enumerate(todo, 1):
        rec = records[vid]
        visual, audio = load_arrays(rec, pipe.spec)
        out = detect_features(pipe, visual, audio, rec.duration)
        row = {"video_id": vid, "split": splits.get(vid, ""), "duration": out["duration"], "n": out["n"],
               "feature_source": rec.source, "run": pipe.run, "time_params": tp, "proposals": out["proposals"]}
        append_jsonl(cfg.raw_path, [row])
        if cfg.save_z:
            save_z(cfg.z_dir / f"{vid}.npz", out["z0"], out["n"], out["duration"], rec.source, pipe.run, tp)
        done[vid] = row
        if i <= 3 or i % 100 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"[detect] {i}/{len(todo)} {vid}: n={out['n']} {len(out['proposals'])} proposals "
                  f"| {el / i:.2f} s/video", flush=True)
    return {v: done[v] for v in records if v in done}
