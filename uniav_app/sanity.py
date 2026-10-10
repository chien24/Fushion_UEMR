"""Sanity check: reproduce the checkpoint's own evaluation (``ck['final_eval']``) on the same split.

``train.py::evaluate`` scores the main weights (best_cap, i.e. ``use_seg_weights=False``) on the 394 val
videos: GT = ``annotations[:16]`` of ``youcookii_annotations_trainval.json``, duration from that json,
k = min(#GT, #pred) highest-scoring proposals, a GT counts when its best tIoU among them is >= t;
mIoU = mean of those best tIoUs. Expected for athena.pth: R@0.3/0.5/0.7 = 72.6 / 53.6 / 30.9, mIoU 49.9.
Any gap > ``tol`` points stops the notebook (``strict``), except in SMOKE.

Reference only (printed, never stops): F1@0.5 of the API selection (seg weights, min_score 0.40,
max_overlap 0.3) with ``athena.calibrate.match``, against the 0.535 written in ``athena/config.py``.
"""

from __future__ import annotations

import numpy as np
import torch

from .athena_api import checkpoint_info, greedy_match_count, load_pipeline, select_events
from .config import Config
from .features import FeatureRecord, load_arrays
from .io_utils import write_csv, write_json
from .metrics import best_tious
from .subset import load_gt_annotations

KEYS = ("R@0.3", "R@0.5", "R@0.7", "mIoU")
REF_F1 = 0.535


@torch.no_grad()
def _proposals(pipe, rec: FeatureRecord, duration: float) -> tuple[np.ndarray, np.ndarray]:
    visual, audio = load_arrays(rec, pipe.spec)
    fv, fa, n = pipe.spec.prepare(visual, audio)
    segs, scores, _ = pipe.seg_model(fv.to(pipe.device), fa.to(pipe.device))
    return pipe.spec.to_seconds(segs, n, duration), scores


def official_parity(cfg: Config, records: dict[str, FeatureRecord], pipe_seg=None, strict: bool = True,
                    tol: float = 1.0) -> dict:
    ann = load_gt_annotations(cfg.annotations, "validation")
    vids = sorted(v for v in ann if v in records)
    info = checkpoint_info(cfg.checkpoint)
    ref = info.get("final_eval") or {}
    print(f"[sanity] {len(vids)} / {len(ann)} val videos of the annotation json have features ({cfg.source})")

    # 1. main weights, k = #GT  (train.py::evaluate)
    pipe_main = load_pipeline(cfg, use_seg_weights=False)
    preds, gts = {}, {}
    for v in vids:
        secs, scores = _proposals(pipe_main, records[v], ann[v]["duration"])
        preds[v] = [(float(a), float(b), float(c)) for (a, b), c in zip(secs, scores)]
        gts[v] = [(e["ts"], e["te"]) for e in ann[v]["events"]]
    del pipe_main
    best = np.concatenate([b for b in best_tious(preds, gts, "K_GT").values()] or [np.zeros(0)])
    ours = {f"R@{t}": 100.0 * float((best >= t).mean()) for t in (0.3, 0.5, 0.7)}
    ours["mIoU"] = 100.0 * float(best.mean()) if best.size else 0.0
    rows = []
    for k in KEYS:
        r = ref.get(k)
        rows.append({"metric": k, "ours": ours[k], "final_eval": r, "diff": None if r is None else ours[k] - float(r)})
        print(f"[sanity] {k:6s} ours {ours[k]:6.2f} | final_eval {r if r is None else round(float(r), 2)} | "
              f"diff {'-' if r is None else round(ours[k] - float(r), 2)}")

    # 2. reference: API selection with the seg weights, F1@0.5 by athena.calibrate.match
    pipe_seg = pipe_seg or load_pipeline(cfg, use_seg_weights=True)
    tp = n_pred = n_gt = 0
    for v in vids:
        secs, scores = _proposals(pipe_seg, records[v], ann[v]["duration"])
        keep = select_events(secs, scores, cfg.api_min_score, cfg.api_max_overlap, cfg.api_max_events)
        g = [[e["ts"], e["te"]] for e in ann[v]["events"]]
        tp += greedy_match_count([secs[i] for i in keep], g, 0.5)
        n_pred += len(keep); n_gt += len(g)
    p, r = tp / max(n_pred, 1), tp / max(n_gt, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    print(f"[sanity] reference: API selection F1@0.5 {f1:.3f} (calibrate.py: {REF_F1}) | P {p:.3f} R {r:.3f} "
          f"| {n_pred / max(len(vids), 1):.2f} events/video")
    rows.append({"metric": "F1@0.5 api (seg weights)", "ours": f1, "final_eval": REF_F1, "diff": f1 - REF_F1})

    gaps = [abs(x["diff"]) for x in rows[:4] if x["diff"] is not None]
    ok = bool(gaps) and max(gaps) <= tol and len(vids) == len(ann)
    out = {"n_videos": len(vids), "n_val_in_json": len(ann), "ours": ours, "final_eval": {k: ref.get(k) for k in KEYS},
           "max_gap": max(gaps) if gaps else None, "tol": tol, "pass": ok, "f1_api": f1, "f1_ref": REF_F1,
           "feature_source": cfg.source, "checkpoint_run": info.get("run")}
    write_json(cfg.out_dir / "sanity.json", out)
    write_csv(cfg.analysis_dir / "sanity.csv", rows)
    if ok:
        print(f"[sanity] PASS: max gap {max(gaps):.2f} <= {tol} points on all {len(vids)} val videos")
    else:
        msg = (f"[sanity] FAIL: max gap {out['max_gap']} points (tol {tol}), {len(vids)}/{len(ann)} val videos")
        if strict and not cfg.smoke:
            raise AssertionError(msg)
        print(msg + (" -- SMOKE: not stopping" if cfg.smoke else " -- strict=False: not stopping"))
    return out
