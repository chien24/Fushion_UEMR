"""Parity of the video path ('extract') with the stored features ('hf') on a few videos.

Per video: row-wise cosine of ``v768`` and ``a768`` (min / mean over the common rows) and the difference
in rows ``n``. Then the detector runs on both feature sets: R@0.5 (raw proposals, k = K_GT, as Athena's
evaluation) and P/R/F1@0.5 + number of events of the API selection (min_score 0.40; independent of the
theta chosen later). PASS when the mean cosine is >= 0.99 for both streams and every metric differs by
<= 1 point; otherwise a clear warning (the notebook does not stop).
"""

from __future__ import annotations

import random

import numpy as np

from .config import Config
from .detect import detect_features
from .features import FeatureRecord, load_arrays
from .io_utils import write_csv
from .metrics import prf, recall_at_n
from .postprocess import postprocess_api


def parity_ids(cfg: Config, gallery: list[dict], n_random: int = 15) -> list[str]:
    """The 5 SMOKE samples + ``n_random`` other gallery videos (seed ``cfg.seed``)."""
    rest = sorted(r["video_id"] for r in gallery if r["video_id"] not in cfg.smoke_ids)
    return list(cfg.smoke_ids) + random.Random(cfg.seed).sample(rest, min(n_random, len(rest)))


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = min(len(a), len(b))
    a, b = np.asarray(a[:m], np.float32), np.asarray(b[:m], np.float32)
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    return np.sum(a * b, axis=1) / np.clip(na * nb, 1e-12, None)


def feature_parity(rec_a: FeatureRecord, rec_b: FeatureRecord) -> dict:
    out = {"n_a": rec_a.n, "n_b": rec_b.n, "n_diff": rec_a.n - rec_b.n}
    with np.load(rec_a.path) as za, np.load(rec_b.path) as zb:
        for k in ("v768", "a768"):
            c = row_cosine(za[k], zb[k])
            out[f"{k}_cos_min"] = float(c.min()) if c.size else None
            out[f"{k}_cos_mean"] = float(c.mean()) if c.size else None
    return out


def _detect_summary(pipe, cfg: Config, recs: dict[str, FeatureRecord], gt: dict) -> dict:
    raw, sel, gts = {}, {}, {}
    for v, rec in recs.items():
        visual, audio = load_arrays(rec, pipe.spec)
        props = detect_features(pipe, visual, audio, rec.duration)["proposals"]
        raw[v] = [(p["ts"], p["te"], p["conf"]) for p in props]
        sel[v] = [(p["ts"], p["te"], p["conf"]) for p in postprocess_api(props, **cfg.api_params())]
        gts[v] = [(e["ts"], e["te"]) for e in gt[v]]
    p = prf(sel, gts, 0.5)
    return {"R@0.5_raw@K_GT": 100 * recall_at_n(raw, gts, "K_GT", (0.5,))[0.5], "P@0.5_api": 100 * p["precision"],
            "R@0.5_api": 100 * p["recall"], "F1@0.5_api": 100 * p["f1"],
            "events_per_video_api": float(np.mean([len(s) for s in sel.values()]))}


def run_parity(cfg: Config, pipe, rec_extract: dict[str, FeatureRecord], rec_hf: dict[str, FeatureRecord],
               gt: dict) -> dict:
    vids = [v for v in rec_extract if v in rec_hf and v in gt]
    rows = [dict(video_id=v, **feature_parity(rec_extract[v], rec_hf[v])) for v in vids]
    det_e = _detect_summary(pipe, cfg, {v: rec_extract[v] for v in vids}, gt)
    det_h = _detect_summary(pipe, cfg, {v: rec_hf[v] for v in vids}, gt)
    summary = {"n_videos": len(vids),
               "v768_cos_mean": float(np.mean([r["v768_cos_mean"] for r in rows])) if rows else None,
               "v768_cos_min": float(np.min([r["v768_cos_min"] for r in rows])) if rows else None,
               "a768_cos_mean": float(np.mean([r["a768_cos_mean"] for r in rows])) if rows else None,
               "a768_cos_min": float(np.min([r["a768_cos_min"] for r in rows])) if rows else None,
               "max_abs_n_diff": int(max(abs(r["n_diff"]) for r in rows)) if rows else None,
               "extract": det_e, "hf": det_h,
               "metric_diff": {k: det_e[k] - det_h[k] for k in det_e}}
    metric_ok = all(abs(d) <= 1.0 for k, d in summary["metric_diff"].items() if k != "events_per_video_api")
    cos_ok = bool(rows) and summary["v768_cos_mean"] >= 0.99 and summary["a768_cos_mean"] >= 0.99
    summary["pass"] = bool(cos_ok and metric_ok)
    table = rows + [{"video_id": f"SUMMARY_{k}", **v} for k, v in (("extract", det_e), ("hf", det_h))]
    write_csv(cfg.analysis_dir / "parity_extract_vs_hf.csv", table)
    print(f"[parity] {len(vids)} videos | v768 cos mean {summary['v768_cos_mean']:.5f} min {summary['v768_cos_min']:.5f} "
          f"| a768 cos mean {summary['a768_cos_mean']:.5f} min {summary['a768_cos_min']:.5f} "
          f"| max |n diff| {summary['max_abs_n_diff']}" if rows else "[parity] no common videos")
    for k in det_e:
        print(f"[parity] {k:22s} extract {det_e[k]:7.2f} | hf {det_h[k]:7.2f} | diff {det_e[k] - det_h[k]:+.2f}")
    if summary["pass"]:
        print("[parity] PASS: the video path reproduces the stored features")
    else:
        print("[parity] *** CẢNH BÁO: đường từ video LỆCH so với feature HF (mean cos < 0.99 hoặc metric lệch > 1 điểm). "
              "Kết quả đánh giá chính vẫn dùng feature HF; xem analysis/parity_extract_vs_hf.csv ***")
    return summary
