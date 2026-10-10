"""Any mp4 (YouCook2 or not) -> events with captions, always through the video path ('extract').

mp4 -> InternVideo2 + BEATs (``features.extract.ExtractSource.encode_file``, cached in feats/extract/)
-> Athena raw proposals -> post-processing ('uemr', theta from theta.json, else the API's 0.40)
-> captions. The same functions as the evaluation pipeline, nothing specific to this demo.
"""

from __future__ import annotations

from pathlib import Path

from .caption import caption_segments, forward_caption_model
from .config import Config
from .detect import detect_features
from .features import load_arrays
from .features.extract import ExtractSource
from .io_utils import read_json
from .postprocess import postprocess_api, postprocess_uemr


def current_theta(cfg: Config) -> float:
    if cfg.theta is not None:
        return float(cfg.theta)
    saved = read_json(cfg.theta_path)
    if saved and saved.get("theta") is not None:
        return float(saved["theta"])
    print(f"[demo] no theta chosen yet ({cfg.theta_path}) -> using the API min_score {cfg.api_min_score}")
    return cfg.api_min_score


def describe_mp4(cfg: Config, pipe, path: str, video_id: str | None = None, extractor: ExtractSource | None = None) -> dict:
    """``{video_id, duration, n, theta, events, events_api}`` for one video file."""
    ex = extractor or ExtractSource(cfg)
    video_id = video_id or Path(path).stem
    rec = ex.encode_file(path, video_id)
    visual, audio = load_arrays(rec, pipe.spec)
    det = detect_features(pipe, visual, audio, rec.duration)
    theta = current_theta(cfg)
    segs = postprocess_uemr(det["proposals"], det["duration"], **cfg.postprocess_params(theta))
    segs_api = postprocess_api(det["proposals"], **cfg.api_params())
    fv, fa, n = pipe.spec.prepare(visual, audio)
    forward_caption_model(pipe, fv, fa)
    caps = caption_segments(pipe, n, segs + segs_api, alternatives=cfg.alternatives)

    def events(ss, cc):
        return [{"t_s": round(s["ts"], 2), "t_e": round(s["te"], 2), "conf": round(float(s["conf"]), 4),
                 "caption": c["caption"], "similarity": c["similarity"]} for s, c in zip(ss, cc)]

    return {"video_id": video_id, "duration": det["duration"], "n": det["n"], "theta": theta,
            "events": events(segs, caps[:len(segs)]), "events_api": events(segs_api, caps[len(segs):])}
