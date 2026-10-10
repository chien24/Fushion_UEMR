"""Post-process raw proposals and caption the final segments.

Writes (overwritten on every run; it takes about a second per 10 videos):
  predictions.jsonl      config 'uemr', one line per video:
                         {video_id, split, duration, n, feature_source, theta, events: [{t_s, t_e, conf, caption,
                          similarity, consensus, alternatives, caption_oracle, similarity_oracle, merged, extended}]}
  predictions_api.jsonl  config 'api' (Athena's select_events), same format
  segments_pred.jsonl    config 'uemr', one line per segment: {video_id, ts, te, conf, caption, split,
                         feature_source}; the format Omni can encode later (docs/UNIAV_SEGMENTS_FORMAT.md)
"""

from __future__ import annotations

import time

from .caption import OracleVocab, caption_segments, forward_caption_model
from .config import Config
from .features import FeatureRecord, load_arrays
from .io_utils import write_jsonl
from .postprocess import postprocess_api, postprocess_uemr

KEEP = ("conf", "merged", "extended")


def _events(segs: list[dict], caps: list[dict], oracle: list[dict] | None) -> list[dict]:
    out = []
    for i, (s, c) in enumerate(zip(segs, caps)):
        e = {"t_s": round(float(s["ts"]), 3), "t_e": round(float(s["te"]), 3),
             **{k: s[k] for k in KEEP if k in s},
             "caption": c["caption"], "similarity": c["similarity"], "consensus": c["consensus"],
             "alternatives": [a["caption"] for a in c["alternatives"]]}
        if oracle:
            e["caption_oracle"] = oracle[i]["caption"]
            e["similarity_oracle"] = oracle[i]["similarity"]
        out.append(e)
    return out


def run_predictions(cfg: Config, pipe, records: dict[str, FeatureRecord], raw: dict[str, dict], theta: float,
                    oracle: OracleVocab | None = None) -> tuple[dict[str, dict], dict[str, dict]]:
    """``({vid: uemr prediction}, {vid: api prediction})`` for every video with raw proposals and features."""
    params = cfg.postprocess_params(theta)
    api = cfg.api_params()
    pred, pred_api, segments = {}, {}, []
    vids = [v for v in raw if v in records]
    t0 = time.time()
    for i, vid in enumerate(vids, 1):
        r = raw[vid]
        props = r["proposals"]
        segs_u = postprocess_uemr(props, r["duration"], **params)
        segs_a = postprocess_api(props, **api)
        visual, audio = load_arrays(records[vid], pipe.spec)
        fv, fa, n = pipe.spec.prepare(visual, audio)
        forward_caption_model(pipe, fv, fa)
        caps = caption_segments(pipe, n, segs_u + segs_a, alternatives=cfg.alternatives)
        ocap = oracle.captioner(pipe, vid) if oracle else None
        ocaps = caption_segments(pipe, n, segs_u + segs_a, captioner=ocap, alternatives=0) if ocap else None
        nu = len(segs_u)
        head = {"video_id": vid, "split": r["split"], "duration": r["duration"], "n": n,
                "feature_source": r["feature_source"]}
        pred[vid] = dict(head, config="uemr", theta=params["theta"], params=params,
                         events=_events(segs_u, caps[:nu], ocaps[:nu] if ocaps else None))
        pred_api[vid] = dict(head, config="api", params=api,
                             events=_events(segs_a, caps[nu:], ocaps[nu:] if ocaps else None))
        segments += [{"video_id": vid, "ts": e["t_s"], "te": e["t_e"], "conf": e["conf"], "caption": e["caption"],
                      "split": r["split"], "feature_source": r["feature_source"]} for e in pred[vid]["events"]]
        if i == 1 or i % 100 == 0 or i == len(vids):
            print(f"[predict] {i}/{len(vids)} | {(time.time() - t0) / i:.2f} s/video", flush=True)
    write_jsonl(cfg.predictions_path, pred.values())
    write_jsonl(cfg.predictions_api_path, pred_api.values())
    write_jsonl(cfg.segments_path, segments)
    print(f"[predict] theta={params['theta']} -> {cfg.predictions_path.name}, {cfg.predictions_api_path.name}, "
          f"{cfg.segments_path.name} ({len(segments)} segments)")
    return pred, pred_api
