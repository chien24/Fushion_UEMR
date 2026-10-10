"""Command line for the same steps as the notebook (Colab only).

    python -m uniav_app.cli features [--smoke] [--source hf|extract|drive] [--videos ID ...]
    python -m uniav_app.cli detect   [--smoke] [--source ...]
    python -m uniav_app.cli predict  [--smoke] [--theta 0.3]      # theta: else the dev sweep
    python -m uniav_app.cli evaluate [--smoke] [--no-judge]
    python -m uniav_app.cli sanity   [--no-strict]
    python -m uniav_app.cli z --videos ID ...                     # Z only, e.g. train videos
    python -m uniav_app.cli demo path/to/video.mp4
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import Config


def _setup(a):
    from .features import get_source
    from .subset import load_gallery
    cfg = Config.colab(smoke=a.smoke, feature_source=a.source, **({"checkpoint": a.checkpoint} if a.checkpoint else {}))
    gallery = load_gallery(cfg)
    return cfg, gallery, get_source(cfg)


def _records(cfg, gallery, src, ids=None):
    durs = {r["video_id"]: r["duration"] for r in gallery}
    return src.prepare(ids or [r["video_id"] for r in gallery], durs)


def _gt(cfg, gallery):
    from .subset import gt_for_gallery, load_gt_annotations, load_gt_manifest
    return gt_for_gallery(gallery, load_gt_manifest(cfg.meta_dir), load_gt_annotations(cfg.annotations, subset=None))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="UniAV/Athena application pipeline")
    p.add_argument("step", choices=("features", "detect", "predict", "evaluate", "sanity", "z", "demo"))
    p.add_argument("path", nargs="?", help="video file for 'demo'")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--source", default="hf", choices=("hf", "extract", "drive"))
    p.add_argument("--checkpoint", default="")
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--theta", type=float, default=None)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--no-strict", action="store_true")
    a = p.parse_args(argv)

    from .athena_api import load_pipeline
    from .io_utils import latest_by, read_json, read_jsonl, write_json
    cfg, gallery, src = _setup(a)
    if a.step == "features":
        _records(cfg, gallery, src, a.videos)
        return 0
    if a.step == "demo":
        from .demo import describe_mp4
        print(json.dumps(describe_mp4(cfg, load_pipeline(cfg), a.path), indent=1, ensure_ascii=False))
        return 0
    recs = _records(cfg, gallery, src, a.videos)
    if a.step == "z":
        from .z_export import export_z
        export_z(load_pipeline(cfg), recs, cfg.z_dir)
        return 0
    if a.step == "sanity":
        from .sanity import official_parity
        official_parity(cfg, recs, strict=not a.no_strict)
        return 0
    splits = {r["video_id"]: r["split"] for r in gallery}
    gt = _gt(cfg, gallery)
    if a.step == "detect":
        from .detect import run_detection
        run_detection(cfg, load_pipeline(cfg), recs, splits)
        return 0
    raw = {v: r for v, r in latest_by(read_jsonl(cfg.raw_path)).items() if v in recs}
    from .evaluate import choose_theta, evaluate_all, print_summary, theta_sweep
    sweep = theta_sweep(raw, gt, splits, cfg)
    if a.theta is not None:
        theta_info = {"theta": a.theta, "chosen_on": "manual"}
    else:
        th, on = choose_theta(sweep, "dev")
        theta_info = {"theta": th, "chosen_on": on, "rule": "mean K_pred closest to mean K_GT"}
    write_json(cfg.theta_path, theta_info)
    if a.step == "predict":
        from .caption import OracleVocab
        from .predict import run_predictions
        run_predictions(cfg, load_pipeline(cfg), recs, raw, theta_info["theta"], OracleVocab(cfg.caption_emb))
        return 0
    pred = latest_by(read_jsonl(cfg.predictions_path))
    pred_api = latest_by(read_jsonl(cfg.predictions_api_path))
    judge = None
    if not a.no_judge:
        from .semantic import Judge, caption_embedder
        judge = Judge(caption_embedder(cfg.judge_model))
    res = evaluate_all(cfg, gt, raw, pred, pred_api, judge, theta_info, sweep,
                       extra={"sanity": read_json(cfg.out_dir / "sanity.json")})
    print_summary(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
