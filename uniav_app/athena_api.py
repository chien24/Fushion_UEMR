"""The only place that imports the vendored Athena code (third_party/athena, unmodified).

Every call goes to a function or class that exists in Athena commit 48918c5:
``athena.config.Config``, ``athena.pipeline.AthenaPipeline``, ``athena.pipeline.select_events``,
``athena.features.FeatureSpec``, ``athena.calibrate.match``,
``athena.encoders.internvideo2.InternVideo2AVEncoder``, ``athena.encoders.media.probe``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .config import ATHENA_ROOT, Config


def ensure_athena_path() -> Path:
    root = str(ATHENA_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return ATHENA_ROOT


def import_athena():
    """``import athena`` from third_party/athena, never from somewhere else on sys.path."""
    ensure_athena_path()
    import athena
    got = Path(athena.__file__).resolve().parent
    want = (ATHENA_ROOT / "athena").resolve()
    if got != want:
        raise ImportError(f"athena imported from {got}, expected the vendored copy {want}")
    return athena


def resolve_device(pref: str) -> str:
    import torch
    if pref == "cuda" and not torch.cuda.is_available():
        print("[athena] CUDA not available -> running the event model on CPU")
        return "cpu"
    return pref


def athena_config(cfg: Config, use_seg_weights: bool = True):
    """An ``athena.config.Config`` with every path pointing at the Colab layout (no patching needed)."""
    athena = import_athena()
    return athena.Config(
        checkpoint=cfg.checkpoint,
        caption_pool=str(ATHENA_ROOT / "athena" / "assets" / "caption_pool.npz"),
        samples_dir=cfg.samples_dir,
        iv2_video_encoder=cfg.iv2_video_encoder,
        iv2_audio_encoder=cfg.iv2_audio_encoder,
        iv2_repo=cfg.iv2_repo,
        index_dir=cfg.athena_index,
        device=resolve_device(cfg.device),
        min_score=cfg.api_min_score,
        max_overlap=cfg.api_max_overlap,
        max_events=cfg.api_max_events,
        alternatives=cfg.alternatives,
        caption_mode="retrieve",
        use_seg_weights=use_seg_weights,
    )


def load_pipeline(cfg: Config, use_seg_weights: bool = True):
    """``AthenaPipeline`` for the checkpoint. Its state dicts load with ``strict=True``, so a missing or
    unexpected key raises here. ``seg_model`` segments (best_seg), ``model`` captions (best_cap)."""
    athena = import_athena()
    pipe = athena.AthenaPipeline(athena_config(cfg, use_seg_weights))
    if pipe.run != cfg.expected_run:
        raise ValueError(f"checkpoint run is {pipe.run!r}, expected {cfg.expected_run!r}")
    return pipe


def checkpoint_info(path: str) -> dict:
    """The fields of an API checkpoint (tools/export_api_ckpt.py) that the report and the sanity check need."""
    import torch
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:   # torch < 1.13 has no weights_only
        ck = torch.load(path, map_location="cpu")
    pool = ck.get("caption_pool")
    sd = ck["state_dict"]
    info = {
        "run": ck.get("run"),
        "commit": ck.get("commit"),
        "epoch": ck.get("epoch"),
        "epoch_seg": ck.get("epoch_seg"),
        "has_state_dict_seg": "state_dict_seg" in ck,
        "n_tensors": len(sd),
        "n_params_M": round(sum(v.numel() for v in sd.values()) / 1e6, 2),
        "dataset": ck.get("config", {}).get("dataset"),
        "test_cfg": ck.get("config", {}).get("model", {}).get("test_cfg"),
        "caption_pool_file": pool.get("file") if isinstance(pool, dict) else pool,
        "final_eval": ck.get("final_eval"),
    }
    del ck, sd
    return info


def select_events(secs, scores, min_score: float, max_overlap: float, max_events: int) -> list[int]:
    """``athena.pipeline.select_events`` (the API's event choice), unchanged."""
    import_athena()
    from athena.pipeline import select_events as _select
    return _select(secs, scores, min_score, max_overlap, max_events)


def greedy_match_count(preds, gts, thr: float = 0.5) -> int:
    """``athena.calibrate.match``: true positives of its greedy one-to-one matching (for the 0.535 reference)."""
    import_athena()
    from athena.calibrate import match
    return match(preds, gts, thr)


def make_encoder(cfg: Config):
    """``InternVideo2AVEncoder`` as ``AthenaPipeline.encoder`` builds it. The models load on the first
    ``encode_raw`` call, not here."""
    import torch
    import_athena()
    from athena.encoders.internvideo2 import InternVideo2AVEncoder
    for p in (cfg.iv2_video_encoder, cfg.iv2_audio_encoder, cfg.iv2_repo):
        if not Path(p).exists():
            raise FileNotFoundError(f"{p} not found: FEATURE_SOURCE 'extract' needs the InternVideo2 encoders "
                                    "and the InternVideo repo (notebook cell 'NEED_EXTRACT')")
    device = torch.device(resolve_device(cfg.device))
    return InternVideo2AVEncoder(cfg.iv2_video_encoder, cfg.iv2_audio_encoder, cfg.iv2_repo, device,
                                 video_keys=("v768",), l2norm=True, keep_loaded=True)


def probe(path: str) -> dict:
    """``athena.encoders.media.probe``: duration, width, height, sample_rate from ``ffmpeg -i``."""
    import_athena()
    from athena.encoders.media import probe as _probe
    return _probe(path)
