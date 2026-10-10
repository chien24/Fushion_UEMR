"""Paths and run settings, one place for the notebook and the CLI.

``Config.colab()`` gives the Colab/Drive layout. Nothing here runs on the local machine.

Layout on Drive (``cache_root`` = ``MyDrive/uemr/uniav_cache``)::

    feats/<source>/<video_id>.npz            v768, v512, a768 (float16), one row per second
    feats/<source>/feats_meta.jsonl          {video_id, n, duration, source, ...}
    feats/<source>/failed.jsonl              videos that could not be prepared
    common/subset_videos[_smoke].json        only when the Omni gallery file is missing
    <tag>/                                   results with FEATURE_SOURCE 'hf'
    <tag>/from_<source>/                     results with another feature source ('extract', 'drive')
    <tag>/smoke/                             SMOKE runs (features of athena/samples)
        raw_proposals.jsonl, predictions.jsonl, predictions_api.jsonl, segments_pred.jsonl,
        theta.json, results.json, per_video.csv, per_gt_event.csv, analysis/*.csv, Z/<video_id>.npz

Feature files are shared by every tag; results are not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

FEATURE_SOURCES = ("hf", "extract", "drive", "samples")
THETA_GRID = tuple(round(0.05 * i, 2) for i in range(1, 13))   # 0.05 .. 0.60
TIOUS = (0.3, 0.5, 0.7)
MAP_TIOUS = (0.3, 0.4, 0.5, 0.6, 0.7)
JUDGE_TAUS = (0.7, 0.8, 0.9)
# the first five shipped samples (sorted); all are YouCook2 val videos with GT in the Omni manifest
SMOKE_IDS = ("-Ju39A-G0Dk", "6uHoTJSLoL8", "9GX8f5EwwE4", "SOMsxGGSTUk", "W2gnFLOi_AQ")

ATHENA_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "athena"

CAVEATS = [
    "Checkpoint athena.pth (run ov_E1) was trained on all 1106 YouCook2 train videos, including the 104 dev "
    "videos of the gallery; its epochs (best_seg / best_cap) and the API min_score 0.40 were chosen on val.",
    "dev: đã thấy khi train (ranh giới + câu). Val is the main result; dev is for reference only.",
    "Semantic metrics on dev: caption GT của dev nằm sẵn trong danh sách ứng viên (the 8218 train captions).",
    "ORACLE columns caption with the video's own GT captions as candidates: an upper bound, not a result.",
]


@dataclass
class Config:
    # --- run ---------------------------------------------------------------
    tag: str = "ft"                       # 'ft' = athena.pth (run ov_E1)
    smoke: bool = False                   # 5 shipped sample videos, no HF token needed
    seed: int = 0
    feature_source: str = "hf"            # 'hf' | 'extract' | 'drive' ('samples' is forced by smoke)
    device: str = "cuda"

    # --- data --------------------------------------------------------------
    drive_data: str = ""                  # .../YouCookII on Drive (metadata/, videos/)
    meta_dir: str = ""
    video_src: str = ""                   # mp4 on Drive
    video_root: str = "/content/videos"   # local copies for the encoder / clips
    omni_subset: str = ""                 # Omni gallery (read only)
    annotations: str = str(ATHENA_ROOT / "data" / "youcookii" / "annotations" / "youcookii_annotations_trainval.json")
    caption_emb: str = str(ATHENA_ROOT / "data" / "youcookii" / "caption_emb_iv2j.npz")
    samples_dir: str = str(ATHENA_ROOT / "athena" / "samples")

    # --- models ------------------------------------------------------------
    checkpoint: str = "/content/ckpt/athena.pth"
    expected_run: str = "ov_E1"
    iv2_video_encoder: str = "/content/ckpt/internvideo2/InternVideo2-stage2_1b-224p-f4.pt"
    iv2_audio_encoder: str = "/content/ckpt/internvideo2/audio_6b.pth"
    iv2_repo: str = "/content/InternVideo"
    athena_index: str = "/content/athena_index"   # AthenaPipeline keeps an index dir; never on Drive

    # --- feature sources ---------------------------------------------------
    hf_repo: str = "nguyenminh04/uniav-youcook2-data"
    hf_subdir: str = "iv2_feats"
    hf_download_dir: str = "/content/hf_iv2"
    hf_keep_tars: bool = False            # delete each tar once its videos are extracted
    drive_feats_src: str = ""             # folder of <id>.npz for FEATURE_SOURCE 'drive'

    # --- output ------------------------------------------------------------
    cache_root: str = ""

    # --- post-processing (UEMR_FINAL.md, section A) --------------------------
    theta: float | None = None            # None: chosen on dev (mean K_pred closest to mean K_GT)
    k_max: int = 30
    d_min: float = 2.0
    merge_tiou: float = 0.7
    # reference config 'api' = athena.pipeline.select_events
    api_min_score: float = 0.40
    api_max_overlap: float = 0.3
    api_max_events: int = 30

    save_z: bool = True
    alternatives: int = 3
    judge_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    theta_grid: tuple = THETA_GRID
    smoke_ids: tuple = SMOKE_IDS
    extra: dict = field(default_factory=dict)

    # ----------------------------------------------------------------------
    @classmethod
    def colab(cls, **overrides) -> "Config":
        drive_data = "/content/drive/MyDrive/Colab Notebooks/code KL/Omni/data/YouCookII"
        cfg = cls(
            drive_data=drive_data,
            meta_dir=f"{drive_data}/metadata",
            video_src=f"{drive_data}/videos",
            omni_subset="/content/drive/MyDrive/uemr/omni_cache/common/subset_videos.json",
            cache_root="/content/drive/MyDrive/uemr/uniav_cache",
            drive_feats_src="/content/drive/MyDrive/uniav_omni/iv2_feats",
        )
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise AttributeError(f"Config has no field {k!r}")
            setattr(cfg, k, v)
        if cfg.feature_source not in FEATURE_SOURCES:
            raise ValueError(f"feature_source must be one of {FEATURE_SOURCES}, got {cfg.feature_source!r}")
        return cfg

    # --- derived -----------------------------------------------------------
    @property
    def source(self) -> str:
        """The feature source actually used: SMOKE always reads the shipped samples."""
        return "samples" if self.smoke else self.feature_source

    @property
    def feats_dir(self) -> Path:
        return Path(self.cache_root) / "feats" / self.source

    def feats_dir_for(self, source: str) -> Path:
        return Path(self.cache_root) / "feats" / source

    @property
    def common_dir(self) -> Path:
        return Path(self.cache_root) / "common"

    @property
    def tag_dir(self) -> Path:
        return Path(self.cache_root) / self.tag

    @property
    def out_dir(self) -> Path:
        if self.smoke:
            return self.tag_dir / "smoke"
        return self.tag_dir if self.source == "hf" else self.tag_dir / f"from_{self.source}"

    @property
    def analysis_dir(self) -> Path:
        return self.out_dir / "analysis"

    @property
    def z_dir(self) -> Path:
        return self.out_dir / "Z"

    @property
    def raw_path(self) -> Path:
        return self.out_dir / "raw_proposals.jsonl"

    @property
    def predictions_path(self) -> Path:
        return self.out_dir / "predictions.jsonl"

    @property
    def predictions_api_path(self) -> Path:
        return self.out_dir / "predictions_api.jsonl"

    @property
    def segments_path(self) -> Path:
        return self.out_dir / "segments_pred.jsonl"

    @property
    def theta_path(self) -> Path:
        return self.out_dir / "theta.json"

    def postprocess_params(self, theta: float | None = None) -> dict:
        t = self.theta if theta is None else theta
        if t is None:
            raise ValueError("theta is not set: run the theta sweep (choose_theta) first")
        return {"theta": float(t), "k_max": self.k_max, "d_min": self.d_min, "merge_tiou": self.merge_tiou}

    def api_params(self) -> dict:
        return {"min_score": self.api_min_score, "max_overlap": self.api_max_overlap,
                "max_events": self.api_max_events}
