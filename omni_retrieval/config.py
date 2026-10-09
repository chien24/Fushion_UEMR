"""Paths and run settings, one place for the notebook and the CLIs.

``Config.colab()`` gives the Colab/Drive layout. Everything runs on Colab, including the
tests and the dry-run; nothing here is meant to run on the local Windows machine.

Layout of the cache (one tree per Drive, shared by every adapter)::

    <CACHE_ROOT>/common/subset_videos[_smoke].json   videos in the gallery
    <CACHE_ROOT>/common/partitions/<name>.json        video_id -> [[ts, te], ...]
    <CACHE_ROOT>/common/segs_todo.jsonl, caps_todo.jsonl
    <CACHE_ROOT>/<tag>/raw/av/chunk_*.npz             encoder output, keys <seg_key>__av
    <CACHE_ROOT>/<tag>/raw/text/chunk_*.npz           encoder output, keys <caption_id>__text
    <CACHE_ROOT>/<tag>/segments.npz, text.npz, meta.json, results_<tag>.csv

``raw/`` is the real cache (deduplicated, append-only); ``segments.npz`` / ``text.npz``
are rebuilt from it by ``collect`` whenever a partition is added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Partitions every run encodes. Extra ones (uniav_pred, uni_M, rand_M, kmedoids_M, ...)
# come from JSON files, see Config.extra_partitions.
BASE_PARTITIONS = ("global", "gt", "uni_M_gt")


@dataclass
class Config:
    # --- run ---------------------------------------------------------------
    tag: str = "ft"                 # "ft" (fine-tuned adapter) or "pt" (pretrained)
    n_videos: int = 500             # capped at val + dev videos available (498)
    seed: int = 0
    smoke: bool = False             # 5 videos end to end
    smoke_videos: int = 5

    # --- data --------------------------------------------------------------
    drive_data: str = ""            # .../YouCookII on Drive (metadata/, videos/)
    meta_dir: str = ""
    video_src: str = ""             # where the mp4s live permanently (Drive)
    video_root: str = ""            # where the encoder reads them (local copy on Colab)

    # --- models ------------------------------------------------------------
    omni_repo: str = ""
    wave_path: str = ""
    beats_path: str = ""
    adapter_ft: str = ""
    adapter_pt: str = ""

    # --- output ------------------------------------------------------------
    cache_root: str = ""
    # dev/val eval JSON the fine-tune notebook wrote; the sanity check compares to it.
    ref_eval_json: str = ""

    # name -> JSON path (video_id -> [[ts, te], ...]); merged into the partition set.
    extra_partitions: dict = field(default_factory=dict)

    # ----------------------------------------------------------------------
    @classmethod
    def colab(cls, **overrides) -> "Config":
        drive_data = "/content/drive/MyDrive/Colab Notebooks/code KL/Omni/data/YouCookII"
        run = "/content/drive/MyDrive/omniretriever/youcookii_ft_uemr"
        cfg = cls(
            drive_data=drive_data,
            meta_dir=f"{drive_data}/metadata",
            video_src=f"{drive_data}/videos",
            video_root="/content/videos",
            omni_repo="/content/Omni-fix",
            wave_path="/content/WAVE_HOME/WAVE-7B",
            beats_path="/content/WAVE_HOME/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
            adapter_ft="/content/my_checkpoint",          # copy of {run}/best
            adapter_pt="/content/adapters/omniretriever-7b",
            cache_root="/content/drive/MyDrive/uemr/omni_cache",
            ref_eval_json=f"{run}/eval/best_val.json",
        )
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise AttributeError(f"Config has no field {k!r}")
            setattr(cfg, k, v)
        return cfg

    # --- derived paths -----------------------------------------------------
    @property
    def adapter(self) -> str:
        return {"ft": self.adapter_ft, "pt": self.adapter_pt}.get(self.tag, self.adapter_ft)

    @property
    def common_dir(self) -> Path:
        return Path(self.cache_root) / "common"

    @property
    def partitions_dir(self) -> Path:
        return self.common_dir / "partitions"

    @property
    def subset_path(self) -> Path:
        return self.common_dir / ("subset_videos_smoke.json" if self.smoke else "subset_videos.json")

    @property
    def segs_todo(self) -> Path:
        return self.common_dir / f"segs_todo_{self.tag}.jsonl"

    @property
    def caps_todo(self) -> Path:
        return self.common_dir / f"caps_todo_{self.tag}.jsonl"

    @property
    def tag_dir(self) -> Path:
        # Smoke runs share raw/ with the full run (same encoder, same segments, so the
        # embeddings are reusable) but write their collected files elsewhere.
        return Path(self.cache_root) / self.tag

    @property
    def out_dir(self) -> Path:
        return self.tag_dir / "smoke" if self.smoke else self.tag_dir

    @property
    def av_store(self) -> Path:
        return self.tag_dir / "raw" / "av"

    @property
    def text_store(self) -> Path:
        return self.tag_dir / "raw" / "text"

    def partition_names(self) -> list[str]:
        return list(BASE_PARTITIONS) + [n for n in self.extra_partitions if n not in BASE_PARTITIONS]
