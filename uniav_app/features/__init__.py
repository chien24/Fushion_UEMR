"""Pluggable feature sources with one interface (``FeatureSource.prepare`` -> ``FeatureRecord``)."""

from __future__ import annotations

from ..config import Config
from .base import FeatureRecord, FeatureSource, check_n_vs_duration, load_arrays, save_feature_npz


def get_source(cfg: Config, name: str | None = None) -> FeatureSource:
    """The source named ``name`` (default: ``cfg.source``, i.e. 'samples' in SMOKE, else FEATURE_SOURCE)."""
    name = name or cfg.source
    if name == "hf":
        from .hf import HFSource
        return HFSource(cfg)
    if name == "extract":
        from .extract import ExtractSource
        return ExtractSource(cfg)
    if name == "drive":
        from .drive import DriveSource
        return DriveSource(cfg)
    if name == "samples":
        from .samples import SamplesSource
        return SamplesSource(cfg)
    raise ValueError(f"unknown feature source {name!r}")


__all__ = ["FeatureRecord", "FeatureSource", "get_source", "load_arrays", "save_feature_npz", "check_n_vs_duration"]
