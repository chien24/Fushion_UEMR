"""Fake encoder output for testing ``collect`` + ``evaluate`` without the model.

Writes two single-file stores in the ``omniretriever.cli extract`` layout
(``<seg_key>__av`` and ``<caption_id>__text`` keys) for the real segment/caption tables
of a dry-run, then collects and evaluates them. Each caption's text vector is its own
GT clip's vector plus noise, so text->clip on ``gt`` must come out near 100 % and the
``gt`` partition must beat ``global``; anything else means the bookkeeping is wrong.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .collect import collect_events, collect_segments, collect_text
from .evaluate import evaluate, evaluate_events, markdown, text_to_clip


def _vec(key: str, dim: int) -> np.ndarray:
    seed = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).normal(size=dim).astype(np.float32)


def write_fake_stores(prepared: dict, out_dir, dim: int = 64, noise: float = 0.3) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    av = {f"{k}__av": _vec(k, dim) for k in prepared["segments"]}
    text = {f"{c['caption_id']}__text": _vec(c["gt_seg_key"], dim) + noise * _vec(c["caption_id"], dim)
            for c in prepared["captions"]}
    np.savez(out / "fake_av.npz", **av)
    np.savez(out / "fake_text.npz", **text)
    return out / "fake_av.npz", out / "fake_text.npz"


def run_fake_pipeline(prepared: dict, out_dir, dim: int = 64) -> list[dict]:
    av_store, text_store = write_fake_stores(prepared, out_dir, dim)
    cache = Path(out_dir) / "cache"
    print("[fake] events  :", collect_events(prepared["event_rows"], av_store, cache / "events.npz"))
    print("[fake] segments:", collect_segments(prepared["rows"], av_store, cache / "segments.npz"))
    print("[fake] text    :", collect_text(prepared["captions"], text_store, cache / "text.npz"))
    summary, _ = evaluate_events(cache)
    print("[fake] event retrieval:", {k: round(v, 2) if isinstance(v, float) else v for k, v in summary.items()})
    assert summary["event R@1"] > 95, "fake event R@1 should be ~100: caption <-> event mapping is broken"
    clip = text_to_clip(cache)
    print(f"[fake] text->clip R@1: eval gallery {clip['eval']['R@1']:.1f} | subset gallery {clip['subset']['R@1']:.1f}")
    rows = evaluate(cache, [p for p in ("global", "event_single", "uni_M_gt", "gt")
                            if p == "event_single" or p in prepared["parts"]])
    print(markdown(rows))
    by = {r["partition"]: r for r in rows}
    assert clip["eval"]["R@1"] > 95, "fake text->clip should be ~100: caption <-> gt clip mapping is broken"
    if "global" in by:
        assert by["gt"]["R@1"] > by["global"]["R@1"], "fake gt should beat global"
    print("[fake] OK")
    return rows
