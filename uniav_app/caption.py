"""Captions for final segments, through Athena's own captioning path.

``AthenaPipeline.process_features`` captions segments found by ``seg_model`` with the caption weights:
``pipe.model(fv, fa)`` then ``pipe.model.span_vectors(segs)`` and ``pipe.captioner.caption(vecs)``
(MBR pick among the 8218 train captions, InternVideo2 text space). Our segments are in seconds after
post-processing (merged / extended), so they go back to grid units first: ``g = t * 256 / n - 0.5``.

ORACLE vocabulary: the same ``athena.captioner.Captioner`` class with a pool made of the video's own
GT captions (InternVideo2 vectors from ``caption_emb_iv2j.npz``, the file the checkpoint's caption pool
was built from), ``k = min(20, N)``. An upper bound, never a result.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from .timegrid import sec_to_grid, spec_params


@torch.no_grad()
def forward_caption_model(pipe, fv, fa) -> None:
    """Run the caption weights on this video so ``span_vectors`` reads its features."""
    pipe.model(fv.to(pipe.device), fa.to(pipe.device))


@torch.no_grad()
def caption_segments(pipe, n: int, segs: list[dict], captioner=None, alternatives: int = 3, k: int = 20) -> list[dict]:
    """``[{caption, similarity, consensus, alternatives}]`` for segments in seconds. Call
    ``forward_caption_model`` on the same video first."""
    if not segs:
        return []
    g = sec_to_grid([[s["ts"], s["te"]] for s in segs], n, **spec_params(pipe.spec))
    vecs = pipe.model.span_vectors(g)
    cap = captioner or pipe.captioner
    return cap.caption(vecs, k=min(k, len(cap.texts)), alternatives=alternatives)


class OracleVocab:
    """Per-video caption pools of GT captions (keys ``<video_id>#<i>`` of caption_emb_iv2j.npz)."""

    def __init__(self, caption_emb_path: str, tmp_dir: str = "/content/tmp_oracle_pools"):
        z = np.load(caption_emb_path, allow_pickle=True)   # vendored repo file (keys / sentences are object arrays)
        self.mean = np.asarray(z["mean"], np.float32)
        self.by_video: dict[str, list[tuple[int, str, np.ndarray]]] = {}
        for key, text, emb in zip(z["keys"], z["sentences"], z["emb"]):
            if "#" not in str(key):   # e.g. a non-caption entry
                continue
            vid, i = str(key).rsplit("#", 1)
            self.by_video.setdefault(vid, []).append((int(i), str(text), np.asarray(emb, np.float32)))
        for v in self.by_video.values():
            v.sort(key=lambda x: x[0])
        self.tmp_dir = Path(tmp_dir)
        self._cache: dict[str, object] = {}

    def captioner(self, pipe, video_id: str):
        """An ``athena.captioner.Captioner`` whose candidates are this video's GT captions, or None."""
        if video_id in self._cache:
            return self._cache[video_id]
        rows = self.by_video.get(video_id)
        if not rows:
            return None
        from .athena_api import import_athena
        import_athena()
        from athena.captioner import Captioner
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        path = self.tmp_dir / f"{video_id}.npz"
        tmp = path.with_name(path.name + ".part")
        with open(tmp, "wb") as f:
            np.savez(f, texts=np.array([r[1] for r in rows], dtype=str), emb=np.stack([r[2] for r in rows]),
                     mean=self.mean)
        os.replace(tmp, path)
        cap = Captioner(str(path), pipe.model, pipe.device)
        self._cache = {video_id: cap}   # one video at a time
        return cap
