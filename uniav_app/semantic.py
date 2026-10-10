"""Independent caption judge: ``sentence-transformers/all-MiniLM-L6-v2`` (the judge Omni's analysis uses).

A GT step and the segment matched to it count as "right place AND right meaning" when
cos(judge(GT caption), judge(predicted caption)) >= tau, tau in {0.7, 0.8, 0.9}.
"""

from __future__ import annotations

import re
from typing import Callable

import numpy as np

DEFAULT_JUDGE = "sentence-transformers/all-MiniLM-L6-v2"


def normalize_caption(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(s).lower()).split())


def caption_embedder(model_name: str = DEFAULT_JUDGE, device: str | None = None,
                     batch_size: int = 256) -> Callable[[list[str]], np.ndarray]:
    """``f(texts) -> (n, d)`` L2-normalised vectors from a sentence-transformers model."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)

    def embed(texts: list[str]) -> np.ndarray:
        return model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    embed.name = model_name
    return embed


class Judge:
    """Caches one vector per normalised caption; ``cos(a, b)`` for lists of caption pairs."""

    def __init__(self, embed: Callable[[list[str]], np.ndarray]):
        self.embed = embed
        self.name = getattr(embed, "name", "judge")
        self._vec: dict[str, np.ndarray] = {}

    def add(self, texts) -> None:
        new = sorted({normalize_caption(t) for t in texts if t is not None} - set(self._vec))
        if new:
            vecs = np.asarray(self.embed(new), dtype=np.float32)
            vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
            self._vec.update(zip(new, vecs))

    def cos(self, a: list[str], b: list[str]) -> np.ndarray:
        if not len(a):
            return np.zeros(0, dtype=np.float32)
        self.add(list(a) + list(b))
        va = np.stack([self._vec[normalize_caption(t)] for t in a])
        vb = np.stack([self._vec[normalize_caption(t)] for t in b])
        return np.sum(va * vb, axis=1)
