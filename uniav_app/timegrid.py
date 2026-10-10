"""Athena's time grid: seconds <-> grid units of the resampled sequence.

The features (``n`` rows, row ``i`` centred on ``i + 0.5`` s for 1 row/s) are linearly resampled to
``max_seq_len = 256`` steps whatever the video length (``athena/features.py::FeatureSpec.prepare``).
``FeatureSpec.to_seconds`` (= ``libs/modeling/event_archs.py::inference``) maps a grid coordinate ``g`` to

    step  = ((n - 1) * stride + window) / max_seq_len        # in "frames" at fps
    t_sec = clip((g * step + 0.5 * step) / fps, 0, duration)

With the checkpoint's ``default_fps 16``, ``iv2_rows_per_sec 1`` (stride = window = 16) this is
``t = (g + 0.5) * n / 256``. Pyramid level ``l`` has ``256 / 2**l`` steps whose anchor point is grid
``j * 2**l`` (point_generator.py, use_offset False), so ``t(l, j) = (j * 2**l + 0.5) * n / 256``.
``n`` is the number of feature rows, not the duration.
"""

from __future__ import annotations

import numpy as np

MAX_SEQ_LEN = 256
FPS = 16
STRIDE = 16
WINDOW = 16
FORMULA = "t_sec = (j * 2**level + 0.5) * n / 256, n = feature rows (1 row/s); clipped to [0, duration]"


def step_frames(n: int, stride: int = STRIDE, window: int = WINDOW, max_seq_len: int = MAX_SEQ_LEN) -> float:
    return float((n - 1) * stride + window) / max_seq_len


def grid_to_sec(g, n: int, duration: float | None = None, fps: int = FPS, stride: int = STRIDE,
                window: int = WINDOW, max_seq_len: int = MAX_SEQ_LEN) -> np.ndarray:
    """Grid coordinates (any shape) -> seconds, exactly as ``FeatureSpec.to_seconds``."""
    step = step_frames(n, stride, window, max_seq_len)
    t = (np.asarray(g, dtype=np.float64) * step + 0.5 * step) / fps
    if duration is not None:
        t = np.clip(t, 0.0, float(duration))
    return t


def sec_to_grid(t, n: int, fps: int = FPS, stride: int = STRIDE, window: int = WINDOW,
                max_seq_len: int = MAX_SEQ_LEN) -> np.ndarray:
    """Inverse of ``grid_to_sec`` (without the clipping): ``g = t * 256 / n - 0.5`` for 1 row/s."""
    step = step_frames(n, stride, window, max_seq_len)
    return np.asarray(t, dtype=np.float64) * fps / step - 0.5


def level_times(n: int, level: int = 0, max_seq_len: int = MAX_SEQ_LEN, **kw) -> np.ndarray:
    """Centre time (s) of every step of pyramid level ``level``: ``(j * 2**level + 0.5) * n / 256``."""
    s = 2 ** level
    return grid_to_sec(np.arange(max_seq_len // s) * s, n, max_seq_len=max_seq_len, **kw)


def spec_params(spec) -> dict:
    """Time parameters of an ``athena.features.FeatureSpec`` (from the checkpoint's config)."""
    return {"fps": int(spec.fps), "stride": int(spec.stride), "window": int(spec.window),
            "max_seq_len": int(spec.max_seq_len)}
