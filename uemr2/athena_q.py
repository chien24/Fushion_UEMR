"""q (512, Athena caption space) of any segments, from the frozen Athena checkpoint.

Same path as ``AthenaPipeline`` captions an event (third_party/athena, commit 48918c5): ``model(fv, fa)`` with the
``state_dict`` (best_cap) weights, then ``model.span_vectors(g)`` = ``seg_ctx`` (span mean over raw0 + MLP with the
whole-video mean and relative position, L2). Seconds -> grid with ``uniav_app.timegrid.sec_to_grid``
(``g = t * 256 / n - 0.5``, ``n`` = feature rows), never ``g = t``.
"""

from __future__ import annotations

import numpy as np
import torch

from uniav_app.timegrid import sec_to_grid, spec_params


def span_q(pipe, visual, audio, segs_sec) -> tuple[np.ndarray, int]:
    """``(q [K, 512] float32, n)`` for segments ``[K, 2]`` in seconds of one video."""
    fv, fa, n = pipe.spec.prepare(visual, audio)
    segs = np.asarray(segs_sec, np.float64).reshape(-1, 2)
    with torch.no_grad():
        pipe.model(fv.to(pipe.device), fa.to(pipe.device))
        q = pipe.model.span_vectors(sec_to_grid(segs, n, **spec_params(pipe.spec)))
    return q.float().cpu().numpy(), int(n)
