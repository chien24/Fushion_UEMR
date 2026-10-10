"""Control baselines with the same number of segments (RQ2 / RQ4): split the video into K equal parts.

Every uniform segment gets conf 1.0, so ranking metrics (AR@N) take them in time order.
"""

from __future__ import annotations


def uniform_segments(duration: float, k: int) -> list[dict]:
    k = max(int(k), 1)
    d = float(duration) / k
    return [{"ts": i * d, "te": float(duration) if i == k - 1 else (i + 1) * d, "conf": 1.0} for i in range(k)]
