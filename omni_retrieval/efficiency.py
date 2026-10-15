"""Exp 5, efficiency part (``docs/UEMR_VIEC_TIEP_THEO.md`` §A2): index size, query latency, build cost, int8.

* size     -- vectors x d x bytes (fp32 4, fp16 2; int8 1 + a 4-byte fp32 scale per vector)
* latency  -- MaxSim of one query against the whole gallery (``SegmentIndex.score_matrix``), mean over the
              queries: one query at a time (online search) and in blocks of 256 (offline evaluation), numpy on
              CPU; the same on GPU with torch when one is there
* build    -- Omni clips per video x seconds per clip, plus the UniAV run per video for partitions that need it
* int8     -- symmetric per-vector quantisation (scale = max |x| / 127), scored after dequantisation
"""

from __future__ import annotations

import time

import numpy as np

from .search import SegmentIndex

OMNI_SEC_PER_CLIP = 0.26        # A100, measured over the P3 encode (UEMR_VIEC_TIEP_THEO.md §A1)


def index_bytes(n_vectors: int, d: int) -> dict[str, float]:
    """Index size in MB for fp32, fp16 and int8 (+ fp32 scale)."""
    mb = 1024 ** 2
    return {"fp32 (MB)": n_vectors * d * 4 / mb, "fp16 (MB)": n_vectors * d * 2 / mb,
            "int8 (MB)": n_vectors * (d + 4) / mb}


def quantize_int8(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(codes int8 [N, d], scale fp32 [N])`` with ``emb ~ codes * scale``."""
    emb = np.asarray(emb, dtype=np.float32)
    scale = np.abs(emb).max(axis=1) / 127.0
    scale = np.where(scale > 0, scale, 1.0).astype(np.float32)
    return np.clip(np.rint(emb / scale[:, None]), -127, 127).astype(np.int8), scale


def int8_index(index: SegmentIndex, name: str | None = None) -> SegmentIndex:
    """The same index with every vector quantised to int8 and back (what an int8 index would score)."""
    codes, scale = quantize_int8(index.emb)
    return SegmentIndex(codes.astype(np.float32) * scale[:, None], index.seg_video, index.ts, index.te,
                        name or f"{index.name} int8")


def _timed(fn, repeat: int = 1) -> float:
    fn()                                                  # warm-up (allocation, BLAS threads, CUDA kernels)
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t0) / repeat


def latency_cpu(index: SegmentIndex, q: np.ndarray, single: int = 1000) -> dict[str, float]:
    """ms per query: ``single`` queries one by one, and all of ``q`` in blocks of 256."""
    q = np.asarray(q, dtype=np.float32)
    one = q[:single]
    t_single = _timed(lambda: [index.score_matrix(x[None]) for x in one]) / len(one)
    t_block = _timed(lambda: index.score_matrix(q, chunk=256)) / len(q)
    return {"CPU 1 query (ms)": 1e3 * t_single, "CPU block 256 (ms/query)": 1e3 * t_block}


def latency_gpu(index: SegmentIndex, q: np.ndarray, single: int = 1000) -> dict[str, float]:
    """Same as ``latency_cpu`` with torch fp16 on CUDA (``{}`` without a GPU)."""
    try:
        import torch
    except ImportError:
        return {}
    if not torch.cuda.is_available():
        return {}
    dev = torch.device("cuda")
    emb = torch.as_tensor(index.emb, device=dev, dtype=torch.float16)
    valid = torch.as_tensor(index.pad >= 0, device=dev)
    pad = torch.as_tensor(np.where(index.pad >= 0, index.pad, 0), device=dev)
    qt = torch.as_tensor(np.asarray(q, dtype=np.float32), device=dev, dtype=torch.float16)

    def maxsim(x):
        sim = x @ emb.T                                                     # [b, N]
        return sim[:, pad].masked_fill(~valid[None], float("-inf")).amax(-1)   # [b, V]

    def run(batches):
        for b in batches:
            maxsim(b)
        torch.cuda.synchronize()

    t_single = _timed(lambda: run([qt[i:i + 1] for i in range(min(single, len(qt)))])) / min(single, len(qt))
    t_block = _timed(lambda: run([qt[i:i + 256] for i in range(0, len(qt), 256)])) / len(qt)
    return {"GPU 1 query (ms)": 1e3 * t_single, "GPU block 256 (ms/query)": 1e3 * t_block,
            "GPU": torch.cuda.get_device_name(0)}


def efficiency_row(index: SegmentIndex, q: np.ndarray, needs_uniav: bool, uniav_sec_per_video: float | None = None,
                   sec_per_clip: float = OMNI_SEC_PER_CLIP, gpu: bool = True) -> dict:
    """Size, latency and build cost of one partition's index (the whole gallery it holds)."""
    n, d = index.emb.shape
    clips = float(np.mean(index.counts))
    row = {"partition": index.name, "videos": index.n_videos, "vectors": int(n), "vec/video": clips, "d": int(d),
           **index_bytes(n, d), **latency_cpu(index, q), **(latency_gpu(index, q) if gpu else {}),
           "Omni clips/video": clips, "Omni s/video": clips * sec_per_clip, "needs UniAV": needs_uniav}
    if needs_uniav:
        row["UniAV s/video"] = uniav_sec_per_video
    return row
