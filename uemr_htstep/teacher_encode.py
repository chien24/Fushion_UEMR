"""One load of UniAV's teacher (Omni + the LoRA of omni-adapters/checkpoint): GT clips must match omni_av/, then every query.

The clips are encoded first, with ``OmniRetriever.encode_av(mp4, timestamps=(start, end))`` as ``scripts/extract_htstep_av.py``
did; a clip under ``--min-cos`` with its stored vector stops the run before any text is encoded, so the query vectors can only
come from a model that was shown to be the teacher. Text: ``encode_text``, batch 1 (``scripts/eval_htstep_av.py``).

Runs in its own environment (transformers 4.57.3, peft 0.20.0, Omni-fix branch htstep-av-extract on PYTHONPATH).

    python -m uemr_htstep.teacher_encode --base-model WAVE-7B --adapter teacher_full --clips clips.json \
        --captions caps.jsonl --store DIR/text --report DIR/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from omni_retrieval.encode import ChunkWriter
from omni_retrieval.manifest import store_keys


def check_clips(encode, clips: list[dict], min_cos: float) -> dict:
    """``encode(clip) -> vector``; cosine of every clip with the row of its stored teacher file."""
    cos = {}
    t0 = time.time()
    for c in clips:
        with np.load(c["npz"]) as z:
            ref = z["emb"][c["row"]].astype(np.float32)
        v = np.asarray(encode(c), np.float32).reshape(-1)
        cos[c["id"]] = float(v @ ref / np.linalg.norm(v) / np.linalg.norm(ref))
    vals = list(cos.values())
    return {"n": len(vals), "min": min(vals), "mean": float(np.mean(vals)), "min_required": min_cos, "ok": min(vals) >= min_cos,
            "sec_per_clip": (time.time() - t0) / max(len(vals), 1), "below": {k: round(v, 5) for k, v in cos.items() if v < min_cos}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("--base-model", "--adapter", "--clips", "--captions", "--store", "--report"):
        p.add_argument(flag, required=True)
    p.add_argument("--min-cos", type=float, default=0.999)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    a = p.parse_args()
    from omniretriever import OmniRetriever

    model = OmniRetriever.from_pretrained(a.base_model, a.adapter, device=a.device, dtype=a.dtype)
    clips = json.load(open(a.clips, encoding="utf-8"))
    rep = check_clips(lambda c: model.encode_av(c["video"], timestamps=(c["start"], c["end"])).float().cpu().numpy(), clips, a.min_cos)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep), flush=True)
    if not rep["ok"]:
        sys.exit(f"CHECK teacher encode_av FAIL: min cos {rep['min']:.5f} < {a.min_cos}")

    caps = [json.loads(line) for line in open(a.captions, encoding="utf-8") if line.strip()]
    store = Path(a.store)
    done = store_keys(store, "text")
    writer = ChunkWriter(store, "text", 200)
    seen: dict[str, np.ndarray] = {}
    t0 = time.time()
    todo = [c for c in caps if c["id"] not in done]
    for c in todo:
        if c["text"] not in seen:
            seen[c["text"]] = model.encode_text(c["text"]).float().reshape(-1).cpu().numpy()
        writer.add(c["id"], seen[c["text"]])
    writer.flush()
    print(f"[teacher text] {len(todo)} captions, {len(seen)} distinct texts, shape {next(iter(seen.values())).shape if seen else None}, "
          f"{(time.time() - t0) / max(len(seen), 1):.3f} s/text", flush=True)


if __name__ == "__main__":
    main()
