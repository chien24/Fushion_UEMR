"""Is this UniAV's teacher (Omni + the LoRA of omni-adapters/checkpoint)? Check on GT clips, then encode every query in the same load.

The clips go through ``OmniRetriever.encode_av(mp4, timestamps=(start, end))`` as ``scripts/extract_htstep_av.py`` did, once with
the user's ``best`` adapter and once with the teacher's, and both are compared with the stored teacher vectors (protocol
``reproduction.teacher_clip_check``). The teacher must be closer than ``best`` on every clip and at least ``--ratio`` times closer
on average, and clear the absolute bars; otherwise the run stops before any text is encoded, so the query vectors can only come
from a model that was shown to be the teacher. Text: ``encode_text``, batch 1 (``scripts/eval_htstep_av.py``).

Runs in its own environment (transformers 4.57.3, peft 0.20.0, Omni-fix branch htstep-av-extract on PYTHONPATH).

    python -m uemr_htstep.teacher_encode --base-model WAVE-7B --adapter teacher_full --best-adapter best_full --clips clips.json \
        --repeat-id "video|start|end" --captions caps.jsonl --store DIR/text --report DIR/report.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

from omni_retrieval.encode import ChunkWriter
from omni_retrieval.manifest import store_keys


def stored_cos(encode, clips: list[dict]) -> tuple[dict, dict]:
    """``encode(clip) -> vector``; ``({id: cosine with the row of its stored teacher file}, {id: vector})``."""
    cos, vecs = {}, {}
    for c in clips:
        with np.load(c["npz"]) as z:
            ref = z["emb"][c["row"]].astype(np.float32)
        v = np.asarray(encode(c), np.float32).reshape(-1)
        cos[c["id"]], vecs[c["id"]] = float(v @ ref / np.linalg.norm(v) / np.linalg.norm(ref)), v
    return cos, vecs


def judge(teacher: dict, best: dict, min_mean: float, min_clip: float, ratio: float, list_below: float) -> dict:
    """The two conditions of the check. ``discriminates``: the teacher beats ``best`` on every clip and mean(1 - cos best) >=
    ratio x mean(1 - cos teacher). ``absolute``: bars on the teacher alone. Clips under ``list_below`` are listed, not fatal."""
    t = np.array([teacher[k] for k in teacher])
    gap_t, gap_b = float(np.mean(1 - t)), float(np.mean([1 - best[k] for k in teacher]))
    not_closer = {k: [round(teacher[k], 5), round(best[k], 5)] for k in teacher if teacher[k] <= best[k]}
    discriminates = not not_closer and gap_b >= ratio * gap_t
    absolute = float(t.mean()) >= min_mean and float(t.min()) >= min_clip
    return {"n": len(t), "mean": float(t.mean()), "min": float(t.min()), "best_mean": float(np.mean([best[k] for k in teacher])),
            "gap_teacher": gap_t, "gap_best": gap_b, "gap_ratio": gap_b / gap_t if gap_t > 0 else None, "ratio_required": ratio,
            "not_closer": not_closer, "discriminates": discriminates, "min_mean": min_mean, "min_clip": min_clip, "absolute": absolute,
            "ok": discriminates and absolute, "below": {k: round(v, 5) for k, v in teacher.items() if v < list_below},
            "cos_teacher": teacher, "cos_best": best}


def pairwise_cos(vectors: list) -> list[float]:
    """Cosine between every pair of repeated encodings of one clip."""
    unit = [np.asarray(v, np.float32) / np.linalg.norm(v) for v in vectors]
    return [float(unit[i] @ unit[j]) for i in range(len(unit)) for j in range(i + 1, len(unit))]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("--base-model", "--adapter", "--best-adapter", "--clips", "--repeat-id", "--captions", "--store", "--report"):
        p.add_argument(flag, required=True)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--av-clips", default="", help="jsonl {id, video, start, end}: encode_av after the check (plan bổ sung §5.5)")
    p.add_argument("--av-store", default="")
    p.add_argument("--min-mean", type=float, required=True)
    p.add_argument("--min-clip", type=float, required=True)
    p.add_argument("--ratio", type=float, required=True)
    p.add_argument("--list-below", type=float, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    a = p.parse_args()
    import torch
    from omniretriever import OmniRetriever

    load = lambda adapter: OmniRetriever.from_pretrained(a.base_model, adapter, device=a.device, dtype=a.dtype)
    encoder = lambda m: lambda c: m.encode_av(c["video"], timestamps=(c["start"], c["end"])).float().cpu().numpy()
    clips = json.load(open(a.clips, encoding="utf-8"))

    best = load(a.best_adapter)
    cos_best, _ = stored_cos(encoder(best), clips)
    del best
    gc.collect()
    torch.cuda.empty_cache()

    model = load(a.adapter)        # this load also encodes the text below
    t0 = time.time()
    cos_teacher, vecs = stored_cos(encoder(model), clips)
    rep = judge(cos_teacher, cos_best, a.min_mean, a.min_clip, a.ratio, a.list_below)
    rep["sec_per_clip"] = (time.time() - t0) / max(len(clips), 1)
    clip = next(c for c in clips if c["id"] == a.repeat_id)
    runs = [vecs[a.repeat_id]] + [np.asarray(encoder(model)(clip), np.float32).reshape(-1) for _ in range(a.repeats)]
    rep["stability"] = {"id": a.repeat_id, "runs": len(runs), "cos_pairs": pairwise_cos(runs)}
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    if not rep["ok"]:
        sys.exit(f"CHECK teacher encode_av FAIL: discriminates={rep['discriminates']} absolute={rep['absolute']}")

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
    if a.av_clips:          # plan bổ sung §5.5: the teacher on the R6 segments, same load, after the check passed
        av = [json.loads(line) for line in open(a.av_clips, encoding="utf-8") if line.strip()]
        done = store_keys(Path(a.av_store), "av")
        writer = ChunkWriter(Path(a.av_store), "av", 200)
        todo, failed, t0 = [c for c in av if c["id"] not in done], [], time.time()
        for i, c in enumerate(todo, 1):
            try:
                writer.add(c["id"], encoder(model)(c).reshape(-1))
            except Exception as e:     # noqa: BLE001 - one clip must not lose the others; listed in the report
                failed.append({"id": c["id"], "error": f"{type(e).__name__}: {e}"})
            if i % 200 == 0:
                writer.flush()
                print(f"[teacher av] {i}/{len(todo)} {(time.time() - t0) / i:.3f} s/clip", flush=True)
        writer.flush()
        rep["av_clips"] = {"todo": len(todo), "failed": failed, "sec_per_clip": (time.time() - t0) / max(len(todo), 1)}
        Path(a.report).write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(f"[teacher av] {len(todo)} clips, {len(failed)} failed", flush=True)


if __name__ == "__main__":
    main()
