"""Encode segments (``av``) and captions (``text``) exactly as ``eval_youcookii.py`` does.

Why not ``omniretriever.cli extract``: the fine-tuned adapter was scored with
``scripts/eval_youcookii.py``, which feeds the *training* pipeline
(``LazySupervisedDataset``: decord, frames resized to 50176 px keeping the aspect ratio,
chat-template prompt, doubled ``<|AUDIO|>`` slots for BEATs). The CLI goes through
``omniretriever.inference`` (PyAV, centre-crop, WAVE's 336 px floor, no doubling), so
its vectors are not the ones the eval measured. Here nothing in Omni-fix is changed:
``MockDataArgs`` and the dataset/collator are imported from it, and the model is loaded
with the same steps as ``eval_youcookii.main``.

* ``av``  : one record per segment, no caption turn -> ``outputs.mllm_embeds``
  (all-layer fusion head over ``<video>Please describe the video.<|im_end|>``).
* ``text``: ``caption + <|im_end|>`` through the thinker's text model, last token of the
  last layer -- the label branch of ``Qwen2_5OmniThinkerForConditionalGeneration.forward``
  that produced ``text_embeds`` in the eval. This is *not* ``OmniRetriever.encode_text``
  (which routes text through ``classify_linear``); that vector lives in another space.

Output is a directory of ``chunk_*.npz`` files keyed ``<id>__<modality>`` (L2-normalised
fp32, as the eval normalises). A rerun skips every id already in a chunk, so a broken
Colab session loses at most one chunk. Records that fail to load are logged to
``failed.jsonl`` instead of being silently swapped for a random sample, which is what the
dataset does when ``run_test`` is off.

    python -m omni_retrieval.encode --modality av --manifest segs_todo.jsonl --store raw/av \\
        --omni-repo /content/Omni-fix --base-model .../WAVE-7B --beats-path .../BEATs.pt \\
        --adapter .../best --video-root /content/videos
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import tempfile
import sys
import time
from pathlib import Path

import numpy as np

from .manifest import store_keys

PAD_TOKEN_ID = 151643  # training/qwenvl/train/utils.py


# --------------------------------------------------------------------------- #
# Model, loaded the way scripts/eval_youcookii.py loads it                     #
# --------------------------------------------------------------------------- #

def import_eval_module(omni_repo: str):
    """Import ``scripts/eval_youcookii.py`` as a module (its ``main`` is not run).

    Importing it also puts ``training/`` and ``src/`` on ``sys.path``, which is what makes
    ``qwenvl`` importable here.
    """
    path = Path(omni_repo) / "scripts" / "eval_youcookii.py"
    spec = importlib.util.spec_from_file_location("eval_youcookii", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(omni_repo: str, base_model: str, beats_path: str, adapter: str,
               device: str = "cuda", dtype: str = "bfloat16"):
    """Return ``(model, processor, eval_module)``; mirrors ``eval_youcookii.main`` steps 1-2."""
    ev = import_eval_module(omni_repo)
    torch = ev.torch
    for name, p in (("base model", base_model), ("BEATs", beats_path),
                    ("adapter", os.path.join(adapter, "adapter_model.safetensors"))):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{name} not found: {p}")
    os.environ["BEATS_PATH"] = beats_path
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype, torch.float32)

    processor = ev.Qwen2_5OmniProcessor.from_pretrained(base_model)
    cfg = ev.Qwen2_5OmniThinkerConfig.from_pretrained(base_model)
    if hasattr(cfg, "text_config"):
        if getattr(cfg.text_config, "pad_token_id", None) is None:
            cfg.text_config.pad_token_id = getattr(cfg, "pad_token_id", 151643)
        if getattr(cfg.text_config, "bos_token_id", None) is None:
            cfg.text_config.bos_token_id = getattr(cfg, "bos_token_id", 151644)
        if getattr(cfg.text_config, "eos_token_id", None) is None:
            cfg.text_config.eos_token_id = getattr(cfg, "eos_token_id", 151645)
    cfg.train_classify = True
    cfg.classify_type = "all_layer"
    cfg.audio_config.beats_path = beats_path
    cfg.audio_config.beats_only = False

    model = ev.Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        base_model, config=cfg, torch_dtype=torch_dtype)
    beats_ckpt = torch.load(beats_path, map_location="cpu", weights_only=False)
    model.beats.load_state_dict(beats_ckpt["model"])
    model = ev.PeftModel.from_pretrained(model, adapter)
    model = model.to(device).eval()
    return model, processor, ev


def preprocessing_params(ev, processor) -> dict:
    """The data args the eval used, read from ``MockDataArgs`` itself (for meta.json)."""
    args = ev.MockDataArgs("<manifest>", processor)
    keys = ("video_max_frames", "video_min_frames", "base_interval", "max_pixels", "min_pixels",
            "fixed_audio_duration", "use_beats", "beats_only", "train_classify", "classify_type",
            "sampling_rate", "feature_size", "chunk_length", "hop_length")
    params = {k: getattr(args, k) for k in keys}
    params.update({
        "pipeline": "scripts/eval_youcookii.py (LazySupervisedDataset, batch 1)",
        "av_prompt": "<video>\\nPlease describe the video. via chat template, <|AUDIO|> doubled",
        "text": "caption + <|im_end|>, thinker.model last layer, last token (eval label branch)",
        "normalize": "L2",
    })
    return params


def git_commit(repo: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------- #
# Encoders                                                                    #
# --------------------------------------------------------------------------- #

def _to_device(batch: dict, device, torch_dtype, torch) -> dict:
    """Move a collated batch to the device; the same rule as eval_youcookii's loop."""
    out = {}
    for k, v in batch.items():
        if v is None:
            continue
        if isinstance(v, torch.Tensor):
            if k in ("pixel_values", "pixel_values_videos", "input_features") and v.dtype == torch.float32:
                out[k] = v.to(device, dtype=torch_dtype)
            else:
                out[k] = v.to(device)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            out[k] = [t.to(device) for t in v]
        else:
            out[k] = v
    return out


def encode_texts(model, tokenizer, texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Caption embeddings, L2-normalised ``[N, D]`` fp32.

    Reproduces the label branch: ``tokenizer(caption + "<|im_end|>")`` -> input embeddings ->
    ``thinker.model`` -> ``[0][:, -1, :]``. Captions are batched only with captions of the
    same token length, so no padding (and no shifted positions) enters a batch.
    """
    import torch

    thinker = model.get_base_model() if hasattr(model, "get_base_model") else model
    device = next(thinker.parameters()).device
    ids = [tokenizer(t if t.endswith("<|im_end|>") else t + "<|im_end|>", padding=True,
                     padding_side="left", return_tensors="pt")["input_ids"][0] for t in texts]
    by_len: dict[int, list[int]] = {}
    for i, row in enumerate(ids):
        by_len.setdefault(len(row), []).append(i)
    vecs: dict[int, np.ndarray] = {}
    with torch.inference_mode():
        for _, idx in sorted(by_len.items()):
            for s in range(0, len(idx), batch_size):
                part = idx[s:s + batch_size]
                label_ids = torch.stack([ids[i] for i in part]).to(device)
                mask = label_ids.ne(PAD_TOKEN_ID).to(torch.int64)
                hidden = thinker.model(inputs_embeds=thinker.get_input_embeddings()(label_ids),
                                       attention_mask=mask, output_hidden_states=True,
                                       return_dict=True)[0][:, -1, :].float()
                hidden = hidden / hidden.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                for i, v in zip(part, hidden.cpu().numpy()):
                    vecs[i] = v
    return np.stack([vecs[i] for i in range(len(texts))]) if texts else np.zeros((0, 0), np.float32)


def _first(batch):
    return batch[0]


class _Records:
    """Dataset wrapper: ``(index, item or None, error)``; never substitutes a sample."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        try:
            return i, self.ds._get_item(i), None
        except Exception as e:  # noqa: BLE001
            return i, None, repr(e)


class ChunkWriter:
    def __init__(self, store: Path, modality: str, chunk_size: int):
        self.store, self.modality, self.chunk_size = store, modality, chunk_size
        store.mkdir(parents=True, exist_ok=True)
        self.buf: dict[str, np.ndarray] = {}
        self.failed = store / "failed.jsonl"

    def add(self, rid: str, vec: np.ndarray) -> None:
        self.buf[f"{rid}__{self.modality}"] = vec.astype(np.float32)
        if len(self.buf) >= self.chunk_size:
            self.flush()

    def fail(self, rid: str, error: str) -> None:
        with open(self.failed, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": rid, "error": error}) + "\n")

    def flush(self) -> None:
        if not self.buf:
            return
        name = f"chunk_{time.strftime('%Y%m%d-%H%M%S')}_{os.getpid()}_{len(list(self.store.glob('chunk_*.npz'))):05d}"
        tmp = self.store / f"{name}.tmp.npz"
        np.savez(tmp, **self.buf)
        os.replace(tmp, self.store / f"{name}.npz")
        self.buf = {}


def encode_av(model, processor, ev, records_path: Path, writer: ChunkWriter, device: str,
              dtype: str, num_workers: int) -> dict:
    import torch
    from torch.utils.data import DataLoader

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype, torch.float32)
    data_args = ev.MockDataArgs(str(records_path), processor)
    # run_test only changes error handling here: a record that fails to load comes back
    # as None (logged below) instead of being replaced by a random other sample.
    data_args.run_test = True
    ds = ev.LazySupervisedDataset(tokenizer=processor.tokenizer, data_args=data_args)
    ids = [r["id"] for r in ds.list_data_dict]
    collator = ev.DataCollatorForOmniDataset()
    # Batch 1 like the eval: the fusion head pools a fixed last position, so padding
    # would change the embeddings. Workers only overlap the CPU decode with the GPU.
    loader = DataLoader(_Records(ds), batch_size=1, shuffle=False, collate_fn=_first,
                        num_workers=num_workers, prefetch_factor=4 if num_workers else None)

    times, t_prev, n_ok = [], None, 0
    with torch.inference_mode():
        for i, item, err in ev.tqdm(loader, total=len(ds), desc="av"):
            if item is None:
                writer.fail(ids[i], err or "dataset returned None")
                continue
            batch = _to_device(collator([item]), device, torch_dtype, torch)
            out = model(**batch, output_hidden_states=True, pred_embeds=True, return_dict=True)
            emb = out.mllm_embeds.float()
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            writer.add(ids[i], emb[0].cpu().numpy())
            n_ok += 1
            now = time.time()
            if t_prev is not None:
                times.append(now - t_prev)
            t_prev = now
    writer.flush()
    return {"encoded": n_ok, "failed": len(ds) - n_ok,
            "sec_per_clip": float(np.median(times)) if times else None,
            "mean_sec_per_clip": float(np.mean(times)) if times else None}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--modality", choices=("av", "text"), required=True)
    p.add_argument("--manifest", required=True, help="segs_todo.jsonl or caps_todo.jsonl")
    p.add_argument("--store", required=True, help="chunk directory (raw/av or raw/text)")
    p.add_argument("--omni-repo", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--beats-path", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--video-root", default=None, help="sets $VIDEO_ROOT for the bare mp4 names")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    p.add_argument("--num-workers", type=int, default=4, help="decode workers (av)")
    p.add_argument("--batch-size", type=int, default=32, help="text only; av is always 1")
    p.add_argument("--chunk-size", type=int, default=200, help="embeddings per chunk file")
    p.add_argument("--limit", type=int, default=None, help="encode only the first N pending records")
    p.add_argument("--timing-out", default=None, help="write timing JSON here")
    a = p.parse_args(argv)

    if a.video_root:
        os.environ["VIDEO_ROOT"] = a.video_root
    # LazySupervisedDataset appends one line per clip to $DATA_LOADING_LOG (default ./tmp/...);
    # keep it off Drive and out of the code checkout.
    os.environ.setdefault("DATA_LOADING_LOG", os.path.join(tempfile.gettempdir(), "omni_data_loading.log"))
    store = Path(a.store)
    with open(a.manifest, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    done = store_keys(store, a.modality)
    pending = [r for r in records if r["id"] not in done]
    n_left = len(pending)
    if a.limit is not None:
        pending = pending[:a.limit]
    print(f"[encode {a.modality}] manifest {len(records)} | already in store {len(records) - n_left} "
          f"| to do now {len(pending)}")
    if not pending:
        return 0

    t0 = time.time()
    model, processor, ev = load_model(a.omni_repo, a.base_model, a.beats_path, a.adapter, a.device, a.dtype)
    load_sec = time.time() - t0
    writer = ChunkWriter(store, a.modality, a.chunk_size)
    meta = {"adapter": a.adapter, "base_model": a.base_model, "beats_path": a.beats_path,
            "omni_commit": git_commit(a.omni_repo), "dtype": a.dtype,
            "preprocessing": preprocessing_params(ev, processor)}
    (store / "encoder_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    t1 = time.time()
    if a.modality == "av":
        pending_path = store / "_pending.jsonl"
        with open(pending_path, "w", encoding="utf-8") as f:
            for r in pending:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats = encode_av(model, processor, ev, pending_path, writer, a.device, a.dtype, a.num_workers)
    else:
        vecs = encode_texts(model, processor.tokenizer, [r["text"] for r in pending], a.batch_size)
        for r, v in zip(pending, vecs):
            writer.add(r["id"], v)
        writer.flush()
        stats = {"encoded": len(pending), "failed": 0,
                 "sec_per_clip": (time.time() - t1) / max(len(pending), 1)}
    stats.update({"model_load_sec": load_sec, "wall_sec": time.time() - t1, "n": len(pending)})
    print(f"[encode {a.modality}] {json.dumps(stats)}")
    with open(store / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(stats, finished=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
    if a.timing_out:
        Path(a.timing_out).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
