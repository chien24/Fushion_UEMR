"""Pack every fusion input into one file per (split, partition) (plan P6 §B4).

    fusion_cache/<split>_Z.pt            video_ids, Z [V, 256, 1024] fp16, t_sec [V, 256], n [V]
    fusion_cache/<split>_<partition>.pt  video_ids, z_index [V'] (row in <split>_Z.pt), E (list [K_v, d] fp16,
                                         L2-normalised, sorted by time), segs (list [K_v, 2] s), duration [V'],
                                         eg [V', d] fp16 + eg_ok [V'] (the R0 vector, for c_g),
                                         captions {caption_id, sentence, video_idx [N], q [N, d] fp16, gt_seg [N, 2]}
    fusion_cache/SOURCE.json             sha256 of every input, counts, commit

Z sits in its own file per split instead of inside every partition file (plan §B4 lists it as a field): the
four partitions of a split share it, which saves about 1.5 GB of Drive per split. ``d`` is read from the caches.

Sources (``--splits train dev val``):
    train  omni_cache_train/<tag>/{segments,text}.npz, uniav_cache/ft_train/{Z/, durations.json}
    dev    omni_cache/<tag>/segments.npz (the 104 dev videos), omni_cache/<tag>/text_dev.npz, uniav_cache/ft/Z
    val    omni_cache/<tag>/segments.npz (the 394 val videos), omni_cache/<tag>/text.npz, uniav_cache/ft/Z

    python -m uemr_fusion.build_cache --out <fusion_cache> --gallery-omni <omni_cache>/ft \\
        --subset <omni_cache>/common/subset_videos.json --gallery-z <uniav_cache>/ft/Z \\
        --train-omni <omni_cache_train>/ft --train-z <uniav_cache>/ft_train/Z \\
        --train-durations <uniav_cache>/ft_train/durations.json
    python -m uemr_fusion.build_cache --out <fusion_cache> --parity-only
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PARTITIONS = ("uniav_pred", "uni_M", "gt", "global")
SPLITS = ("train", "dev", "val")
# Exp 1 column (b), gallery 394 val videos, agg max (NHAN_XET_KET_QUA_P4.md §2.2)
EXPECTED_VAL_R1 = {"global": 8.32, "uni_M": 18.02, "uniav_pred": 24.88, "gt": 37.52}


def _sha_bytes(path) -> tuple[bytes, str]:
    data = Path(path).read_bytes()
    return data, hashlib.sha256(data).hexdigest()


def load_npz(path, sources: dict) -> dict[str, np.ndarray]:
    """Every array of an npz, read once; its sha256 goes into ``sources``."""
    data, sha = _sha_bytes(path)
    sources[str(path)] = sha
    with np.load(io.BytesIO(data)) as blob:
        return {k: blob[k] for k in blob.files}


def segments_by_video(seg: dict, partition: str, videos) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{video: (E [K, d] fp16, segs [K, 2] fp32)}`` of one partition, sorted by (ts, te)."""
    videos = set(videos)
    sel = np.flatnonzero((seg["partition"] == partition) & np.isin(seg["video_id"], list(videos)))
    rows = defaultdict(list)
    for i in sel:
        rows[str(seg["video_id"][i])].append(i)
    out = {}
    for v, idx in rows.items():
        idx = np.array(sorted(idx, key=lambda i: (seg["ts"][i], seg["te"][i])))
        out[v] = (seg["emb"][idx].astype(np.float16),
                  np.stack([seg["ts"][idx], seg["te"][idx]], 1).astype(np.float32))
    return out


def load_z(z_dir, videos, sources: dict) -> dict:
    """Z of ``videos`` in that order; a video without a Z file is an error."""
    zs, ts, ns, h = [], [], [], hashlib.sha256()
    for v in videos:
        path = Path(z_dir) / f"{v}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"no Z for {v} in {z_dir}")
        data, sha = _sha_bytes(path)
        h.update(f"{v}:{sha}\n".encode())
        with np.load(io.BytesIO(data)) as blob:
            zs.append(blob["z"].astype(np.float16))
            ts.append(blob["t_sec"].astype(np.float32))
            ns.append(int(json.loads(str(blob["meta"])).get("n", 0)))
    sources[str(z_dir)] = {"files": len(videos), "sha256_of_list": h.hexdigest()}
    return {"Z": np.stack(zs), "t_sec": np.stack(ts), "n": np.array(ns)}


def split_inputs(split: str, a) -> dict:
    """Paths and video list of one split."""
    if split == "train":
        durations = json.loads(Path(a.train_durations).read_text(encoding="utf-8"))
        return {"segments": Path(a.train_omni) / "segments.npz", "text": Path(a.train_omni) / "text.npz",
                "z_dir": a.train_z, "durations": durations}
    subset = json.loads(Path(a.subset).read_text(encoding="utf-8"))["videos"]
    durations = {r["video_id"]: float(r["duration"]) for r in subset if r["split"] == split}
    text = "text.npz" if split == "val" else "text_dev.npz"
    return {"segments": Path(a.gallery_omni) / "segments.npz", "text": Path(a.gallery_omni) / text,
            "z_dir": a.gallery_z, "durations": durations}


def build_split(split: str, a, sources: dict, npz_cache: dict) -> list[dict]:
    import torch

    inp = split_inputs(split, a)
    key = str(inp["segments"])
    if key not in npz_cache:
        npz_cache[key] = load_npz(inp["segments"], sources)
    seg = npz_cache[key]
    text = load_npz(inp["text"], sources)
    encoded = set(seg["video_id"].tolist())
    videos = sorted(v for v in inp["durations"] if v in encoded)
    dropped = sorted(set(inp["durations"]) - set(videos))
    if dropped:
        print(f"[{split}] {len(dropped)} videos have no Omni vector at all, dropped: {dropped[:5]}")
    z = load_z(inp["z_dir"], videos, sources)
    out = Path(a.out)
    torch.save({"split": split, "video_ids": videos, "Z": torch.from_numpy(z["Z"]),
                "t_sec": torch.from_numpy(z["t_sec"]), "n": torch.from_numpy(z["n"])}, out / f"{split}_Z.pt")
    z_row = {v: i for i, v in enumerate(videos)}
    glob = segments_by_video(seg, "global", videos)

    rows = []
    for part in a.partitions:
        by = segments_by_video(seg, part, videos)
        vids = [v for v in videos if v in by]
        if not vids:
            print(f"[{split}] partition {part!r} not in {inp['segments']} -> skipped")
            continue
        d = int(by[vids[0]][0].shape[1])
        vidx = {v: i for i, v in enumerate(vids)}
        keep = np.array([str(v) in vidx for v in text["video_id"]], dtype=bool)
        eg = np.zeros((len(vids), d), np.float16)
        eg_ok = np.zeros(len(vids), bool)
        for i, v in enumerate(vids):
            if v in glob:
                eg[i], eg_ok[i] = glob[v][0][0], True
        caps = {"caption_id": [str(c) for c in text["caption_id"][keep]],
                "sentence": [str(s) for s in text["sentence"][keep]],
                "video_idx": torch.as_tensor([vidx[str(v)] for v in text["video_id"][keep]], dtype=torch.long),
                "q": torch.from_numpy(text["emb"][keep].astype(np.float16)),
                "gt_seg": torch.from_numpy(np.stack([text["gt_ts"][keep], text["gt_te"][keep]], 1).astype(np.float32))}
        torch.save({"split": split, "partition": part, "d": d, "video_ids": vids,
                    "z_index": torch.as_tensor([z_row[v] for v in vids], dtype=torch.long),
                    "E": [torch.from_numpy(by[v][0]) for v in vids], "segs": [torch.from_numpy(by[v][1]) for v in vids],
                    "duration": torch.as_tensor([inp["durations"][v] for v in vids], dtype=torch.float32),
                    "eg": torch.from_numpy(eg), "eg_ok": torch.from_numpy(eg_ok), "captions": caps},
                   out / f"{split}_{part}.pt")
        k = [len(by[v][0]) for v in vids]
        rows.append({"split": split, "partition": part, "videos": len(vids), "segments": int(sum(k)),
                     "vec/video": round(float(np.mean(k)), 2), "K max": int(max(k)), "d": d,
                     "captions": int(keep.sum()), "captions dropped": int((~keep).sum()),
                     "videos without global": int((~eg_ok).sum())})
        print(f"[{split}] {part}: {rows[-1]}")
    return rows


def parity(cache_dir, expected=EXPECTED_VAL_R1, tol: float = 0.006) -> list[dict]:
    """MaxSim on the cached E (no fusion) over the 394 val videos must give column (b) of P4 exactly.

    Also printed: the same with ``L2norm(E)`` in fp32 (what the fusion outputs at alpha = 0), which may move
    a query or two on near ties. Raises when the plain-E number is off by more than ``tol``.
    """
    import torch

    from omni_retrieval.evaluate import evaluate_partition
    from omni_retrieval.search import SegmentIndex

    rows, bad = [], []
    for part, ref in expected.items():
        path = Path(cache_dir) / f"val_{part}.pt"
        if not path.is_file():
            print(f"[parity] no {path.name}, skipped")
            continue
        c = torch.load(path, weights_only=False)
        text = text_dict(c)
        r = {}
        for name, norm in (("E", False), ("L2norm(E) fp32", True)):
            emb = np.concatenate([e.float().numpy() for e in c["E"]])
            if norm:
                emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            vid, ts, te = flat_meta(c)
            r[name] = evaluate_partition(SegmentIndex(emb, vid, ts, te, part), text)["R@1"]
        rows.append({"partition": part, "expected R@1": ref, "R@1 (E)": round(r["E"], 2),
                     "R@1 (L2norm E)": round(r["L2norm(E) fp32"], 2), "queries": len(text["video_id"])})
        if abs(r["E"] - ref) > tol:
            bad.append(part)
    for row in rows:
        print("[parity]", row)
    if bad:
        raise RuntimeError(f"parity failed for {bad}: the fusion cache does not reproduce column (b)")
    return rows


def text_dict(c: dict) -> dict:
    """``captions`` of a partition file in the ``omni_retrieval.evaluate.load_text`` layout."""
    caps = c["captions"]
    gt = caps["gt_seg"].numpy()
    return {"emb": caps["q"].float().numpy(), "video_id": np.array(c["video_ids"])[caps["video_idx"].numpy()],
            "gt_ts": gt[:, 0], "gt_te": gt[:, 1], "caption_id": np.array(caps["caption_id"])}


def flat_meta(c: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vid = np.concatenate([[v] * len(e) for v, e in zip(c["video_ids"], c["E"])])
    segs = np.concatenate([s.numpy() for s in c["segs"]])
    return vid, segs[:, 0], segs[:, 1]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--gallery-omni", help="omni_cache/<tag> (segments.npz, text.npz, text_dev.npz)")
    p.add_argument("--subset", help="omni_cache/common/subset_videos.json")
    p.add_argument("--gallery-z", help="uniav_cache/ft/Z")
    p.add_argument("--train-omni", help="omni_cache_train/<tag> (segments.npz, text.npz)")
    p.add_argument("--train-z", help="uniav_cache/ft_train/Z")
    p.add_argument("--train-durations", help="uniav_cache/ft_train/durations.json")
    p.add_argument("--splits", nargs="*", default=list(SPLITS), choices=SPLITS)
    p.add_argument("--partitions", nargs="*", default=list(PARTITIONS))
    p.add_argument("--parity-only", action="store_true")
    p.add_argument("--no-parity", action="store_true", help="skip the column (b) check (fake data in the tests)")
    a = p.parse_args(argv)
    out = Path(a.out)
    if not a.parity_only:
        out.mkdir(parents=True, exist_ok=True)
        sources, npz_cache, rows = {}, {}, []
        for split in a.splits:
            rows += build_split(split, a, sources, npz_cache)
        src = out / "SOURCE.json"
        blob = json.loads(src.read_text(encoding="utf-8")) if src.is_file() else {}
        from omni_retrieval.partitions_uemr import _git_commit
        blob.update({"sources": {**blob.get("sources", {}), **sources}, "commit": _git_commit(),
                     "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "files": {**blob.get("files", {}), **{f"{r['split']}_{r['partition']}": r for r in rows}}})
        src.write_text(json.dumps(blob, indent=1), encoding="utf-8")
    if "val" in a.splits and not a.no_parity:
        par = parity(out)
        src = out / "SOURCE.json"
        blob = json.loads(src.read_text(encoding="utf-8")) if src.is_file() else {}
        blob["parity"] = par
        src.write_text(json.dumps(blob, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
