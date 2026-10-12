"""A tiny fake world for the fusion tests: gallery + train caches in the real file layouts, small d. CPU only."""

import json

import numpy as np
import pytest

D = 16


def _unit(x):
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


def _segments_npz(path, rows, emb):
    np.savez(path, keys=np.array([f"{r[0]}__{r[1]}__{r[4]}" for r in rows]),
             seg_key=np.array([f"{r[0]}__{r[2]:.2f}_{r[3]:.2f}" for r in rows]),
             video_id=np.array([r[0] for r in rows]), partition=np.array([r[1] for r in rows]),
             ts=np.array([r[2] for r in rows], np.float32), te=np.array([r[3] for r in rows], np.float32),
             emb=emb.astype(np.float16))


def _text_npz(path, caps, emb):
    np.savez(path, caption_id=np.array([c[0] for c in caps]), video_id=np.array([c[1] for c in caps]),
             sentence=np.array([f"caption {c[0]}" for c in caps]),
             gt_ts=np.array([c[2] for c in caps], np.float32), gt_te=np.array([c[3] for c in caps], np.float32),
             gt_seg_key=np.array([""] * len(caps)), emb=emb.astype(np.float16))


def _z_dir(path, videos, durations, rng):
    path.mkdir(parents=True, exist_ok=True)
    for v in videos:
        n = int(round(durations[v]))
        np.savez(path / f"{v}.npz", z=rng.normal(size=(256, 1024)).astype(np.float16),
                 t_sec=((np.arange(256) + 0.5) * n / 256).astype(np.float32),
                 meta=np.array(json.dumps({"level": 0, "n": n})))


def make_world(root, seed=0, n_val=5, n_dev=3, n_train=8, k=4, caps_per_video=2):
    """Write the sources of ``build_cache`` under ``root``; returns the CLI arguments."""
    rng = np.random.default_rng(seed)
    splits = {"val": [f"val{i}" for i in range(n_val)], "dev": [f"dev{i}" for i in range(n_dev)],
              "train": [f"tr{i}" for i in range(n_train)]}
    dur = {v: float(60 + 10 * i) for i, v in enumerate(sum(splits.values(), []))}

    def world(videos):
        rows, emb, caps, q = [], [], [], []
        for v in videos:
            step = dur[v] / k
            vecs = _unit(rng.normal(size=(k, D)))
            for j in range(k):                    # uniav_pred: k segments, gt: the same windows
                for part in ("uniav_pred", "gt"):
                    rows.append((v, part, j * step, (j + 1) * step, j))
                    emb.append(vecs[j])
            rows.append((v, "global", 0.0, dur[v], 0))
            emb.append(_unit(vecs.mean(0)))
            for c in range(caps_per_video):       # a caption = its segment's vector + noise
                j = (c * 2) % k
                caps.append((f"{v}_{c}", v, j * step, (j + 1) * step))
                q.append(_unit(vecs[j] + 0.8 * rng.normal(size=D)))
        return rows, np.stack(emb), caps, np.stack(q)

    gal, tr = root / "omni_cache" / "ft", root / "omni_cache_train" / "ft"
    gal.mkdir(parents=True)
    tr.mkdir(parents=True)
    rows, emb, caps, q = world(splits["val"] + splits["dev"])
    _segments_npz(gal / "segments.npz", rows, emb)
    val_caps = [i for i, c in enumerate(caps) if c[1].startswith("val")]
    dev_caps = [i for i, c in enumerate(caps) if c[1].startswith("dev")]
    _text_npz(gal / "text.npz", [caps[i] for i in val_caps], q[val_caps])
    _text_npz(gal / "text_dev.npz", [caps[i] for i in dev_caps], q[dev_caps])
    rows, emb, caps, q = world(splits["train"])
    _segments_npz(tr / "segments.npz", rows, emb)
    _text_npz(tr / "text.npz", caps, q)

    subset = root / "subset_videos.json"
    subset.write_text(json.dumps({"videos": [{"video_id": v, "split": s, "duration": dur[v]}
                                            for s in ("val", "dev") for v in splits[s]]}))
    _z_dir(root / "Z_gallery", splits["val"] + splits["dev"], dur, rng)
    _z_dir(root / "Z_train", splits["train"], dur, rng)
    (root / "durations.json").write_text(json.dumps({v: dur[v] for v in splits["train"]}))
    return ["--gallery-omni", str(gal), "--subset", str(subset), "--gallery-z", str(root / "Z_gallery"),
            "--train-omni", str(tr), "--train-z", str(root / "Z_train"),
            "--train-durations", str(root / "durations.json")]


@pytest.fixture
def fusion_cache(tmp_path):
    from uemr_fusion.build_cache import main

    args = make_world(tmp_path)
    out = tmp_path / "fusion_cache"
    main(["--out", str(out), *args, "--partitions", "uniav_pred", "gt", "global", "--no-parity"])
    return out
