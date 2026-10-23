"""uemr2.data against UEMR's own scorer, uemr2.folds on a toy Athena repo (synthetic caches, CPU)."""
import json

import numpy as np

from omni_retrieval.evaluate import load_text, text_to_clip
from omni_retrieval.manifest import seg_key
from uemr2 import data as D, folds as Fo, uemr2_core as U


def write_cache(root, n_vid=12, seed=0, d=32):
    """segments.npz with gt / uniav_pred / r8 (= R6 + fills) rows in shuffled order, text.npz with one caption per GT."""
    rng = np.random.default_rng(seed)
    rows, caps = [], []
    for v in range(n_vid):
        vid = f"v{v:02d}"
        gt = [[10.0 * k, 10.0 * k + 8] for k in range(4)]
        r6 = [[10.0 * k + 1, 10.0 * k + 9] for k in (0, 2)]
        r8 = sorted(r6 + [[9.0, 21.0], [29.0, 40.0]])
        for p, segs in (("gt", gt), ("uniav_pred", r6), ("r8", r8)):
            rows += [(p, vid, a, b) for a, b in segs]
        caps += [(f"{vid}_{k}", vid, a, b, f"step {k % 3} of {v % 4}") for k, (a, b) in enumerate(gt)]
    perm = rng.permutation(len(rows))
    rows = [rows[i] for i in perm]
    emb = rng.standard_normal((len(rows), d)).astype(np.float16)
    np.savez(root / "segments.npz", keys=np.array([f"{r[1]}__{r[0]}__{i}" for i, r in enumerate(rows)]),
             seg_key=np.array([seg_key(r[1], r[2], r[3]) for r in rows]), video_id=np.array([r[1] for r in rows]),
             partition=np.array([r[0] for r in rows]), ts=np.array([r[2] for r in rows], np.float32),
             te=np.array([r[3] for r in rows], np.float32), emb=emb)
    t = rng.standard_normal((len(caps), d)).astype(np.float32)
    np.savez(root / "text.npz", caption_id=np.array([c[0] for c in caps]), video_id=np.array([c[1] for c in caps]),
             sentence=np.array([c[4] for c in caps]), gt_ts=np.array([c[2] for c in caps], np.float32),
             gt_te=np.array([c[3] for c in caps], np.float32),
             gt_seg_key=np.array([seg_key(c[1], c[2], c[3]) for c in caps]), emb=t / np.linalg.norm(t, axis=1, keepdims=True))
    return caps


def test_gallery_reproduces_uemr_ranks_and_fill(tmp_path):
    write_cache(tmp_path)
    text = load_text(tmp_path)
    for part in ("gt", "uniav_pred", "r8"):
        g = D.gallery(tmp_path, part)
        q = D.queries(tmp_path)
        ref = D.reference(tmp_path, part, text)
        r = U.run([(1.0, D.scores(q["T"], g["E"]))], g["seg_vid"], len(g["videos"]), D.cap_vid(q, g),
                  q["gt"], g["segs"])
        assert np.array_equal(r["rank"], ref["rank"]) and np.array_equal(r["moment_hit"] & (r["rank"] == 1), ref["vcmr"])
    g8 = D.mark_fill(D.gallery(tmp_path, "r8"), D.gallery(tmp_path, "uniav_pred"))
    assert g8["is_fill"].sum() == 2 * 12 and (~g8["is_fill"]).sum() == 2 * 12
    assert np.allclose(g8["segs"][g8["is_fill"]][:2], [[9, 21], [29, 40]])


def test_train_set_and_text_to_clip(tmp_path):
    caps = write_cache(tmp_path)
    rows = D.train_rows(tmp_path)
    q = D.queries(tmp_path)
    iv2 = {D.iv2_key(c[0]): np.ones(8, np.float32) for c in caps}
    ts = D.train_set(rows, q, iv2)
    gt = rows["partition"] == "gt"
    sent = ts["pool_sentence"][ts["cap"][gt]]
    own = {k: s for k, s in zip(q["gt_seg_key"].tolist(), q["sentence"].tolist())}
    assert sent.tolist() == [own[k] for k in rows["seg_key"][gt].tolist()]
    assert ts["counts"]["unique captions (pool)"] == len(set(q["sentence"].tolist())) == len(ts["T_pool"])
    assert ts["counts"]["r6 without caption"] == 0          # [1, 9] vs GT [0, 8]: tIoU 7/9
    g7 = D.gallery(tmp_path, "gt")
    mine = D.text_to_clip(q["T"], g7["E"][D.clip_rows(q, g7)])
    ref = text_to_clip(tmp_path)["eval"]
    assert all(abs(mine[k] - ref[k]) < 1e-9 for k in ("R@1", "R@5", "R@10"))


def test_folds(tmp_path):
    vids = [f"t{i:03d}" for i in range(11)]
    db = {v: {"subset": "training", "annotations": []} for v in vids + ["nofeat1", "nofeat2"]}
    db.update({"d0": {"subset": "dev", "annotations": []}, "x0": {"subset": "validation", "annotations": []}})
    (tmp_path / "data/youcookii/annotations").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / Fo.ANNO).write_text(json.dumps({"database": db}))
    (tmp_path / Fo.CONFIG).write_text("dataset:\n  json_file: ./data/youcookii/annotations/youcookii_annotations_yc2clean.json\n")
    r = Fo.write_folds(tmp_path, vids, exclude={"d0", "x0"})
    assert (len(r["A"]), len(r["B"])) == (6, 5) and sorted(r["A"] + r["B"]) == vids
    assert r == Fo.write_folds(tmp_path, vids)                 # deterministic
    a = json.loads((tmp_path / r["files"]["A"]["annotation"]).read_text())["database"]
    assert sorted(v for v, x in a.items() if x["subset"] == "training") == r["A"]
    assert a["nofeat1"]["subset"] == "holdout" and a["d0"]["subset"] == "dev" and a["x0"]["subset"] == "validation"
    cfg = (tmp_path / r["files"]["B"]["config"]).read_text()
    assert "youcookii_annotations_yc2clean_foldB.json" in cfg
