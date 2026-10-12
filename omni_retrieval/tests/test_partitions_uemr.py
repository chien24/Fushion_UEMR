"""UEMR P2 partitions (rand_M, kmedoids_M, uni_chunk) on fake data. No model, no GPU."""

import json

import numpy as np
import pytest

from omni_retrieval.partitions_uemr import (D_MIN, check, kmedoids, kmedoids_m, rand_m, run, uni_chunk,
                                            video_rng, window)


def _contiguous(segs, duration):
    assert segs[0][0] == pytest.approx(0) and segs[-1][1] == pytest.approx(duration)
    assert all(a[1] == pytest.approx(b[0]) for a, b in zip(segs, segs[1:]))


def test_rand_m_tiles_and_is_reproducible():
    segs = rand_m(100.0, 7, video_rng(0, "vidA"))
    assert len(segs) == 7
    _contiguous(segs, 100.0)
    assert all(te - ts >= D_MIN - 1e-9 for ts, te in segs)
    assert segs == rand_m(100.0, 7, video_rng(0, "vidA"))          # same seed, same video
    assert segs != rand_m(100.0, 7, video_rng(0, "vidB"))          # another video
    assert rand_m(10.0, 1, video_rng(0, "vidA")) == [[0.0, 10.0]]


def _clustered_z(sizes, d=8, seed=0):
    """Rows of cluster c sit next to basis vector e_c; returns (z, cluster label per row)."""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for c, n in enumerate(sizes):
        rows.append(np.eye(d)[c] + 0.01 * rng.normal(size=(n, d)))
        labels += [c] * n
    return np.concatenate(rows).astype(np.float16), np.array(labels)


def test_kmedoids_one_medoid_per_cluster():
    z, labels = _clustered_z((100, 150, 6))
    med, it, ok = kmedoids(z, 3, video_rng(0, "vidA"))
    assert ok and it <= 100
    assert sorted(labels[med]) == [0, 1, 2]


def test_kmedoids_windows_stay_inside_video():
    # the last cluster is rows 250..255, whose t_sec (computed from n = 110 feature rows) is past the
    # 100 s mp4, as for wii9jNiNl9Y
    z, _ = _clustered_z((100, 150, 6))
    t_sec = (np.arange(256) + 0.5) * 110 / 256
    segs, _, ok = kmedoids_m(z, t_sec, 100.0, 3, video_rng(0, "vidA"))
    assert ok and len(segs) == 3
    assert all(0 <= ts < te <= 100.0 and te - ts == pytest.approx(2.0) for ts, te in segs)
    assert segs[-1] == [98.0, 100.0]


def test_window_shifts_inwards():
    assert window(0.3, 100.0) == [0.0, 2.0]
    assert window(150.0, 100.0) == [98.0, 100.0]
    assert window(50.0, 100.0) == [49.0, 51.0]
    assert window(0.5, 1.5) == [0.0, 1.5]


def test_uni_chunk():
    assert uni_chunk(30.0) == [[0, 14.0], [14.0, 30.0]]          # last 2 s chunk merged
    assert uni_chunk(42.0) == [[0, 14.0], [14.0, 28.0], [28.0, 42.0]]
    assert uni_chunk(5.0) == [[0, 5.0]]


def test_check_catches_wrong_count():
    durations, k_pred = {"v": 30.0}, {"v": 2}
    check("uni_M", {"v": [[0.0, 15.0], [15.0, 30.0]]}, durations, k_pred)
    with pytest.raises(AssertionError):
        check("uni_M", {"v": [[0.0, 30.0]]}, durations, k_pred)
    with pytest.raises(AssertionError):                          # gap
        check("uni_chunk", {"v": [[0.0, 14.0], [15.0, 30.0]]}, durations, k_pred)


def test_run_end_to_end(tmp_path):
    subset = {"settings": {}, "videos": [{"video_id": "vidA", "split": "val", "duration": 60.0},
                                         {"video_id": "vidB", "split": "dev", "duration": 100.0}]}
    (tmp_path / "subset_videos.json").write_text(json.dumps(subset))
    seg_lines = [{"video_id": "vidA", "ts": 5.0, "te": 20.0}, {"video_id": "vidA", "ts": 30.0, "te": 50.0},
                 {"video_id": "vidB", "ts": 10.0, "te": 40.0}, {"video_id": "vidB", "ts": 50.0, "te": 70.0},
                 {"video_id": "vidB", "ts": 80.0, "te": 99.0}]
    seg_path = tmp_path / "segments_pred.jsonl"
    seg_path.write_text("".join(json.dumps(r) + "\n" for r in seg_lines))
    zdir = tmp_path / "Z"
    zdir.mkdir()
    for vid, n in (("vidA", 60), ("vidB", 100)):
        z, _ = _clustered_z((90, 90, 76), seed=n)
        np.savez(zdir / f"{vid}.npz", z=z, t_sec=((np.arange(256) + 0.5) * n / 256).astype(np.float32))

    out = tmp_path / "partitions"
    rows = {r["partition"]: r for r in run(out, seg_path, zdir, tmp_path / "subset_videos.json")}
    for name in ("uniav_pred", "uni_M", "rand_M", "kmedoids_M"):
        assert rows[name]["segments"] == 5, name
    assert rows["uni_chunk"]["segments"] == 5 + 7                 # 60 s -> 4 x 14 + 4; 100 s -> 6 x 14 + 16
    assert all(r["dropped by clamp"] == 0 for r in rows.values())
    parts = json.loads((out / "uniav_pred.json").read_text())
    assert parts["vidB"] == [[10.0, 40.0], [50.0, 70.0], [80.0, 99.0]]
    source = json.loads((out / "SOURCE.json").read_text())
    assert set(source["partitions"]) == {"uniav_pred", "uni_M", "rand_M", "kmedoids_M", "uni_chunk"}
    assert (out / "kmedoids_M.meta.json").is_file()

    # same input -> identical files; another segments_pred.jsonl -> refused
    before = (out / "rand_M.json").read_text()
    run(out, seg_path, zdir, tmp_path / "subset_videos.json")
    assert (out / "rand_M.json").read_text() == before
    seg_path.write_text("".join(json.dumps(r) + "\n" for r in seg_lines[:-1]))
    with pytest.raises(RuntimeError):
        run(out, seg_path, zdir, tmp_path / "subset_videos.json")
