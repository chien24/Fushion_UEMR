"""UEMR P2 partitions (rand_M, kmedoids_M, uni_chunk) on fake data. No model, no GPU."""

import json

import numpy as np
import pytest

from omni_retrieval.partitions import MIN_SEG_SEC
from omni_retrieval.partitions_uemr import (D_MIN, build_controls, check, check_shift, coverage, kmedoids, kmedoids_m,
                                            rand_m, run, shift, uni_chunk, union,
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


# --------------------------------------------------------------------------- #
# Controls of plan P6 §A3                                                     #
# --------------------------------------------------------------------------- #

def test_shift_keeps_count_and_lengths():
    segs = [[5.0, 20.0], [30.0, 50.0], [52.5, 54.5]]
    a = shift(segs, 60.0, video_rng(0, "vidA#uniav_shift"))
    assert len(a) == 3 and a == sorted(a)
    assert sorted(round(te - ts, 6) for ts, te in a) == [2.0, 15.0, 20.0]
    assert all(0 <= ts < te <= 60.0 and ts <= 60.0 - MIN_SEG_SEC for ts, te in a)
    assert a == shift(segs, 60.0, video_rng(0, "vidA#uniav_shift"))              # reproducible
    assert a != shift(segs, 60.0, video_rng(1, "vidA#uniav_shift"))
    assert shift([[0.0, 60.0]], 60.0, video_rng(0, "v")) == [[0.0, 60.0]]          # no room: stays at 0
    # a 0.5 s GT event must not start in the last second (clamp would drop it at encode time)
    short = shift([[10.0, 10.5]] * 200, 20.0, video_rng(0, "v"))
    assert max(ts for ts, _ in short) <= 20.0 - MIN_SEG_SEC


def test_union_and_coverage():
    assert union([[0.0, 14.0], [5.0, 20.0]], [[0.001, 14.004], [14.0, 28.0]]) == [[0.0, 14.0], [5.0, 20.0],
                                                                                  [14.0, 28.0]]
    assert coverage([[0, 10], [5, 20], [30, 40]], 50.0) == pytest.approx(30 / 50)   # 0-20 and 30-40
    assert coverage([[0, 50]], 50.0) == pytest.approx(1.0)


def test_build_controls(tmp_path):
    durations = {"vidA": 60.0, "vidB": 100.0}
    files = {"uniav_pred": {"vidA": [[5.0, 20.0], [30.0, 50.0]], "vidB": [[10.0, 40.0]]},
             "gt": {"vidA": [[0.0, 10.0]], "vidB": [[3.0, 9.0], [60.0, 75.5]]},
             "uni_chunk": {"vidA": [[0.0, 14.0], [14.0, 28.0], [28.0, 42.0], [42.0, 60.0]],
                           "vidB": [[i * 14.0, min((i + 1) * 14.0, 100.0)] for i in range(7)]}}
    for name, part in files.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(part))
    parts, rows = build_controls(tmp_path, durations, seed=0)
    assert set(parts) == {"uniav_shift", "gt_shift", "uniav_pred+uni_chunk"}
    assert [len(parts["uniav_shift"][v]) for v in ("vidA", "vidB")] == [2, 1]
    assert [len(parts["gt_shift"][v]) for v in ("vidA", "vidB")] == [1, 2]
    assert len(parts["uniav_pred+uni_chunk"]["vidA"]) == 6
    by = {r["partition"]: r for r in rows}
    assert by["uniav_shift"]["coverage source"] == pytest.approx(100 * (35 / 60 + 30 / 100) / 2, abs=0.01)
    # a shifted partition with a wrong length is caught
    bad = {"vidA": [[0.0, 10.0], [20.0, 41.0]], "vidB": [[0.0, 30.0]]}
    with pytest.raises(AssertionError):
        check_shift("uniav_shift", bad, files["uniav_pred"], durations)
    # a video without segments in a source file is an error
    (tmp_path / "gt.json").write_text(json.dumps({"vidA": [[0.0, 10.0]]}))
    with pytest.raises(ValueError):
        build_controls(tmp_path, durations)
