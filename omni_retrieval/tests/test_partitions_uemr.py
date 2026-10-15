"""UEMR P2 partitions (rand_M, kmedoids_M, uni_chunk) on fake data. No model, no GPU."""

import json

import numpy as np
import pytest

from omni_retrieval.partitions import MIN_SEG_SEC
from omni_retrieval.partitions_uemr import (D_MIN, NOISE_TOL, best_tiou, build_controls, calibrate_sigma, check,
                                            check_shift, coverage, kmedoids, kmedoids_m, noise_directions, noise_name,
                                            perturb, rand_m, run, run_noise, shift, uni_chunk, union, video_rng,
                                            window)


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


# --------------------------------------------------------------------------- #
# Boundary noise of P5 / Exp 3                                                #
# --------------------------------------------------------------------------- #

def _fake_gt(n_videos=40, seed=0):
    """Videos of 60-300 s with 3-8 GT events of 3-40 s (some overlapping, some at the borders)."""
    rng = np.random.default_rng(seed)
    durations, gt = {}, {}
    for i in range(n_videos):
        v = f"vid{i:02d}"
        dur = round(float(rng.uniform(60, 300)), 2)
        lens = rng.uniform(3, 40, size=rng.integers(3, 9))
        starts = rng.uniform(0, dur - lens)
        durations[v] = dur
        gt[v] = sorted([round(float(s), 2), round(float(s + l), 2)] for s, l in zip(starts, lens))
    gt["vid00"][0] = [0.0, 12.0]                                  # an event at each border
    gt["vid00"][-1] = [round(durations["vid00"] - 5.0, 2), durations["vid00"]]
    gt["vid00"].sort()
    return durations, gt


def test_noise_name():
    assert [noise_name(t) for t in (0.9, 0.7, 0.5, 0.3)] == ["gt_noise_09", "gt_noise_07", "gt_noise_05", "gt_noise_03"]
    assert noise_name(0.5, seed=1) == "gt_noise_05_s1"


def test_perturb_bounds_and_sigma_zero():
    durations, gt = _fake_gt()
    dirs = noise_directions(gt, seed=0)
    v = "vid00"
    assert perturb(gt[v], durations[v], dirs[v], 0.0) == gt[v]                    # all events >= d_min
    for sigma in (0.3, 2.0, 16.0):
        for vid in gt:
            segs = perturb(gt[vid], durations[vid], dirs[vid], sigma)
            assert len(segs) == len(gt[vid]) and segs == sorted(segs)
            assert all(0 <= ts < te <= durations[vid] for ts, te in segs)
            assert all(te - ts >= D_MIN - 0.011 for ts, te in segs)
            assert all(ts <= durations[vid] - MIN_SEG_SEC for ts, _ in segs)
    # a 1 s event keeps d_min, and the directions are drawn once per (seed, video)
    assert perturb([[50.0, 51.0]], 100.0, np.zeros((1, 2)), 1.0) == [[49.5, 51.5]]
    assert np.array_equal(noise_directions(gt, 0)[v], dirs[v])
    assert not np.array_equal(noise_directions(gt, 1)[v], dirs[v])


def test_best_tiou():
    gt = {"a": [[0.0, 10.0], [20.0, 30.0]], "b": [[5.0, 15.0]]}
    part = {"a": [[0.0, 10.0], [25.0, 35.0]], "b": []}
    assert best_tiou(part, gt).tolist() == pytest.approx([1.0, 5 / 15, 0.0])
    assert best_tiou(gt, gt).mean() == pytest.approx(1.0)


def test_calibrate_sigma_hits_targets_and_is_monotone():
    durations, gt = _fake_gt()
    dirs = noise_directions(gt, seed=0)
    sigmas = []
    for target in (0.9, 0.7, 0.5, 0.3):
        sigma, achieved, part = calibrate_sigma(gt, durations, dirs, target)
        assert abs(achieved - target) <= NOISE_TOL
        assert best_tiou(part, gt).mean() == pytest.approx(achieved)
        sigmas.append(sigma)
    assert sigmas == sorted(sigmas) and sigmas[0] > 0


def test_run_noise_end_to_end(tmp_path):
    durations, gt = _fake_gt()
    subset = {"settings": {}, "videos": [{"video_id": v, "split": "val", "duration": d} for v, d in durations.items()]}
    (tmp_path / "subset_videos.json").write_text(json.dumps(subset))
    (tmp_path / "gt.json").write_text(json.dumps(gt))
    (tmp_path / "uni_M_gt.json").write_text(json.dumps(
        {v: [[d * i / len(gt[v]), d * (i + 1) / len(gt[v])] for i in range(len(gt[v]))] for v, d in durations.items()}))
    rows = run_noise(tmp_path, tmp_path / "subset_videos.json", seed=0)
    src = json.loads((tmp_path / "SOURCE.json").read_text())["noise"]
    assert [r["partition"] for r in src["levels"]] == ["gt_noise_09", "gt_noise_07", "gt_noise_05", "gt_noise_03"]
    assert all(abs(r["mean tIoU"] - r["target tIoU"]) <= NOISE_TOL for r in src["levels"])
    ref = {r["partition"]: r["mean tIoU"] for r in src["reference"]}
    assert ref["gt"] == pytest.approx(1.0) and 0 < ref["uni_M_gt"] < 1
    n_gt = sum(map(len, gt.values()))
    assert all(r["segments"] == n_gt for r in src["levels"]) and len(rows) == 4 + 2
    before = (tmp_path / "gt_noise_05.json").read_text()
    run_noise(tmp_path, tmp_path / "subset_videos.json", seed=0)                   # same seed, same files
    assert (tmp_path / "gt_noise_05.json").read_text() == before
    run_noise(tmp_path, tmp_path / "subset_videos.json", seed=1, targets=(0.5,))
    blob = json.loads((tmp_path / "SOURCE.json").read_text())
    assert (tmp_path / "gt_noise_05_s1.json").is_file() and "noise" in blob and "noise_s1" in blob
