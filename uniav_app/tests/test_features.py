"""Feature-source bookkeeping on fake npz files (samples source, no model, no network)."""

import numpy as np

from uniav_app.config import Config
from uniav_app.features import check_n_vs_duration, get_source, save_feature_npz
from uniav_app.features.base import feature_rows
from uniav_app.io_utils import read_jsonl


def fake_npz(path, n_v, n_a, duration=None):
    rng = np.random.default_rng(0)
    kw = {"v768": rng.standard_normal((n_v, 768)), "v512": rng.standard_normal((n_v, 512)),
          "a768": rng.standard_normal((n_a, 768))}
    if duration is None:
        save_feature_npz(path, **kw)
    else:
        np.savez(path, duration=duration, **{k: v.astype(np.float16) for k, v in kw.items()})


def test_save_and_rows(tmp_path):
    p = tmp_path / "v.npz"
    fake_npz(p, 12, 11)
    assert feature_rows(p) == 11
    with np.load(p) as z:
        assert z["v768"].dtype == np.float16 and set(z.files) == {"v768", "v512", "a768"}


def test_samples_source_resumes_and_logs_failures(tmp_path):
    sdir = tmp_path / "samples"
    sdir.mkdir()
    fake_npz(sdir / "aaa.npz", 30, 30, duration=29.5)
    fake_npz(sdir / "bbb.npz", 20, 19, duration=25.0)
    cfg = Config(cache_root=str(tmp_path / "cache"), samples_dir=str(sdir), smoke=True)
    src = get_source(cfg)
    assert src.name == "samples"
    recs = src.prepare(["aaa", "bbb", "zzz"], {"aaa": 29.0})
    assert set(recs) == {"aaa", "bbb"}
    assert recs["aaa"].n == 30 and recs["aaa"].duration == 29.0            # gallery duration wins
    assert recs["bbb"].n == 19 and recs["bbb"].duration == 25.0            # else the sample's own
    assert [r["video_id"] for r in read_jsonl(src.failed_path)] == ["zzz"]
    src.prepare(["aaa", "bbb"])                                            # nothing redone
    assert len(read_jsonl(src.meta_path)) == 2
    bad = check_n_vs_duration(recs, 2.0)
    assert [b["video_id"] for b in bad] == ["bbb"]                         # |19 - 25| > 2


def test_paths_by_source(tmp_path):
    cfg = Config(cache_root=str(tmp_path), feature_source="hf")
    assert cfg.out_dir == tmp_path / "ft" and cfg.feats_dir == tmp_path / "feats" / "hf"
    cfg.feature_source = "extract"
    assert cfg.out_dir == tmp_path / "ft" / "from_extract" and cfg.z_dir == tmp_path / "ft" / "from_extract" / "Z"
    cfg.smoke = True
    assert cfg.source == "samples" and cfg.out_dir == tmp_path / "ft" / "smoke"
