"""Grid <-> seconds (t = (g + 0.5) * n / 256) and level times; same as athena FeatureSpec.to_seconds."""

import numpy as np
import pytest

from uniav_app.timegrid import grid_to_sec, level_times, sec_to_grid


def test_grid_to_sec_by_hand():
    assert grid_to_sec(0, 300) == pytest.approx(0.5 * 300 / 256)
    assert grid_to_sec(255.5, 300) == pytest.approx(300.0)
    assert grid_to_sec(255.5, 300, duration=299) == pytest.approx(299.0)
    assert grid_to_sec(-3, 300, duration=299) == 0.0
    np.testing.assert_allclose(grid_to_sec([[0, 10]], 256), [[0.5, 10.5]])


def test_roundtrip():
    g = np.array([[0.0, 12.3], [100.25, 255.0]])
    np.testing.assert_allclose(sec_to_grid(grid_to_sec(g, 417), 417), g)
    assert sec_to_grid(0.5 * 300 / 256, 300) == pytest.approx(0.0)


def test_level_times():
    t0 = level_times(256, 0)
    assert t0.shape == (256,) and t0[0] == pytest.approx(0.5) and t0[1] == pytest.approx(1.5)
    t1 = level_times(256, 1)
    assert t1.shape == (128,) and t1[0] == pytest.approx(0.5) and t1[1] == pytest.approx(2.5)
    assert level_times(512, 0)[0] == pytest.approx(1.0)     # 512 s video: 2 s per step


def test_matches_athena_feature_spec():
    pytest.importorskip("torch")
    from uniav_app.athena_api import import_athena
    import_athena()
    from athena.features import FeatureSpec
    spec = FeatureSpec({"feat_source": "iv2", "max_seq_len": 256, "default_fps": 16, "iv2_video_keys": ["v768"],
                        "iv2_l2norm": True, "iv2_rows_per_sec": 1})
    segs = np.array([[0.0, 3.0], [17.2, 80.9], [250.0, 255.9]], dtype=np.float32)
    for n, dur in ((60, 59.5), (317, 316.2), (1100, 1101.0)):
        np.testing.assert_allclose(grid_to_sec(segs, n, dur), spec.to_seconds(segs, n, dur), rtol=1e-6, atol=1e-5)
