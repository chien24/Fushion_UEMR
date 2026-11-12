"""Benchmark gallery (docs/uemr_chot/UEMR_CHOT_03 §1, §3). CPU, synthetic data."""

from uemr_htstep import bench, data

DB = {"a": {"subset": "validation", "duration": 100.0, "annotations": [{"segment": [10, 20], "sentence": "Add salt."},
                                                                        {"segment": [90, 130], "sentence": "stir well"},
                                                                        {"segment": [99.5, 120], "sentence": "serve"}]},
      "b": {"subset": "validation", "duration": 50.0, "annotations": [{"segment": [5, 9], "sentence": "add  SALT"}]},
      "c": {"subset": "validation", "duration": 60.0, "annotations": [{"segment": [1, 2], "sentence": "boil"}]},
      "t": {"subset": "training", "duration": 60.0, "annotations": []}}


def test_funnel_and_overlap():
    b = bench.yc2(DB)
    assert sorted(b) == ["a", "b", "c"]
    assert bench.overlap(b, uniav_train=["c", "x"], omni_train=["y"], dev=[]) == {"uniav_train": ["c"], "omni_train": [], "dev": []}
    keep, counts = bench.funnel(b, have_mp4=["a", "b", "c"], with_audio=["a", "b", "c"], readable=["a", "b", "c"], in_training=["c"])
    assert keep == ["a", "b"]
    assert counts == {"annotation": 3, "tải được": 3, "có tiếng": 3, "đọc được (decord)": 3, "không trùng train / dev": 2}


def test_queries_are_the_step_sentences_cut_to_the_mp4():
    b = bench.yc2(DB)
    caps, dropped = bench.queries(b, ["a", "b"], {"a": 99.5, "b": 50.0})      # the mp4 of a is shorter than the annotation says
    assert [c["caption_id"] for c in caps] == ["a_0", "a_1", "b_0"] and dropped == ["a_2"]
    assert caps[0]["sentence"] == "Add salt." and caps[0]["headline"] == caps[2]["headline"] == "add salt"     # verbatim; shared text
    assert (caps[1]["gt_ts"], caps[1]["gt_te"]) == (90.0, 99.5) and caps[1]["gt_seg_key"] == "a__90.00_99.50"
    assert data.gt_partition(caps) == {"a": [[10.0, 20.0], [90.0, 99.5]], "b": [[5.0, 9.0]]}                    # R7
    assert bench.r0({"a": 99.5}) == {"a": [[0.0, 99.5]]}
    assert data.headline_stats(caps)["% queries whose text is in >= 2 videos"] > 60
