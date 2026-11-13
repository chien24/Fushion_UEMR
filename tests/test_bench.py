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
    assert sorted(bench.yc2(DB, None)) == ["a", "b", "c", "t"]
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


def test_final_bench_rows_score_and_criteria(tmp_path):
    """The locked config on a synthetic gallery; video ``c`` has no proposal above θ (no R6 / R3, R8 = fill windows)."""
    import json

    import numpy as np

    from omni_retrieval.manifest import seg_key
    from uemr_htstep import final_bench as fb

    cfg = json.load(open("configs/uemr_final.json", encoding="utf-8"))
    dur = {"a": 100.0, "b": 50.0, "c": 60.0}
    b = bench.yc2(DB)
    caps, _ = bench.queries(b, sorted(dur), dur)
    props = {"a": [{"ts": 10.0, "te": 20.0, "conf": 0.9, "step": 1}, {"ts": 60.0, "te": 70.0, "conf": 0.1, "step": 2}],
             "b": [{"ts": 5.0, "te": 9.0, "conf": 0.8, "step": 1}], "c": []}
    rows, info = fb.build_rows(props, dur, data.gt_partition(caps), cfg)
    assert rows["r6"]["a"] == [[10.0, 20.0]] and rows["r6"]["c"] == [] == rows["r3"]["c"] and info["videos_without_r6"] == ["c"]
    assert rows["r8"]["c"] and [10.0, 20.0] in rows["r8"]["a"] and abs(info["clips"]["r2"] - info["clips"]["r8"]) <= 2
    ann = fb.uniav_annotations(dur, 7)
    assert ann["a"]["subset"] == "val_seen" and ann["a"]["annotations"][0]["id"] == 7
    q = fb.quality(rows, data.gt_partition(caps), dur, {"a": 3, "b": 3, "c": 5})
    assert q["video không có đoạn R6"] == 1 and q["article đoán nhiều nhất"][0] == ["3", 2]

    rng = np.random.default_rng(0)
    unit = lambda n: (lambda x: x / np.linalg.norm(x, axis=1, keepdims=True))(rng.normal(size=(n, 8)).astype(np.float32))
    keys = sorted({seg_key(v, s[0], s[1]) for part in rows.values() for v in part for s in part[v]})
    omni = dict(zip(keys, unit(len(keys))))
    for n, part in rows.items():
        ids = [(v, s[0], s[1]) for v in sorted(part) for s in part[v]]
        np.savez(tmp_path / f"student_{n}.npz", emb=unit(len(ids)), video_id=np.array([x[0] for x in ids]),
                 ts=np.array([x[1] for x in ids], np.float32), te=np.array([x[2] for x in ids], np.float32))
    text = {"caption_id": np.array([c["caption_id"] for c in caps]), "video_id": np.array([c["video_id"] for c in caps]),
            "gt_ts": np.array([c["gt_ts"] for c in caps], np.float32), "gt_te": np.array([c["gt_te"] for c in caps], np.float32),
            "emb": unit(len(caps))}
    crit = fb.score(caps, text, rows, omni, str(tmp_path), cfg, str(tmp_path / "out"), pt={"text": text, "omni": omni})
    assert crit["UEMR"] == "UEMR-Full" and set(crit) >= {"K1", "K2", "K3", "K4", "K5", "kết luận"}
    assert crit["K6 (chỉ báo)"]["pt - best R8 (b)"]["A - B"] == 0          # same vectors as best
    m = json.load(open(tmp_path / "out" / "metrics.json", encoding="utf-8"))
    r6 = next(t for t in m["table"] if t["dòng"] == "R6" and t["cột"] == "b" and t["mode"] == "strict")
    assert r6["video có vector"] == 2 and m["missing_omni"]["r8"] == 0      # c is not a candidate of R6
    assert {t["cột"] for t in m["table"]} == {"b", "S", "U1", "C1", "b_pt"}
