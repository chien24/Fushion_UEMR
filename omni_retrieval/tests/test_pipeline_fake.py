"""collect + evaluate on a fake store in the exact key layout the encoders write.

Two store layouts are exercised: the chunk directory of ``omni_retrieval.encode`` and a
single ``.npz`` like ``omniretriever.cli extract`` (``<id>__av`` / ``<id>__text``).
"""

import json

import numpy as np
import pytest

from omni_retrieval.collect import MissingEmbeddings, collect_segments, collect_text
from omni_retrieval.evaluate import evaluate, sanity_check, text_to_clip
from omni_retrieval.manifest import (caption_table, segment_records, segment_table, store_keys,
                                     unique_segments)
from omni_retrieval.partitions import build_base


def _events():
    def ev(i, vid, ts, te, text):
        return {"id": f"{vid}_{i}", "video": f"{vid}.mp4", "timestamps": [ts, te], "text": text}
    return {"val": [ev(0, "vidA", 0, 10, "cut onion"), ev(1, "vidA", 12, 20, "fry onion"),
                    ev(0, "vidB", 5, 15, "boil pasta"), ev(1, "vidB", 15, 30, "drain pasta")],
            "dev": [ev(0, "vidD", 0, 8, "whisk eggs")]}


SUBSET = [{"video_id": "vidA", "split": "val", "is_query_source": True, "duration": 30.0},
          {"video_id": "vidB", "split": "val", "is_query_source": True, "duration": 40.0},
          {"video_id": "vidD", "split": "dev", "is_query_source": False, "duration": 20.0}]


def _fake_vec(key, d=8):
    rng = np.random.default_rng(abs(hash(key)) % (2**32))
    return rng.normal(size=d).astype(np.float32)


@pytest.fixture
def prepared():
    events = _events()
    parts = build_base(SUBSET, events)
    rows = segment_table(parts)
    return {"events": events, "parts": parts, "rows": rows, "segments": unique_segments(rows),
            "captions": caption_table(events, SUBSET)}


def test_tables(prepared):
    parts, segs = prepared["parts"], prepared["segments"]
    assert parts["global"]["vidB"] == [[0.0, 40.0]]
    assert parts["gt"]["vidA"] == [[0.0, 10.0], [12.0, 20.0]]
    assert parts["uni_M_gt"]["vidA"] == [[0.0, 15.0], [15.0, 30.0]]
    assert len(prepared["rows"]) == 3 + 5 + 5
    # vidD has one GT event, so its uni_M_gt [0,20] is its global [0,20]: one clip, two rows
    assert len(segs) == 12 and sorted(segs["vidD__0.00_20.00"]["refs"]) == ["vidD__global__0", "vidD__uni_M_gt__0"]
    assert [c["caption_id"] for c in prepared["captions"]] == ["vidA_0", "vidA_1", "vidB_0", "vidB_1"]
    assert prepared["captions"][0]["gt_seg_key"] == "vidA__0.00_10.00"


def test_dedup_and_todo():
    events = {"val": [{"id": "v_0", "video": "v.mp4", "timestamps": [0, 10], "text": "x"},
                      {"id": "v_1", "video": "v.mp4", "timestamps": [10, 20], "text": "y"}], "dev": []}
    subset = [{"video_id": "v", "split": "val", "is_query_source": True, "duration": 20.0}]
    rows = segment_table(build_base(subset, events))
    segs = unique_segments(rows)
    # gt == uni_M_gt here: 2 + 2 + 1 rows, 3 unique clips
    assert len(rows) == 5 and len(segs) == 3
    assert sorted(segs["v__0.00_10.00"]["refs"]) == ["v__gt__0", "v__uni_M_gt__0"]
    todo = segment_records(segs, skip={"v__0.00_10.00"})
    assert [r["id"] for r in todo] == ["v__0.00_20.00", "v__10.00_20.00"]
    assert todo[0]["conversations"] == [{"from": "human", "value": "<video>\nPlease describe the video."}]
    assert "audio" not in todo[0]


def _write_store(prepared, tmp_path, layout):
    av = {f"{k}__av": _fake_vec(k) for k in prepared["segments"]}
    # make each caption's text vector close to its own GT clip, so text->clip is perfect
    text = {f"{c['caption_id']}__text": _fake_vec(c["gt_seg_key"]) + 0.01 * _fake_vec(c["caption_id"])
            for c in prepared["captions"]}
    if layout == "cli":
        np.savez(tmp_path / "av.npz", **av)
        np.savez(tmp_path / "text.npz", **text)
        return tmp_path / "av.npz", tmp_path / "text.npz"
    (tmp_path / "av").mkdir()
    (tmp_path / "text").mkdir()
    keys = list(av)
    np.savez(tmp_path / "av" / "chunk_a.npz", **{k: av[k] for k in keys[:4]})
    np.savez(tmp_path / "av" / "chunk_b.npz", **{k: av[k] for k in keys[4:]})
    np.savez(tmp_path / "text" / "chunk_a.npz", **text)
    return tmp_path / "av", tmp_path / "text"


@pytest.mark.parametrize("layout", ["cli", "chunks"])
def test_collect_and_evaluate(prepared, tmp_path, layout):
    av_store, text_store = _write_store(prepared, tmp_path, layout)
    assert store_keys(av_store, "av") == set(prepared["segments"])
    out = tmp_path / "cache"
    s = collect_segments(prepared["rows"], av_store, out / "segments.npz")
    t = collect_text(prepared["captions"], text_store, out / "text.npz")
    assert s == {"rows": 13, "unique": 12, "missing": 0} and t["captions"] == 4

    with np.load(out / "segments.npz") as blob:
        assert blob["emb"].dtype == np.float16
        np.testing.assert_allclose(np.linalg.norm(blob["emb"].astype(np.float32), axis=1), 1, atol=1e-3)
        assert set(blob["partition"]) == {"global", "gt", "uni_M_gt"}

    clip = text_to_clip(out)
    assert clip["eval"]["R@1"] == 100 and clip["subset"]["R@1"] == 100
    assert clip["subset"]["gallery"] == 5      # 4 val + 1 dev GT clip

    rows = {r["partition"]: r for r in evaluate(out)}
    assert set(rows) == {"global", "event_single", "uni_M_gt", "gt"}
    assert rows["gt"]["R@1"] == 100 and rows["gt"]["mom@.5|top1"] == 100
    assert rows["gt"]["videos"] == 3 and rows["gt"]["coverage"] == 100


def test_sanity_reference(prepared, tmp_path):
    av_store, text_store = _write_store(prepared, tmp_path, "chunks")
    out = tmp_path / "cache"
    collect_segments(prepared["rows"], av_store, out / "segments.npz")
    collect_text(prepared["captions"], text_store, out / "text.npz")
    ref = tmp_path / "best_val.json"
    ref.write_text(json.dumps({"text_to_multimodal (t2m)": {"R@1": 100.0}, "num_samples": 4}))
    assert sanity_check(out, ref)["gap_R@1"] == 0
    ref.write_text(json.dumps({"text_to_multimodal (t2m)": {"R@1": 40.0}, "num_samples": 4}))
    with pytest.raises(RuntimeError, match="SANITY CHECK FAILED"):
        sanity_check(out, ref)


def test_missing_ids_are_reported(prepared, tmp_path):
    av_store, _ = _write_store(prepared, tmp_path, "chunks")
    (av_store / "chunk_b.npz").unlink()
    with pytest.raises(MissingEmbeddings):
        collect_segments(prepared["rows"], av_store, tmp_path / "segments.npz")
    s = collect_segments(prepared["rows"], av_store, tmp_path / "segments.npz", allow_missing=True)
    assert s["missing"] > 0
