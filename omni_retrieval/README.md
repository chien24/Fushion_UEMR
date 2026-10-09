# omni_retrieval

Multi-vector text→video retrieval over YouCook2 with OmniRetriever-7B. This covers UEMR P3 + P4 and columns (a)/(b) of Exp 1. Athena proposals are not in yet; they get added later as one more partition.

Everything runs on **Colab** through `notebooks/omni_retrieval_colab.ipynb`. Nothing here is meant to be run on the local machine.

## Running on Colab

No training happens anywhere here: the fine-tuned LoRA checkpoint is only loaded and run forward.

**No Omni-fix clone is needed.** The model, processor and event-cutting code that encoding needs are copied verbatim into `third_party/omni_fix/` (see `VENDORED.md` there). WAVE-7B and BEATs are downloaded from HuggingFace to `/content/WAVE_HOME`. The `best` checkpoint is copied from Drive to `/content/my_checkpoint` and checked for its fusion heads. Libraries come from this repo's `requirements.txt` (`transformers==4.51.3`, the environment the checkpoint was evaluated in).

1. Cells 1–5 set things up: clone `chien24/Fushion_UEMR`, install requirements, download WAVE-7B and copy the checkpoint, check the heads, configure.
2. Cell 6 runs the unit tests (no GPU).
3. With `SMOKE = True`, run cells 7–12. This encodes the GT events of 5 videos for real and evaluates them.
4. Set `SMOKE = False`, run cell 5, then cells 7–13. Cell 13 evaluates the hand-written queries.
5. Cells 14–15 free memory, then let you type queries.

If the session drops, run cells 1–5, then 8, then the encode cell again. Only the missing events get encoded.

## Queries and GT

- **GT queries:** the 3,030 YouCook2 val captions. Each one's GT is its own event.
- **Hand-written queries:** `queries/custom_queries.jsonl`, one per line, in the form `{"query", "video_id", "ts", "te"}` with optional extra fields such as `source_caption`.
  - The `[ts, te]` GT window is mapped to the database event of the same video with the largest tIoU (at least 0.5). It doesn't have to match the YouCook2 segment exactly.
  - Vectors are cached by query text, so editing a line re-encodes only that line.
  - Queries whose video isn't in the database are listed as skipped.
  - The shipped file has 26 paraphrases of val events, deliberately worded differently from the captions.
- **Typed queries:** cell 15 accepts `query || video_id || ts || te`. With the GT part, the correct event in the result table is marked ✓.

## Event retrieval (what the notebook does now)

YouCook2 videos hold many events while Omni encodes one clip, so the unit stored and returned is **one GT event**:

- **Build** (once, GPU, `base_partitions=("gt",)`): every GT event of the ~500 gallery videos is encoded as a `[ts, te]` clip, giving `events.npz` (`seg_key`, `video_id`, `ts`, `te`, `caption`, `split`, `emb`). The val captions are encoded into `text.npz` as the evaluation queries.
- **Evaluate** (`evaluate.evaluate_events`): each val caption searches all events.
  - `event R@k` counts a hit only on the caption's own event (right video and right window).
  - `video@k` counts any event of the right video.
  - `tIoU.5@1` counts a top-1 in the right video that overlaps the GT event by at least 0.5.
  - The per-query top-1 answers are saved to `per_query.csv`.
- **Query** (`search.EventIndex`): `EventIndex.load(db).search(q_vec, top_k)` returns `[{rank, score, video_id, ts, te, caption, seg_key}]`. A free-text query is encoded with `encode.encode_texts` after a single `encode.load_model(base, beats, adapter)`. That step needs the model, using the vendored code. The search itself is plain numpy.
- **Hand-written queries** (`queries.evaluate_custom`): same metrics, computed by the shared `evaluate.score_event_queries`.

## Why not `omniretriever.cli extract`

The fine-tuned adapter was scored with `scripts/eval_youcookii.py`. That script runs the training pipeline (`LazySupervisedDataset`), which does the following:
- decodes with decord;
- takes 8 frames, resized to 50176 px with the aspect ratio kept;
- uses the chat-template prompt;
- doubles the `<|AUDIO|>` tokens;
- cuts audio from the mp4, then center-crops or pads it to 8 s.

The CLI uses PyAV with a square center-crop and WAVE's 336 px floor, so its vectors are not the ones the eval measured.

`encode.py` uses the vendored dataset and collator (`third_party/omni_fix`, an unmodified copy). `EvalDataArgs` is a field-for-field copy of the eval's `MockDataArgs`, and the model is loaded the same way the eval loads it:
- **av:** a record with no caption turn, giving `mllm_embeds` (the all-layer fusion head).
- **text:** `caption + <|im_end|>` goes through the thinker's text model, and the embedding is the last token of the last layer. This is the eval's label branch, which has **no** `classify_linear`. It is not `OmniRetriever.encode_text`, whose vector lives in a different space.
- **Error handling:** a record that fails to load is written to `failed.jsonl`. Normally the dataset would silently swap it for a random sample.

Cell 12's sanity check confirms all this: text→clip on `gt` with the eval's gallery must reproduce the eval's t2m R@1 within 1 point, otherwise the run stops.

Note: every window, including `global`, gets **8 frames, with audio center-cropped to 8 s**. For `global`, a video several minutes long is seen as just 8 frames plus 8 s of audio.

## Partitions and adding a new one

| name | segments |
|---|---|
| `global` | `[0, duration]` (R0) |
| `gt` | YouCook2 GT events (R7, oracle) |
| `uni_M_gt` | M equal segments, M = number of GT events (count control) |
| `event_single` | not encoded: L2-normalized mean of the `gt` vectors (R1) |

Durations come from the mp4 header (the shorter of the video and audio streams). Every segment is clamped to `[0, duration]`. A segment is dropped if it is empty or starts less than 1 s before the end of the file, the same rule `convert_youcookii.py` uses.

To add a partition, for example Athena proposals (`uniav_pred`), `uni_M`, `rand_M` or `kmedoids_M`:

1. Write a JSON file `{"<video_id>": [[ts, te], ...], ...}`. Extra fields after `te`, such as a confidence, are ignored.
2. In cell 5: `EXTRA_PARTITIONS = {'uniav_pred': '/content/drive/MyDrive/uemr/partitions/uniav_pred.json'}`.
3. Run cells 11, 13, 15 and 16. Only segments that aren't in the cache yet get encoded. Dedup works on `(video, ts, te)` rounded to 0.01 s, so a proposal identical to a GT event or another partition's segment is reused.

## Cache format (`/content/drive/MyDrive/uemr/omni_cache/`)

```
common/subset_videos[_smoke].json   {settings, videos: [{video_id, split, is_query_source, duration}]}
common/partitions/<name>.json       video_id -> [[ts, te], ...]
common/segs_todo_<tag>.jsonl        clips not yet encoded (id = seg_key "<vid>__<ts>_<te>", refs)
common/caps_todo_<tag>.jsonl        captions not yet encoded
<tag>/raw/av/chunk_*.npz            THE cache: "<seg_key>__av" -> fp32 (L2), one per unique clip
<tag>/raw/av/failed.jsonl           clips that failed to load
<tag>/raw/av/encoder_meta.json      adapter, Omni-fix commit, preprocessing (read from MockDataArgs)
<tag>/raw/av/runs.jsonl             time / clip of each encode run
<tag>/raw/text/chunk_*.npz          "<caption_id>__text" -> fp32 (L2)
<tag>/segments.npz                  keys "<vid>__<partition>__<k>", seg_key, video_id, partition, ts, te, emb fp16 (L2)
<tag>/text.npz                      caption_id, video_id, sentence, gt_ts, gt_te, gt_seg_key, emb fp16 (L2)
<tag>/meta.json                     adapter (+ SOURCE.json), preprocessing, commit, videos, counts, times
<tag>/results_<tag>.csv|json        evaluation table + sanity check
```

A smoke run writes its `segments.npz`, `text.npz` and results to `<tag>/smoke/`, but shares `raw/` with the full run.

## API

```python
from omni_retrieval.search import SegmentIndex
index = SegmentIndex.load(cache_dir, partition="gt")       # or "global", "uni_M_gt", "event_single", ...
hits  = index.search(q_vec, top_k=10, agg="max")           # agg: max | topk_mean | lse
# -> [{video_id, score, best_segment: (ts, te), segment_scores: [(ts, te, s), ...]}, ...]
```

The video score is `S(q, V) = max_i cos(q, e_i)` (MaxSim), and the timestamp is the segment at the argmax. The computation is one `Q × E^T` product, gathered into a `[Q, V, K_max]` array and reduced along the last axis. It runs in blocks of 512 queries, with no loop over videos.

`evaluate.py` reports R@1/5/10, MedR and MnR per partition. It also reports `mom@.5|top1`: among queries whose top-1 video is correct, the share where the argmax segment has tIoU ≥ 0.5 with the GT event. `VCMR R@1 (tIoU.5)` is the same check measured over all queries.
