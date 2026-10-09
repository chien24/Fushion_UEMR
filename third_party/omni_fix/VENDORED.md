# Code vendored from Omni-fix

These files let `omni_retrieval/encode.py` run the **same** encoding pipeline that `scripts/eval_youcookii.py` used to score the fine-tuned checkpoint, without cloning Omni-fix.

- **Source:** `im-xiaoming/Omni-fix`, branch `chien`, commit `460257f` (2026-10-09).
  - That commit only adds `.gitignore` on top of `d8444d7`, so the code here is identical to what the eval ran.
- **Copied verbatim:** every Python file below is an unmodified copy, keeping its license header.
- **Added here, not in the original:**
  - `qwenvl/__init__.py` and `qwenvl/train/__init__.py`, both empty. In the original these were namespace packages.
  - `omniretriever/__init__.py`, reduced to a docstring. The original re-exports the inference loader, which isn't copied.

| here | Omni-fix | used for |
|---|---|---|
| `qwenvl/data/data_qwen.py` | `training/qwenvl/data/data_qwen.py` | `LazySupervisedDataset`, `DataCollatorForOmniDataset`: decord frames, audio cut from the mp4, prompt, `<\|AUDIO\|>` doubling |
| `qwenvl/data/processing_qwen2_5_omni.py` | same path | `Qwen2_5OmniProcessor` |
| `qwenvl/model/qwen2_5_omni/{configuration,modeling}_qwen2_5_omni.py` | same path | `Qwen2_5OmniThinkerForConditionalGeneration`, `Qwen2_5OmniThinkerConfig`, fusion head (`classify_linear`) |
| `qwenvl/model/qwen2_5_omni/beats/{BEATs,backbone,modules}.py` | same path | BEATs audio encoder |
| `qwenvl/train/utils.py` | same path | constants (`PAD_TOKEN_ID`, ...) |
| `omniretriever/data/{__init__,media}.py` | `src/omniretriever/data/` | `fit_waveform`, `load_audio_segment` |
| `third-party-licenses/*.txt` | `training/third-party-licenses/` | Apache-2.0 licenses of Qwen2.5-Omni, Qwen2.5-VL, transformers, video-SALMONN 2 |

**Not copied:**
- training code: `train_qwen.py`, `trainer.py`, `argument.py`, `sampler.py`;
- the BEATs tokenizer and quantizer, which aren't used for encoding;
- `omniretriever.inference`, `omniretriever.cli`, the WAVE M-RoPE patch.

The weights (WAVE-7B, BEATs, LoRA adapter) are downloaded or read from Drive and are never stored in this repo.

`omni_retrieval/encode.py` puts this folder on `sys.path` (`VENDOR_DIR`), so `import qwenvl...` and `import omniretriever...` resolve here.
