"""Vendored subset of OmniRetriever (Omni-fix ``src/omniretriever``).

Only ``omniretriever.data.media`` is copied -- the training pipeline in ``qwenvl.data.data_qwen``
imports ``fit_waveform`` / ``load_audio_segment`` from it. The original package ``__init__``
re-exports the inference loader, which is not vendored, so this one is intentionally empty.
"""
