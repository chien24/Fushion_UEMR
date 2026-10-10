"""UniAV/Athena application pipeline for UEMR (inference only, no training).

video -> InternVideo2 + BEATs features (pluggable source) -> Athena proposals -> UEMR post-processing
-> captions -> comparison with YouCook2 GT -> metrics and error analysis.

Independent of ``omni_retrieval`` (nothing is imported from it). The Athena code is vendored
unmodified in ``third_party/athena`` and imported through ``uniav_app.athena_api``.
"""
