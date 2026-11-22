"""STARD | src/utils/losses.py

Owner   : Orang C / Shared (Integration)
Status  : PLACEHOLDER - structure only, no implementation yet.

Responsibility
--------------
- Combined multi-task loss: CrossEntropy(cls) + BCE(spatial) + BCE(temporal).
- Configurable loss weights per task for ablation studies.

Contract
--------
- Consumes src.contracts.ModelOutputs and src.contracts.BatchInput labels.
- Returns total loss plus per-task components for logging.

TODO
----
- [ ] Weighted multi-task loss class
- [ ] Per-task weight schedule from YAML config
- [ ] Handle missing/absent mask annotations
"""

# TODO(orang-c): implement MultiTaskLoss described above.
