"""STARD | scripts/evaluate.py

Owner   : Orang C / Shared (Integration)
Status  : PLACEHOLDER - structure only, no implementation yet.

Responsibility
--------------
- Evaluate a trained checkpoint on the held-out test split.
- Export benchmark numbers and qualitative mask outputs.

Contract
--------
- Usage: python scripts/evaluate.py --config <yaml> --ckpt outputs/checkpoints/<run>.pt
- Prints classification AUC/F1, spatial IoU, temporal F1.

TODO
----
- [ ] Load checkpoint into src/models/detector.py
- [ ] Test-set inference with metric aggregation
- [ ] Dump predictions for notebooks/02_error_analysis.ipynb
"""

# TODO(orang-c): implement evaluation script described above.
