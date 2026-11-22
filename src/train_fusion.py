"""STAF-Net | src/train_fusion.py

Owner   : Person 3 (Fusion / evaluation)
Status  : PLACEHOLDER - structure only, no implementation yet.

Responsibility
--------------
- Stage II training on cached features: single-branch baselines and fusion.
- Run the full ablation matrix A1-A8 from one entry point.

Contract
--------
- Usage: python -m src.train_fusion --config configs/fusion.yaml --experiment A6
- Reads data/features/ variants; no GPU-heavy backbone work in this stage
- Writes checkpoints/fusion_<experiment>.pt and results/metrics/<experiment>.json
- AdamW lr 1e-3, batch 64, max 30 epochs, early stopping 5 on validation AUROC

TODO
----
- [ ] Experiment registry A1..A8 -> (variants, fusion mode)
- [ ] Feature dataloader keyed by video_id
- [ ] Gated and concat heads via src/models/stafnet.py
- [ ] Early stopping on validation AUROC, single final test evaluation
- [ ] Per-experiment metrics + gate statistics export
"""

# TODO(person-3): implement the fusion training entry point described above.
