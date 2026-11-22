"""STAF-Net | src/train_dino.py

Owner   : Person 1 (Spatial branch)
Status  : PLACEHOLDER - structure only, no implementation yet.

Responsibility
--------------
- Stage I supervised training of the DINOv3 branch: frozen baseline and LoRA run.

Contract
--------
- Usage: python -m src.train_dino --config configs/dino_lora.yaml [--frozen]
- Writes checkpoints/dino_{frozen,lora}.pt
- Writes features/dino_{frozen,lora} cache entries (docs/cache_format.md)
- Logs validation AUROC per epoch; selection on best validation AUROC

TODO
----
- [ ] Config loader + CLI overrides
- [ ] Dataloader from src/data/video_dataset.py (clip mode)
- [ ] Frozen run: head only, backbone eval
- [ ] LoRA run: adapters + head trainable via src/utils/lora.py
- [ ] BF16 autocast, AdamW with separate LoRA/head learning rates
- [ ] Export embeddings for the feature cache
- [ ] Record runtime and peak VRAM per epoch
"""

# TODO(person-1): implement the DINOv3 training entry point described above.
