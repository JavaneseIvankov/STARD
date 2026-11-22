"""STAF-Net | src/train_vjepa.py

Owner   : Person 2 (Video branch)
Status  : PLACEHOLDER - structure only, no implementation yet.

Responsibility
--------------
- Stage I supervised training of the V-JEPA 2.1 branch: frozen baseline and LoRA run.

Contract
--------
- Usage: python -m src.train_vjepa --config configs/vjepa_lora.yaml [--frozen]
- Writes checkpoints/vjepa_{frozen,lora}.pt
- Writes features/vjepa_{frozen,lora} cache entries (docs/cache_format.md)
- Logs validation AUROC per epoch; selection on best validation AUROC

TODO
----
- [ ] Config loader + CLI overrides
- [ ] Dataloader from src/data/video_dataset.py (clip mode, (B, C, T, H, W))
- [ ] Frozen run: head only, backbone eval
- [ ] LoRA run: adapters + head trainable via src/utils/lora.py
- [ ] BF16, gradient accumulation 8-16, activation checkpointing
- [ ] Export embeddings for the feature cache
- [ ] Benchmark step time and peak VRAM, then publish the real budget
"""

# TODO(person-2): implement the V-JEPA training entry point described above.
