"""STAF-Net | src/extract_features.py

Owner   : Person 1 + Person 2 (shared)
Status  : PLACEHOLDER - structure only, no implementation yet.

Responsibility
--------------
- Run frozen or LoRA-adapted branches over the manifest once and cache the
  pooled per-video embeddings for Stage II.

Contract
--------
- Usage: python -m src.extract_features --config configs/dataset.yaml \\
           --variant dino_frozen --checkpoint checkpoints/dino_frozen.pt
- Variants: dino_frozen, dino_lora, vjepa_frozen, vjepa_lora
- Writes data/features/<variant>/<split>/shard-XXXXX.pt + index.csv
- Schema and invariants: docs/cache_format.md
- Always torch.no_grad() with model.eval(); resume-safe shard writing

TODO
----
- [ ] Variant + checkpoint resolution
- [ ] Batch inference over the manifest with deterministic sampling
- [ ] Shard writer with format_version + checkpoint_id metadata
- [ ] MANIFEST.json per cache run (backbone, checkpoint, preprocessing_version)
- [ ] Verify feature dims against src/contracts.py before writing
"""

# TODO(person-1,person-2): implement feature extraction described above.
