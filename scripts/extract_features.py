"""STARD | scripts/extract_features.py

Owner   : Orang A + Orang B (Shared)
Status  : PLACEHOLDER - structure only, no implementation yet.

Responsibility
--------------
- One-off offline caching of frozen backbone embeddings to disk.
- Avoids re-running DINOv3 / V-JEPA every epoch: large VRAM and speed win.

Contract
--------
- Usage: python scripts/extract_features.py --config configs/base.yaml
- Writes .pt files into datasets/cached_features/.
- Consumed by the cached-feature variant of src/data/dataset.py.

TODO
----
- [ ] Load backbone wrappers from src/models
- [ ] Batch inference over dataloader under torch.no_grad
- [ ] Save per-video feature tensors + label metadata
- [ ] Resume-safe shard writing
"""

# TODO(orang-a,orang-b): implement feature caching script described above.
