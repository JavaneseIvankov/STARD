"""STARD | src/data/dataset.py

Owner   : Orang 1 (Data & Preproc Lead)
Status  : PLACEHOLDER - structure only, no implementation yet.

Responsibility
--------------
- PyTorch Dataset over raw frames (datasets/processed).
- PyTorch Dataset over cached backbone features (datasets/cached_features, .pt).
- Produce BatchInput tensors exactly as specified in src/contracts.py.
- Cached feature layout and dtype policy: docs/cache_format.md

Contract
--------
- Yields src.contracts.BatchInput.
- Consumed by scripts/train.py and scripts/evaluate.py.

TODO
----
- [ ] Raw-frame dataset variant
- [ ] Cached-feature dataset variant (fast path, low VRAM)
- [ ] Collate function preserving (B, 16, 3, 224, 224) layout
"""

# TODO(orang-1): implement datasets described above.
