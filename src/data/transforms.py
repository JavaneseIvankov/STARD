"""STARD | src/data/transforms.py

Owner   : Orang 1 (Data & Preproc Lead)
Status  : PLACEHOLDER - structure only, no implementation yet.

Responsibility
--------------
- Image and video-level augmentation pipelines.
- Train-time augmentation vs deterministic val/test transform.

Contract
--------
- Must keep output shape (16, 3, 224, 224) and float range [0.0, 1.0].
- Backbone-specific normalization handled inside model wrappers.

TODO
----
- [ ] Spatial augmentations (flip, crop jitter, color jitter)
- [ ] Temporal augmentations (frame dropout, temporal jitter)
- [ ] Mask-aware augmentation for label_spatial / label_temporal
"""

# TODO(orang-1): implement transforms described above.
