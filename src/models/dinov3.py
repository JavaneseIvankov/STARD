"""STARD | src/models/dinov3.py

Owner   : Orang A (Spatial Model Lead)
Status  : PLACEHOLDER - structure only, no implementation yet.

Responsibility
--------------
- Wrapper around DINOv3 ViT-B backbone, weights frozen.
- Per-frame patch token extraction: (B*16, 3, 224, 224) -> patch tokens.
- Unimodal spatial baseline (DINOv3 + multi-task heads).

Contract
--------
- Input  : frames (B, 16, 3, 224, 224) float [0.0, 1.0]
- Output : spatial (B, 16, P, D_spatial) e.g. (B, 16, 196, 768)
- Config : configs/dinov3_baseline.yaml

TODO
----
- [ ] Load pretrained DINOv3 ViT-B (verify hub URL / weights name)
- [ ] Freeze backbone parameters
- [ ] Apply ImageNet normalization before forward
- [ ] Reshape patch tokens back to (B, 16, P, D_spatial)
"""

# TODO(orang-a): implement DINOv3SpatialExtractor described above.
