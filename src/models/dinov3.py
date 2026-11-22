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
- Input  : frames (B, 16, 3, 224, 224) float [0.0, 1.0], NOT normalized
- Output : spatial (B, 16, 196, 384) float32 - DINOv3 ViT-S/16 (num_features=384)
- Config : configs/dinov3_baseline.yaml
- Spec   : src.contracts.DINOV3_VITS16, patch16 @224 -> 14x14 = 196 tokens
- Note   : pretrained crop is 256 with fixed_input_size=false, so 224 is legal

TODO
----
- [ ] Load pretrained DINOv3 ViT-B (verify hub URL / weights name)
- [ ] Freeze backbone parameters
- [ ] Apply ImageNet normalization before forward
- [ ] Reshape patch tokens back to (B, 16, 196, 384)
- [ ] Run scripts/smoke_dinov3.py until it exits 0
"""

# TODO(orang-a): implement DINOv3SpatialExtractor described above.
