"""STARD | src/models/vjepa.py

Owner   : Orang B (Temporal Model Lead)
Status  : PLACEHOLDER - structure only, no implementation yet.

Responsibility
--------------
- Wrapper around V-JEPA 2.1 ViT-B backbone, weights frozen.
- Video embedding / motion feature extraction across the 16-frame clip.
- Unimodal temporal baseline (V-JEPA + multi-task heads).

Contract
--------
- Input  : frames (B, 16, 3, 224, 224) float [0.0, 1.0]
- Output : temporal (B, 16, P, D_temporal) e.g. (B, 16, 196, 768)
- Config : configs/vjepa_baseline.yaml

TODO
----
- [ ] Load pretrained V-JEPA 2.1 ViT-B (verify hub URL / weights name)
- [ ] Freeze backbone parameters
- [ ] Permute to (B, C, T, H, W) for 3D conv / video transformer
- [ ] Temporal alignment: interpolate T_out back to 16 frames
"""

# TODO(orang-b): implement VJEPATemporalExtractor described above.
