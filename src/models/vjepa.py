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
- Input  : frames (B, 16, 3, 224, 224) float [0.0, 1.0], NOT normalized
- Output : temporal (B, 16, 196, 768) float32 - V-JEPA 2.1 ViT-B/16 (hidden_size=768)
- Config : configs/vjepa_baseline.yaml
- Spec   : src.contracts.VJEPA21_VITB16
- Note   : tubelet_size=2, so 16 frames -> T_out=8; wrapper MUST interpolate to 16
- Note   : checkpoint declares crop_size=384, frames_per_clip=64; 224/16 frames is
           only legal because interpolate_rope=true - verify, then record it

TODO
----
- [ ] Load pretrained V-JEPA 2.1 ViT-B (verify hub URL / weights name)
- [ ] Freeze backbone parameters
- [ ] Permute to (B, C, T, H, W) for 3D conv / video transformer
- [ ] Temporal alignment: interpolate T_out=8 back to 16 frames
- [ ] Verify RoPE interpolation from crop 384 to 224
"""

# TODO(orang-b): implement VJEPATemporalExtractor described above.
