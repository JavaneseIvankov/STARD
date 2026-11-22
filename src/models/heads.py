"""STARD | src/models/heads.py

Owner   : Orang C / Shared (Integration)
Status  : PLACEHOLDER - structure only, no implementation yet.

Responsibility
--------------
- Classification head: 3-way logits (Real / Deepfake / Physical Spoof).
- Temporal mask head: per-frame manipulation logits (B, 16).
- Spatial mask head: patch grid (14x14) upsampled to (224, 224) per frame.

Contract
--------
- Input  : fused tokens (B, 16, P, d_hidden)
- Output : src.contracts.ModelOutputs
- logits_cls (B, 3) | mask_spatial (B, 16, 224, 224) | mask_temporal (B, 16)

TODO
----
- [ ] Classification head over globally pooled tokens
- [ ] Temporal head over patch-pooled per-frame tokens
- [ ] Spatial head with ConvTranspose2d upsampling 14 -> 224
- [ ] Return ModelOutputs dataclass
"""

# TODO(orang-c): implement MultiTaskHeads described above.
