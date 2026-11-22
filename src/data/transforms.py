"""STAF-Net | src/data/transforms.py

Owner   : Person 1 + Person 2 (shared)
Status  : PLACEHOLDER - structure only, no implementation yet.

Responsibility
--------------
- Two distinct preprocessing pipelines, one per backbone, using each model's
  official normalization constants.
- DINOv3: 224x224, ImageNet mean/std (0.485/0.456/0.406, 0.229/0.224/0.225).
- V-JEPA 2.1: 384x384, normalization from the released checkpoint config.
- Train-only mild appearance augmentation (brightness, compression, small blur);
  geometry applied identically across all frames of one clip.

Contract
--------
- Outputs stay float32 in [0, 1] before normalization; normalization happens
  inside the branch wrapper, not here.
- No mask-aware augmentation: DFDC ships video-level labels only.

TODO
----
- [ ] dino_transform(train: bool) -> frames (4, 3, 224, 224)
- [ ] vjepa_transform(train: bool) -> frames (3, 16, 384, 384)
- [ ] Clip-consistent geometric augmentation
- [ ] Verify V-JEPA normalization constants against the checkpoint config
"""

# TODO(person-1,person-2): implement per-branch transforms described above.
