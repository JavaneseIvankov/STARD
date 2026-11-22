"""STAF-Net | src/data/video_dataset.py

Owner   : Person 1 + Person 2 (shared)
Status  : PLACEHOLDER - structure only, no implementation yet.

Responsibility
--------------
- PyTorch Dataset over cached clip crops, branch-aware.
- Second Dataset variant that reads cached feature vectors for Stage II
  (fusion / ablation training on fixed features).

Contract
--------
- Clip mode yields src.contracts.ClipBatch:
    dino_frames   (B, 4, 3, 224, 224)   layout (B, T, C, H, W)
    vjepa_frames  (B, 3, 16, 384, 384)  layout (B, C, T, H, W)  <- different!
    label         (B,) float32 in {0.0, 1.0}
    video_id      tuple[str, ...]
- Feature mode yields src.contracts.FusionInput read from data/features/.
- Deterministic sampling for val/test; no random crop jitter there.

TODO
----
- [ ] Manifest CSV reader + per-split filtering
- [ ] Clip dataset with the two layout conventions kept apart
- [ ] Feature dataset (4 variants, aligned by video_id)
- [ ] Collate function preserving contract shapes
- [ ] Verify every batch with ClipBatch.validate() in a debug flag
"""

# TODO(person-1,person-2): implement datasets described above.
