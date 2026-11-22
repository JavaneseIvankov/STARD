"""STAF-Net | src/data/build_manifest.py

Owner   : Person 1 + Person 2 (shared)
Status  : PLACEHOLDER - structure only, no implementation yet.

Responsibility
--------------
- Read DFDC metadata JSON (label FAKE/REAL, split, original) and build one
  manifest row per selected video.
- Enforce source-group splitting: a fake video and its `original` real video,
  plus every known derivative, must land in the same split (train/val/test).
- Balance real vs fake by video count, not by frame count.
- Record decode/face-detection quality columns for later filtering.

Contract
--------
- Writes data/manifests/manifest.csv with columns:
  video_id, filepath, label, original_id, source_group_id, split, clip_start_sec
- Also writes data/manifests/split_summary.json (counts per split, per group).
- Consumers: src/data/video_dataset.py, scripts/smoke_branches.py fixtures.
- Row schema: src/contracts.py ManifestRow

Target subset (plan 2.2)
------------------------
Train 525+525, Val 112+112, Test 113+113 = 1500 videos. Planning targets, not
archive guarantees - reduce counts if the archives are short, keep the protocol.

TODO
----
- [ ] DFDC metadata loader (label / split / original)
- [ ] Build source groups from `original` links, transitively
- [ ] Group-level split assignment with a fixed seed
- [ ] Balance report + refuse to write on split leakage
- [ ] CLI: python -m src.data.build_manifest --config configs/dataset.yaml
"""

# TODO(person-1,person-2): implement manifest builder described above.
