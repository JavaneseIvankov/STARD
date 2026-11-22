"""STARD | src/models/detector.py

Owner   : Orang C / Shared (Integration)
Status  : PLACEHOLDER - structure only, no implementation yet.

Responsibility
--------------
- Full pipeline assembler: DINOv3 + V-JEPA -> fusion -> multi-task heads.
- Single entry point used by training, evaluation, and inference.

Contract
--------
- Input  : frames (B, 16, 3, 224, 224)
- Output : src.contracts.ModelOutputs
- Supports modality flags: spatial-only, temporal-only, full fusion.

TODO
----
- [ ] Assemble sub-networks from config
- [ ] Implement spatial-only / temporal-only / fusion forward paths
- [ ] Expose checkpoint-friendly state_dict structure
"""

# TODO(orang-c): implement STARDDetector assembler described above.
