"""STARD | src/contracts.py

Owner   : Shared (Orang C maintains, everyone consumes)
Status  : PLACEHOLDER - dataclasses not written yet.

Responsibility
--------------
Single source of truth for tensor shapes and dtypes exchanged between modules.
Frozen by team agreement before parallel work starts.

Contract
--------
BatchInput
    frames         : (B, 16, 3, 224, 224) float32 in [0.0, 1.0]
    label_cls      : (B,) long - 0 Real | 1 Deepfake | 2 Physical Spoof
    label_spatial  : (B, 16, 224, 224) float 0.0/1.0
    label_temporal : (B, 16) float 0.0/1.0

BackboneFeatures
    spatial        : (B, 16, P, D_spatial)  e.g. (B, 16, 196, 768)
    temporal       : (B, 16, P, D_temporal) e.g. (B, 16, 196, 768)

ModelOutputs
    logits_cls     : (B, 3)
    mask_spatial   : (B, 16, 224, 224)
    mask_temporal  : (B, 16)

TODO
----
- [ ] @dataclass BatchInput
- [ ] @dataclass BackboneFeatures
- [ ] @dataclass ModelOutputs
- [ ] Shape assertion helper used by every producer/consumer
"""

# TODO(shared): implement dataclasses above. No code yet - contract text is authoritative.
