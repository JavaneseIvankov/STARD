"""STARD | src/models/fusion.py

Owner   : Orang C / Shared (Integration)
Status  : PLACEHOLDER - structure only, no implementation yet.

Responsibility
--------------
- Feature projection to a shared hidden dimension d_hidden.
- Gated fusion: fused = g * S_proj + (1 - g) * T_proj, g = sigmoid(gate_net([S_proj, T_proj])).
- Ablation variant: simple concatenation instead of gating.

Contract
--------
- Input  : src.contracts.BackboneFeatures
- Output : fused tokens (B, 16, P, d_hidden)
- Config : configs/fusion_full.yaml, configs/ablation/*.yaml

TODO
----
- [ ] Linear projection for spatial and temporal branches
- [ ] Gate network (2*d_hidden -> d_hidden -> d_hidden -> sigmoid)
- [ ] Concatenation fusion variant for ablation study
"""

# TODO(orang-c): implement GatedFusionModule described above.
