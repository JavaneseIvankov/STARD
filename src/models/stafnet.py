"""STAF-Net | src/models/stafnet.py

Owner   : Person 3 (Fusion / evaluation)
Status  : PLACEHOLDER - structure only, no implementation yet.

Responsibility
--------------
- Project each branch embedding into a shared 256-dim space with GELU.
- Concatenation baseline and gated fusion.
- Binary classifier head; expose the gate values for analysis.
- Drive ablations A1-A8 through one class with explicit switches.

Contract
--------
- Class name: STAFNet
- Input  : src.contracts.FusionInput (either branch may be None for A1-A4)
- Output : src.contracts.STAFNetOutput(logit=(B,1), gate=(B,256) | None)

Formulas (plan 7.1-7.3)
-----------------------
    h_D = GELU(W_D z_D + b_D)          z_D (B, 384) -> (B, 256)
    h_J = GELU(W_J z_J + b_J)          z_J (B, 768) -> (B, 256)
    concat : logit = MLP([h_D || h_J])
    gated  : g = sigmoid(W_g [h_D || h_J] + b_g)
             h_fused = g * h_D + (1 - g) * h_J
             logit = MLP(h_fused)
- Head: 256 -> 128 -> 1, GELU, dropout 0.2 (concat input is 512 -> 128 -> 1).
- The gate is an adaptive weighting, not a causal or physics-based explanation.

Ablation switches (plan 8)
--------------------------
A1 dino only + MLP            A2 vjepa only + MLP
A3 dino_lora only + MLP       A4 vjepa_lora only + MLP
A5 frozen + concat            A6 frozen + gated
A7 lora + concat              A8 lora + gated (proposed)

TODO
----
- [ ] Projection layers 384->256 and 768->256 with GELU
- [ ] Concat head (512 -> 128 -> 1)
- [ ] Gated head (gate 512 -> 256, then 256 -> 128 -> 1)
- [ ] Single-branch paths for A1-A4
- [ ] Return gate values for analysis plots
- [ ] Train on synthetic features until real caches exist
"""

# TODO(person-3): implement STAFNet described above.
