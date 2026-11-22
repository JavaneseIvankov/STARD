"""STAF-Net source package root.

Spatial-Temporal Aware Fusion Network for video deepfake detection on DFDC.
DINOv3 ViT-S/16 (spatial) + V-JEPA 2.1 ViT-B/16 (spatiotemporal), binary
real/fake classification.

Sub-packages
------------
- data   : DFDC manifest, face tracking, clip dataset, per-branch transforms
- models : DinoBranch, VJepaBranch, STAFNet fusion
- utils  : LoRA injection helper, evaluation metrics

Entry points live at this level: train_dino.py, train_vjepa.py,
extract_features.py, train_fusion.py, evaluate.py.

Shared tensor contract: src/contracts.py
"""

# PLACEHOLDER: no package-level exports yet.
