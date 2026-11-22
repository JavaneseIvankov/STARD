"""STAF-Net | src/models/vjepa_branch.py

Owner   : Person 2 (Video branch)
Status  : PLACEHOLDER - structure only, no implementation yet.

Responsibility
--------------
- Load V-JEPA 2.1 ViT-B/16 video encoder (~80M params) at 384 resolution.
- Run a continuous 16-frame clip and pool encoder representations into z_J.
- Use only the video encoder; the JEPA predictor is not needed for classification.
- Support frozen and LoRA-adapted variants through one class (use_lora flag).

Contract
--------
- Class name: VJepaBranch
- Input  : frames (B, 3, 16, 384, 384) float [0, 1], layout (B, C, T, H, W)
           NOTE: channel-first, opposite of the DINO branch. Do not mix.
- Output : src.contracts.BranchFeatures(z=(B, 768), branch="vjepa")
- Pooling: mean over the returned token grid after the encoder (plan 6.1)

Hyperparameters (plan 6.2)
--------------------------
LoRA r=8, alpha=16, dropout=0.05 on attention projections; lr 5e-5..1e-4 LoRA /
5e-4 head; AdamW wd 0.01; BF16; microbatch 1-2 with grad accumulation 8-16;
SDPA/FlashAttention where supported; activation checkpointing if needed;
1-2 epochs; BCEWithLogitsLoss; select on validation AUROC.

Known risks (docs/open_decisions.md D3)
---------------------------------------
- Checkpoint declares crop_size=384, frames_per_clip=64, tubelet_size=2,
  interpolate_rope=true. 16 frames yields T_out=8 before pooling.
- Verify the encoder actually accepts 16 frames and 384 input before LoRA.

TODO
----
- [ ] Load the released checkpoint and confirm hub/repo entry point
- [ ] Verify accepted num_frames and output tensor structure
- [ ] Print attention module names, then set LoRA target_modules explicitly
- [ ] Pool -> (B, 768)
- [ ] Temporary binary head + BCEWithLogitsLoss
- [ ] Benchmark step time and peak VRAM before scheduling the GPU session
- [ ] scripts/smoke_branches.py --branch vjepa until exit 0
"""

# TODO(person-2): implement VJepaBranch described above.
