"""STAF-Net | src/models/dino_branch.py

Owner   : Person 1 (Spatial branch)
Status  : PLACEHOLDER - structure only, no implementation yet.

Responsibility
--------------
- Load DINOv3 ViT-S/16 (facebook/dinov3-vits16-pretrain-lvd1689m, 21M params).
- Run 4 uniformly spaced frames of one clip through the shared image encoder and
  pool frame features into a video-level embedding z_D.
- Provide the temporary supervised real/fake head used during Stage I.
- Support frozen and LoRA-adapted variants through one class (use_lora flag).

Contract
--------
- Class name: DinoBranch
- Input  : frames (B, 4, 3, 224, 224) float [0, 1], layout (B, T, C, H, W)
- Output : src.contracts.BranchFeatures(z=(B, 384), branch="dino")
- Pooling: z_D = mean over the 4 frame embeddings (plan 5.1)
- Checkpoint identity + preprocessing_version must be recorded in the cache
  manifest (docs/cache_format.md)

Hyperparameters (plan 5.2)
--------------------------
LoRA r=8, alpha=16, dropout=0.05 on attention Q/V (verify module names first);
lr 1e-4 LoRA / 5e-4 head; AdamW wd 0.01; BF16; microbatch 4-8; 1-2 epochs;
BCEWithLogitsLoss; select on validation AUROC.

TODO
----
- [ ] Load checkpoint and inspect returned fields (CLS vs patch tokens, registers)
- [ ] Print attention module names, then set LoRA target_modules explicitly
- [ ] Pool 4 frame embeddings -> (B, 384)
- [ ] Temporary binary head + BCEWithLogitsLoss
- [ ] Frozen vs LoRA toggle without changing the forward signature
- [ ] scripts/smoke_branches.py --branch dino until exit 0
"""

# TODO(person-1): implement DinoBranch described above.
