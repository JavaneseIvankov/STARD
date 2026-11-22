"""STAF-Net | src/models/dino_branch.py

Owner   : Person 1 (Spatial branch)
Status  : IMPLEMENTED - shape/behaviour verified on CPU with a randomly
          initialised backbone. Real weights need gated HF access (see below).

Responsibility
--------------
- Load DINOv3 ViT-S/16 and run 4 uniformly spaced frames of one clip.
- Pool frame embeddings into a video-level vector z_D.
- Provide the temporary supervised real/fake head used during Stage I.
- Support frozen and LoRA-adapted variants through one class (use_lora flag).

Contract
--------
- Input  : frames (B, 4, 3, 224, 224) float [0, 1], layout (B, T, C, H, W)
- Output : src.contracts.BranchFeatures(z=(B, 384), branch="dino")
- Pooling: mean over the 4 frame CLS embeddings (plan 5.1, decision D5)

Pooling note (D5)
-----------------
DINOv3 ViT-S/16 in transformers returns `pooler_output` = CLS token after
layernorm, with `num_register_tokens = 0`. We mean-pool the CLS token across the
four frames (not patch tokens) so z_D stays an image-level descriptor and the
vector is invariant to patch-count changes.

Access note
-----------
facebook/dinov3-vits16-pretrain-lvd1689m is a GATED repo (`gated: manual`).
Request access on the model page, then run `hf auth login`. Without access,
`pretrained=False` builds the same architecture with random weights, which is
enough for shape tests but produces meaningless metrics.

Hyperparameters (plan 5.2)
--------------------------
LoRA r=8, alpha=16, dropout=0.05 on q_proj/v_proj; lr 1e-4 LoRA / 5e-4 head;
AdamW wd 0.01; BF16; microbatch 4-8; 1-2 epochs; BCEWithLogitsLoss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.transforms import Normalize

from src.contracts import DINO_DIM, DINO_FRAMES, DINO_SIZE, BranchFeatures
from src.utils.lora import LoRAReport, attach_lora, count_trainable, freeze_module

__all__ = ["DinoBranch", "DEFAULT_CHECKPOINT", "IMAGENET_MEAN", "IMAGENET_STD"]

DEFAULT_CHECKPOINT = "facebook/dinov3-vits16-pretrain-lvd1689m"

# From the checkpoint's pretrained_cfg (mean / std).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoBranch(nn.Module):
    """DINOv3 ViT-S/16 clip encoder -> (B, 384) real/fake logit.

    Parameters
    ----------
    pretrained_name:
        HF repo id or local directory holding the checkpoint.
    pretrained:
        False builds the architecture with random weights (shape tests only).
    use_lora:
        Wrap the backbone with LoRA adapters instead of training the head alone.
    lora_targets:
        Projection suffixes to adapt. Verified names in transformers 5.x:
        q_proj, k_proj, v_proj, o_proj.
    """

    def __init__(
        self,
        pretrained_name: str | None = None,
        *,
        pretrained: bool = True,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_targets: tuple[str, ...] = ("q_proj", "v_proj"),
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        from transformers import DINOv3ViTConfig, DINOv3ViTModel

        checkpoint = pretrained_name or DEFAULT_CHECKPOINT
        if pretrained:
            backbone = DINOv3ViTModel.from_pretrained(checkpoint)
        else:
            cfg = DINOv3ViTConfig(hidden_size=DINO_DIM, patch_size=16, image_size=DINO_SIZE)
            backbone = DINOv3ViTModel(cfg)

        self.backbone = freeze_module(backbone)
        self.encoder_dim = int(self.backbone.config.hidden_size)
        if self.encoder_dim != DINO_DIM:
            raise ValueError(
                f"DinoBranch: checkpoint hidden size {self.encoder_dim} does not match "
                f"contract DINO_DIM={DINO_DIM}. Use a ViT-S/16 checkpoint or update "
                f"src/contracts.py for the whole team."
            )

        self.normalize = Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.head = nn.Sequential(
            nn.LayerNorm(DINO_DIM),
            nn.Dropout(dropout),
            nn.Linear(DINO_DIM, hidden_dim or DINO_DIM // 2),
            nn.GELU(),
            nn.Linear(hidden_dim or DINO_DIM // 2, 1),
        )

        self.use_lora = False
        self.lora_report: LoRAReport | None = None
        if use_lora:
            self.backbone, self.lora_report = attach_lora(
                self.backbone,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=lora_targets,
            )
            self.use_lora = True

    # ------------------------------------------------------------------ core
    def encode(self, frames: torch.Tensor) -> BranchFeatures:
        """frames (B, 4, 3, 224, 224) -> BranchFeatures(z=(B, 384))."""
        if frames.ndim != 5:
            raise ValueError(f"DinoBranch.encode: expected 5-D input, got {tuple(frames.shape)}")
        b, t, c, h, w = frames.shape
        if (t, c, h, w) != (DINO_FRAMES, 3, DINO_SIZE, DINO_SIZE):
            raise ValueError(
                f"DinoBranch.encode: expected (B, {DINO_FRAMES}, 3, {DINO_SIZE}, {DINO_SIZE}) "
                f"with layout (B, T, C, H, W), got {tuple(frames.shape)}"
            )

        flat = frames.reshape(b * t, c, h, w)
        flat = self.normalize(flat)
        out = self.backbone(pixel_values=flat)
        cls = out.pooler_output  # (B*4, 384), CLS after layernorm

        if cls.shape[-1] != self.encoder_dim:
            raise ValueError(
                f"DinoBranch.encode: unexpected pooled width {cls.shape[-1]}, "
                f"expected {self.encoder_dim}"
            )
        z = cls.reshape(b, t, self.encoder_dim).mean(dim=1)
        z = z.to(torch.float32)
        return BranchFeatures(z=z, branch="dino")

    def forward(self, frames: torch.Tensor) -> BranchFeatures:
        """Embedding forward used by the smoke test and feature extraction."""
        return self.encode(frames)

    def predict_logit(self, frames: torch.Tensor) -> torch.Tensor:
        """frames -> (B, 1) raw logit for BCEWithLogitsLoss."""
        z = self.encode(frames).z
        return self.head(z)

    # -------------------------------------------------------------- training
    def trainable_parameters(self) -> list[nn.Parameter]:
        """Adapter parameters (when LoRA) plus head parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def lora_parameter_groups(self) -> list[dict[str, object]]:
        """Separate LoRA and head groups so they can use different LRs."""
        lora = [p for n, p in self.named_parameters() if p.requires_grad and "lora_" in n]
        head = [p for n, p in self.named_parameters() if p.requires_grad and "lora_" not in n]
        groups: list[dict[str, object]] = []
        if lora:
            groups.append({"params": lora, "lr_scale": "lora"})
        if head:
            groups.append({"params": head, "lr_scale": "head"})
        return groups

    def describe(self) -> str:
        trainable, total, ratio = count_trainable(self)
        mode = "lora" if self.use_lora else "frozen"
        extra = f" | {self.lora_report.summary()}" if self.lora_report else ""
        return (
            f"DinoBranch[{mode}] dim={self.encoder_dim} "
            f"trainable={trainable:,}/{total:,} ({100 * ratio:.2f}%){extra}"
        )
