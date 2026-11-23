"""STAF-Net | src/models/dino_branch.py

Owner   : Person 1 (Spatial branch)
Status  : IMPLEMENTED - dual loader (transformers + timm), verified on CPU with a
          synthetic checkpoint in the exact on-server format.

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

Two checkpoint layouts, auto-detected from config.json
------------------------------------------------------
1. transformers (`model_type: dinov3_vit`, `architectures: [DINOv3ViTModel]`)
     attention projections: q_proj / k_proj / v_proj / o_proj
     pooled CLS: out.pooler_output
     LoRA targets: q_proj, v_proj

2. timm (`architecture: vit_small_patch16_dinov3`, `pretrained_cfg`, tag lvd1689m)
     attention: ONE fused `qkv` Linear(1152, 384) plus `proj`
     prefix tokens: 1 CLS + 4 registers -> forward_features (B, 201, 384)
     pooled CLS: forward_features(x)[:, 0]
     LoRA targets: qkv

This is the practical resolution of decision D6: the LoRA target set depends on
the loader, so it is chosen from the detected backend instead of hardcoded. A
mismatched target list raises immediately instead of silently training nothing.

Access note
-----------
facebook/dinov3-vits16-pretrain-lvd1689m is a gated HF repo. Pointing
`pretrained_name` at a local directory that contains config.json +
model.safetensors bypasses the hub entirely and needs no login.

Hyperparameters (plan 5.2)
--------------------------
LoRA r=8, alpha=16, dropout=0.05; lr 1e-4 LoRA / 5e-4 head; AdamW wd 0.01;
BF16; microbatch 4-8; 1-2 epochs; BCEWithLogitsLoss.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.transforms import Normalize

from src.contracts import DINO_DIM, DINO_FRAMES, DINO_SIZE, BranchFeatures
from src.utils.lora import LoRAReport, attach_lora, count_trainable, freeze_module

__all__ = [
    "DinoBranch",
    "DEFAULT_CHECKPOINT",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "detect_backend",
    "LORA_TARGETS",
]

DEFAULT_CHECKPOINT = "facebook/dinov3-vits16-pretrain-lvd1689m"

# From the checkpoint's pretrained_cfg (mean / std).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# LoRA target modules per loader (decision D6).
LORA_TARGETS: dict[str, tuple[str, ...]] = {
    "transformers": ("q_proj", "v_proj"),
    "timm": ("qkv",),
}


def detect_backend(checkpoint: str | Path) -> str:
    """Return 'transformers' or 'timm' by inspecting a local checkpoint dir.

    Raises FileNotFoundError when the directory has no config.json, so a typo in
the weights path fails loudly instead of silently falling back to the hub.
    """
    path = Path(checkpoint)
    config_path = path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"detect_backend: {config_path} not found. Pass the local weights directory "
            f"(config.json + model.safetensors) or an HF repo id."
        )
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    model_type = str(cfg.get("model_type", ""))
    architectures = " ".join(str(a) for a in cfg.get("architectures", []))
    if model_type.startswith("dinov3") or "DINOv3" in architectures:
        return "transformers"
    if "architecture" in cfg or "pretrained_cfg" in cfg:
        return "timm"
    raise ValueError(
        f"detect_backend: cannot classify {config_path}. Keys: {sorted(cfg)[:10]}"
    )


class DinoBranch(nn.Module):
    """DINOv3 ViT-S/16 clip encoder -> (B, 384) embedding plus a real/fake head.

    Parameters
    ----------
    pretrained_name:
        Local weights directory or HF repo id. A local dir avoids the gated hub.
    pretrained:
        False builds the architecture with random weights (pipeline smoke tests).
    use_lora:
        Wrap the backbone with LoRA adapters instead of training the head alone.
    lora_targets:
        Projection suffixes to adapt. None picks the loader-correct default from
        LORA_TARGETS (q_proj/v_proj for transformers, qkv for timm).
    """

    def __init__(
        self,
        pretrained_name: str | Path | None = None,
        *,
        pretrained: bool = True,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_targets: tuple[str, ...] | None = None,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        checkpoint = str(pretrained_name or DEFAULT_CHECKPOINT)

        if Path(checkpoint).is_dir():
            self.backend = detect_backend(checkpoint)
        else:
            self.backend = "transformers"

        backbone = self._build_backbone(checkpoint, pretrained=pretrained)
        self.backbone = freeze_module(backbone)

        self.encoder_dim = self._resolve_encoder_dim()
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
        self.lora_targets: tuple[str, ...] = tuple(lora_targets or LORA_TARGETS[self.backend])
        if use_lora:
            targets = self.lora_targets
            self.backbone, self.lora_report = attach_lora(
                self.backbone,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=targets,
            )
            self.use_lora = True

    # ---------------------------------------------------------------- loading
    def _build_backbone(self, checkpoint: str, *, pretrained: bool) -> nn.Module:
        if self.backend == "transformers":
            from transformers import DINOv3ViTConfig, DINOv3ViTModel

            if pretrained:
                return DINOv3ViTModel.from_pretrained(checkpoint)
            cfg = DINOv3ViTConfig(
                hidden_size=DINO_DIM, patch_size=16, image_size=DINO_SIZE
            )
            return DINOv3ViTModel(cfg)

        import timm
        from safetensors.torch import load_file

        path = Path(checkpoint)
        cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
        arch = cfg.get("architecture")
        if not arch:
            raise ValueError(f"DinoBranch: timm config in {path} has no 'architecture' key")

        model = timm.create_model(
            arch,
            pretrained=False,
            num_classes=0,
            img_size=DINO_SIZE,
            global_pool=cfg.get("global_pool", "avg"),
        )
        if not pretrained:
            return model

        weights = path / "model.safetensors"
        if not weights.exists():
            raise FileNotFoundError(
                f"DinoBranch: {weights} not found. The timm layout needs "
                f"config.json + model.safetensors in the same directory."
            )
        state = load_file(weights)
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith(("head.", "fc."))]
        if missing or unexpected:
            raise ValueError(
                f"DinoBranch: checkpoint mismatch for {arch}: "
                f"missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        return model

    def _resolve_encoder_dim(self) -> int:
        if self.backend == "transformers":
            return int(self.backbone.config.hidden_size)
        for attr in ("num_features", "embed_dim"):
            value = getattr(self.backbone, attr, None)
            if value:
                return int(value)
        raise ValueError("DinoBranch: cannot determine timm backbone feature width")

    # ------------------------------------------------------------------- core
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
        cls = self._forward_cls(flat)

        if cls.shape[-1] != self.encoder_dim:
            raise ValueError(
                f"DinoBranch.encode: unexpected pooled width {cls.shape[-1]}, "
                f"expected {self.encoder_dim}"
            )
        z = cls.reshape(b, t, self.encoder_dim).mean(dim=1).to(torch.float32)
        return BranchFeatures(z=z, branch="dino")

    def _forward_cls(self, flat: torch.Tensor) -> torch.Tensor:
        """Return the pooled CLS embedding for a batch of single frames."""
        if self.backend == "transformers":
            out = self.backbone(pixel_values=flat)
            return out.pooler_output

        feats = self.backbone.forward_features(flat)
        if isinstance(feats, dict):
            for key in ("x_norm_clstoken", "x_clstoken", "pre_logits"):
                if key in feats:
                    return feats[key]
            feats = feats.get("x_norm_patchtokens")
            if feats is None:
                raise ValueError(f"DinoBranch: unexpected timm feature keys {list(feats)}")
            return feats.mean(dim=1)
        # (B, 1 CLS + 4 registers + 196 patches, 384) -> CLS
        return feats[:, 0]

    def forward(self, frames: torch.Tensor) -> BranchFeatures:
        """Embedding forward used by the smoke test and feature extraction."""
        return self.encode(frames)

    def predict_logit(self, frames: torch.Tensor) -> torch.Tensor:
        """frames -> (B, 1) raw logit for BCEWithLogitsLoss."""
        z = self.encode(frames).z
        return self.head(z)

    # --------------------------------------------------------------- training
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
            f"DinoBranch[{mode}/{self.backend}] dim={self.encoder_dim} "
            f"trainable={trainable:,}/{total:,} ({100 * ratio:.2f}%){extra}"
        )
