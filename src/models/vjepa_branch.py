"""STAF-Net | src/models/vjepa_branch.py

Owner   : Person 2 (Video branch)
Status  : IMPLEMENTED - supports local HF V-JEPA 2.1 (trust_remote_code),
          transformers VJEPA2Model, and synthetic CPU smoke mode.

Contract
--------
- Class name: VJepaBranch
- Input  : frames (B, 3, 16, 384, 384) float [0, 1], layout (B, C, T, H, W)
- Output : src.contracts.BranchFeatures(z=(B, 768), branch="vjepa")
- Pooling: mean over encoder token sequence -> (B, 768) float32
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch
import torch.nn as nn

from src.contracts import VJEPA_DIM, VJEPA_FRAMES, VJEPA_SIZE, BranchFeatures
from src.utils.lora import (
    LoRAReport,
    attach_lora,
    count_trainable,
    freeze_module,
    inspect_targets,
)

__all__ = [
    "VJepaBranch",
    "DEFAULT_CHECKPOINT",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]

DEFAULT_CHECKPOINT = "apiantonio/vjepa2.1-vit-base-384"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Priority order of LoRA attention projection suffixes across V-JEPA backends.
_LORA_TARGET_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("q_proj", "v_proj"),
    ("query", "value"),
    ("qkv",),
)


class _SyntheticVJepaEncoder(nn.Module):
    """Lightweight ViT-style 3D encoder for --untrained-backbone / CPU smoke tests."""

    def __init__(self, hidden_size: int = VJEPA_DIM) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.patch = nn.Conv3d(3, hidden_size, kernel_size=(2, 16, 16), stride=(2, 16, 16))
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        # Expects (B, T, C, H, W) -> permute to (B, C, T, H, W) for Conv3d
        x = pixel_values_videos.permute(0, 2, 1, 3, 4)
        tokens = self.patch(x).flatten(2).transpose(1, 2)  # (B, N, D)
        q = self.q_proj(tokens)
        v = self.v_proj(tokens)
        out = self.norm(tokens + self.o_proj(torch.nn.functional.gelu(q + v)))
        return out


def _pick_lora_targets(backbone: nn.Module, requested: tuple[str, ...] | None) -> tuple[str, ...]:
    if requested:
        return tuple(requested)
    available = set(inspect_targets(backbone))
    for cand in _LORA_TARGET_CANDIDATES:
        if any(t in available for t in cand):
            return tuple(t for t in cand if t in available)
    raise ValueError(
        f"VJepaBranch: could not auto-detect LoRA attention targets. "
        f"Available linear suffixes: {sorted(available)}"
    )


class VJepaBranch(nn.Module):
    """V-JEPA 2.1 ViT-B/16 clip encoder -> (B, 768) embedding plus binary head.

    Parameters
    ----------
    pretrained_name:
        Local directory (e.g. ~/weights/vjepa2.1-vitb16/hf) or HF repo id.
    pretrained:
        False builds a lightweight synthetic encoder for dataset-free smoke tests.
    use_lora:
        Wrap the video encoder with LoRA adapters.
    gradient_checkpointing:
        Enable activation checkpointing if supported by the loaded backbone.
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
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        checkpoint = str(pretrained_name or DEFAULT_CHECKPOINT)
        # If user passed ~/weights/vjepa2.1-vitb16 and hf/ subdir exists, use hf/
        ckpt_path = Path(checkpoint).expanduser()
        if ckpt_path.is_dir() and not (ckpt_path / "config.json").exists() and (ckpt_path / "hf" / "config.json").exists():
            ckpt_path = ckpt_path / "hf"
            checkpoint = str(ckpt_path)
        elif ckpt_path.exists():
            checkpoint = str(ckpt_path)

        self.checkpoint = checkpoint
        self.backend = "synthetic" if not pretrained else "hf"
        backbone = self._build_backbone(checkpoint, pretrained=pretrained)
        self.backbone = freeze_module(backbone)

        if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

        self.encoder_dim = self._resolve_encoder_dim()
        if self.encoder_dim != VJEPA_DIM:
            raise ValueError(
                f"VJepaBranch: checkpoint hidden size {self.encoder_dim} does not match "
                f"contract VJEPA_DIM={VJEPA_DIM}."
            )

        self.register_buffer(
            "_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1, 1),
            persistent=False,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(VJEPA_DIM),
            nn.Dropout(dropout),
            nn.Linear(VJEPA_DIM, hidden_dim or VJEPA_DIM // 2),
            nn.GELU(),
            nn.Linear(hidden_dim or VJEPA_DIM // 2, 1),
        )

        self.use_lora = False
        self.lora_report: LoRAReport | None = None
        self.lora_targets: tuple[str, ...] = ()
        if use_lora:
            targets = _pick_lora_targets(self.backbone, lora_targets)
            self.lora_targets = targets
            self.backbone, self.lora_report = attach_lora(
                self.backbone,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=targets,
            )
            self.use_lora = True

        # Detect which forward arg name and optional skip_predictor flag the backbone accepts
        self._input_arg, self._has_skip_predictor = self._detect_forward_args()

    def _build_backbone(self, checkpoint: str, *, pretrained: bool) -> nn.Module:
        if not pretrained:
            return _SyntheticVJepaEncoder(VJEPA_DIM)

        from transformers import AutoModel

        return AutoModel.from_pretrained(checkpoint, trust_remote_code=True)

    def _resolve_encoder_dim(self) -> int:
        cfg = getattr(self.backbone, "config", None)
        if cfg is not None:
            for attr in ("hidden_size", "embed_dim", "dim"):
                val = getattr(cfg, attr, None)
                if val:
                    return int(val)
        for attr in ("hidden_size", "embed_dim", "num_features"):
            val = getattr(self.backbone, attr, None)
            if val:
                return int(val)
        return VJEPA_DIM

    def _detect_forward_args(self) -> tuple[str, bool]:
        base = self.backbone
        if hasattr(base, "get_base_model"):
            base = base.get_base_model()
        sig = inspect.signature(base.forward)
        arg_name = "pixel_values_videos"
        for name in ("pixel_values_videos", "pixel_values", "x"):
            if name in sig.parameters:
                arg_name = name
                break
        return arg_name, ("skip_predictor" in sig.parameters)

    def _forward_tokens(self, frames_bcthw: torch.Tensor) -> torch.Tensor:
        # Normalize (B, C, T, H, W) using ImageNet mean/std
        normed = (frames_bcthw - self._mean) / self._std
        # HF VJEPA2 / apiantonio port expects (B, T, C, H, W) for pixel_values_videos
        btchw = normed.permute(0, 2, 1, 3, 4).contiguous()
        extra = {"skip_predictor": True} if self._has_skip_predictor else {}

        try:
            out = self.backbone(**{self._input_arg: btchw, **extra})
        except (RuntimeError, ValueError):
            # Fallback for backbones expecting (B, C, T, H, W)
            out = self.backbone(**{self._input_arg: normed, **extra})

        if isinstance(out, torch.Tensor):
            tokens = out
        elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            tokens = out.last_hidden_state
        elif isinstance(out, (tuple, list)):
            tokens = out[0]
        elif isinstance(out, dict):
            tokens = out.get("last_hidden_state") or next(iter(out.values()))
        else:
            raise ValueError(f"VJepaBranch: unexpected backbone output type {type(out)}")
        return tokens

    def encode(self, frames: torch.Tensor) -> BranchFeatures:
        """frames (B, 3, 16, 384, 384) -> BranchFeatures(z=(B, 768))."""
        if frames.ndim != 5:
            raise ValueError(f"VJepaBranch.encode: expected 5-D input, got {tuple(frames.shape)}")
        b, c, t, h, w = frames.shape
        if (c, t, h, w) != (3, VJEPA_FRAMES, VJEPA_SIZE, VJEPA_SIZE):
            raise ValueError(
                f"VJepaBranch.encode: expected (B, 3, {VJEPA_FRAMES}, {VJEPA_SIZE}, {VJEPA_SIZE}) "
                f"with layout (B, C, T, H, W), got {tuple(frames.shape)}"
            )

        tokens = self._forward_tokens(frames)
        if tokens.ndim == 3:
            z = tokens.mean(dim=1)
        elif tokens.ndim == 2:
            z = tokens
        else:
            z = tokens.flatten(1, -2).mean(dim=1)

        if z.shape[-1] != self.encoder_dim:
            raise ValueError(
                f"VJepaBranch.encode: unexpected pooled width {z.shape[-1]}, expected {self.encoder_dim}"
            )
        return BranchFeatures(z=z.to(torch.float32), branch="vjepa")

    def forward(self, frames: torch.Tensor) -> BranchFeatures:
        return self.encode(frames)

    def predict_logit(self, frames: torch.Tensor) -> torch.Tensor:
        """frames -> (B, 1) raw logit for BCEWithLogitsLoss."""
        z = self.encode(frames).z
        return self.head(z)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def lora_parameter_groups(self) -> list[dict[str, object]]:
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
            f"VJepaBranch[{mode}/{self.backend}] dim={self.encoder_dim} "
            f"trainable={trainable:,}/{total:,} ({100 * ratio:.2f}%){extra}"
        )
