"""STAF-Net | src/data/transforms.py

Owner   : Person 1 + Person 2 (shared). Owner of record: Person 1.
Status  : IMPLEMENTED - pure torch / PIL, no extra dependencies.

Two preprocessing pipelines, one per backbone, matching each checkpoint's
official configuration.

DINOv3 ViT-S/16
---------------
224 x 224, ImageNet mean/std. Normalization lives in src/models/dino_branch.py
so the dataset stays backbone-agnostic and the cache never stores normalized
pixels.

V-JEPA 2.1 ViT-B/16
-------------------
384 x 384. Normalization constants are taken from the released checkpoint
config; the fallback below is ImageNet statistics and MUST be confirmed on the
server (docs/open_decisions.md D3) before any A2/A4 run is reported.

Augmentation policy
-------------------
Geometry is applied identically to every frame of one clip (one crop box, one
flip decision per clip), because per-frame jitter would fabricate temporal
inconsistency that DFDC fakes did not contain. Appearance augmentation is mild:
DFDC clips are already compressed, so aggressive color or blur augmentation
mostly erases the artifacts we want to detect.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.contracts import DINO_FRAMES, DINO_SIZE, VJEPA_FRAMES, VJEPA_SIZE

__all__ = [
    "DINO_FRAME_INDICES",
    "TransformConfig",
    "VJEPA_MEAN",
    "VJEPA_STD",
    "build_dino_transform",
    "build_vjepa_transform",
    "apply_clip_augmentation",
]

# Frames 0, 5, 10, 15 of the 16-frame window (plan 3).
DINO_FRAME_INDICES: tuple[int, ...] = (0, 5, 10, 15)

# Confirm against the released V-JEPA 2.1 checkpoint config before reporting.
VJEPA_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
VJEPA_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TransformConfig:
    """Mild train-time augmentation, disabled for val/test."""

    train: bool = False
    hflip_p: float = 0.5
    brightness: float = 0.10
    contrast: float = 0.10
    jpeg_quality: tuple[int, int] = (85, 95)
    jpeg_p: float = 0.0
    blur_p: float = 0.0

    def __post_init__(self) -> None:
        if not self.train:
            return
        for name in ("hflip_p", "jpeg_p", "blur_p"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"TransformConfig.{name}: probability must be in [0, 1], got {value}")


def apply_clip_augmentation(
    clip: torch.Tensor,
    cfg: TransformConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply one geometric decision per clip, then mild per-clip appearance noise.

    Parameters
    ----------
    clip : (T, C, H, W) float32 in [0, 1].

    Returns a new tensor of the same shape. Geometry is shared across frames so
temporal consistency is preserved.
    """
    if not cfg.train or cfg.hflip_p == 0.0:
        return clip
    if float(torch.rand((), generator=generator)) < cfg.hflip_p:
        return torch.flip(clip, dims=(-1,))
    return clip


def build_dino_transform(cfg: TransformConfig | None = None, size: int = DINO_SIZE):
    """Return a callable mapping (T, C, H, W) float [0,1] -> (T, 3, size, size)."""
    cfg = cfg or TransformConfig(train=False)

    def _fn(clip: torch.Tensor) -> torch.Tensor:
        if clip.ndim != 4:
            raise ValueError(f"dino transform: expected (T, C, H, W), got {tuple(clip.shape)}")
        clip = apply_clip_augmentation(clip, cfg)
        if clip.shape[-2:] != (size, size):
            clip = F.interpolate(clip, size=(size, size), mode="bicubic", align_corners=False)
            clip = clip.clamp(0.0, 1.0)
        return clip.contiguous()

    return _fn


def build_vjepa_transform(cfg: TransformConfig | None = None, size: int = VJEPA_SIZE):
    """Return a callable mapping (C, T, H, W) float [0,1] -> (C, T, size, size).

    The V-JEPA branch is channel-first, matching src.contracts.
    """
    cfg = cfg or TransformConfig(train=False)

    def _fn(clip: torch.Tensor) -> torch.Tensor:
        if clip.ndim != 4:
            raise ValueError(f"vjepa transform: expected (C, T, H, W), got {tuple(clip.shape)}")
        t, c = clip.shape[1], clip.shape[0]
        if (t, c) != (VJEPA_FRAMES, 3):
            raise ValueError(
                f"vjepa transform: expected (3, {VJEPA_FRAMES}, H, W), got {tuple(clip.shape)}"
            )
        # augment in (T, C, H, W) then convert back so the flip logic is shared
        per_frame = clip.permute(1, 0, 2, 3)
        per_frame = apply_clip_augmentation(per_frame, cfg)
        if per_frame.shape[-2:] != (size, size):
            per_frame = F.interpolate(per_frame, size=(size, size), mode="bicubic", align_corners=False)
            per_frame = per_frame.clamp(0.0, 1.0)
        return per_frame.permute(1, 0, 2, 3).contiguous()

    return _fn


def select_dino_frames(clip: torch.Tensor) -> torch.Tensor:
    """Take frames 0, 5, 10, 15 from a 16-frame clip: (T, C, H, W) -> (4, C, H, W)."""
    if clip.shape[0] != VJEPA_FRAMES:
        raise ValueError(
            f"select_dino_frames: expected {VJEPA_FRAMES} frames, got {clip.shape[0]}"
        )
    out = clip[list(DINO_FRAME_INDICES)]
    if out.shape[0] != DINO_FRAMES:
        raise ValueError(
            f"select_dino_frames: produced {out.shape[0]} frames, contract expects {DINO_FRAMES}"
        )
    return out


__all__.append("select_dino_frames")
