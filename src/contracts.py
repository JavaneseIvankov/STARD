"""STARD | src/contracts.py

Owner   : Shared (Orang C maintains; Orang 1 / A / B consume)
Status  : FROZEN v0.1 - shapes locked, model implementations not required.

Single source of truth for tensor shapes, dtypes, and class order exchanged
between data, backbone, fusion, and head modules. Change here FIRST, then
notify the team - never silently diverge inside a module.

Locked geometry
---------------
NUM_FRAMES  = 16          frames per clip
IMAGE_SIZE  = 224         facial crop resolution (square)
PATCH_SIZE  = 16          both backbones are patch-16
GRID_SIZE   = 14          224 / 16
NUM_PATCHES = 196         14 * 14 patch tokens per frame

Backbones
---------
DINOv3 ViT-S/16   : D_spatial  = 384  (num_features from dinov3-vits16/config.json)
V-JEPA 2.1 ViT-B/16: D_temporal = 768  (hidden_size from vjepa2.1-vitb16/config.json)

Gotcha #1 - token count
    P = 196 is only valid for patch-16 at 224. A patch-14 backbone (e.g. the
    DINOv2 proxy in the development guide) yields 256 tokens and silently
    breaks the 14x14 -> 224 upsample in heads.py.

Gotcha #2 - V-JEPA temporal reduction
    vjepa2.1-vitb16 has tubelet_size = 2, so 16 input frames produce T_out = 8.
    The temporal wrapper must interpolate back to NUM_FRAMES = 16 before
    returning BackboneFeatures. This module asserts the post-alignment shape.

Gotcha #3 - V-JEPA pretrained geometry
    The checkpoint declares crop_size = 384 and frames_per_clip = 64. Feeding
    224 / 16 frames is only legal because interpolate_rope = true; verify it in
    the wrapper and record the result. Contract shape stays (B, 16, 196, 768).

Mask semantics
--------------
ModelOutputs.mask_* are RAW LOGITS (pre-sigmoid). Apply sigmoid once, in the
loss (BCEWithLogits) or in metrics - never both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import torch

CONTRACT_VERSION: Final[str] = "0.1"

# ---------------------------------------------------------------------------
# 1. Frozen geometry
# ---------------------------------------------------------------------------
NUM_FRAMES: Final[int] = 16
IMAGE_SIZE: Final[int] = 224
PATCH_SIZE: Final[int] = 16
GRID_SIZE: Final[int] = IMAGE_SIZE // PATCH_SIZE  # 14
NUM_PATCHES: Final[int] = GRID_SIZE * GRID_SIZE  # 196

CHANNELS: Final[int] = 3
NUM_CLASSES: Final[int] = 3
D_HIDDEN: Final[int] = 512

CLASS_NAMES: Final[tuple[str, ...]] = ("real", "deepfake", "physical_spoof")
CLASS_TO_INDEX: Final[dict[str, int]] = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# ---------------------------------------------------------------------------
# 2. Backbone specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackboneSpec:
    """Static description of a frozen backbone, used for logging and asserts."""

    name: str
    num_features: int
    patch_size: int
    tokens_per_frame: int
    pretrained_input_size: int
    notes: str = ""


DINOV3_VITS16: Final[BackboneSpec] = BackboneSpec(
    name="dinov3-vits16",
    num_features=384,
    patch_size=16,
    tokens_per_frame=NUM_PATCHES,
    pretrained_input_size=256,
    notes="lvd1689m, fixed_input_size=false, ImageNet mean/std",
)

VJEPA21_VITB16: Final[BackboneSpec] = BackboneSpec(
    name="vjepa2.1-vitb16",
    num_features=768,
    patch_size=16,
    tokens_per_frame=NUM_PATCHES,
    pretrained_input_size=384,
    notes="tubelet_size=2, frames_per_clip=64, interpolate_rope=true",
)

D_SPATIAL: Final[int] = DINOV3_VITS16.num_features  # 384
D_TEMPORAL: Final[int] = VJEPA21_VITB16.num_features  # 768

VJEPA_TUBELET_SIZE: Final[int] = 2
VJEPA_RAW_T_OUT: Final[int] = NUM_FRAMES // VJEPA_TUBELET_SIZE  # 8 -> must be 16

# ---------------------------------------------------------------------------
# 3. Shape helpers
# ---------------------------------------------------------------------------
Shape = Sequence[int | None]


def expect_shape(tensor: torch.Tensor, expected: Shape, name: str) -> None:
    """Raise ValueError unless tensor.shape matches expected (None = wildcard)."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name}: expected torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != len(expected):
        raise ValueError(
            f"{name}: expected {len(expected)} dims {tuple(expected)}, "
            f"got {tensor.ndim} dims {tuple(tensor.shape)}"
        )
    for axis, (got, want) in enumerate(zip(tensor.shape, expected)):
        if want is not None and got != want:
            raise ValueError(
                f"{name}: axis {axis} expected {want}, got {got} "
                f"(shape {tuple(tensor.shape)}, expected {tuple(expected)})"
            )


# ---------------------------------------------------------------------------
# 4. BatchInput - produced by src/data/dataset.py (Orang 1)
# ---------------------------------------------------------------------------


@dataclass
class BatchInput:
    """One training/eval batch. frames are float32 in [0.0, 1.0] (no normalization).

    Mask labels are Optional because some source datasets ship classification
    labels only. If a mask is None its loss weight MUST be 0.0 in the config.
    """

    frames: torch.Tensor  # (B, 16, 3, 224, 224) float32 [0, 1]
    label_cls: torch.Tensor  # (B,) int64 in [0, NUM_CLASSES)
    label_spatial: torch.Tensor | None = None  # (B, 16, 224, 224) float 0/1
    label_temporal: torch.Tensor | None = None  # (B, 16) float 0/1

    @property
    def batch_size(self) -> int:
        return int(self.frames.shape[0])

    def validate(self, *, check_range: bool = True) -> "BatchInput":
        expect_shape(
            self.frames,
            (None, NUM_FRAMES, CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
            "BatchInput.frames",
        )
        if not self.frames.is_floating_point():
            raise TypeError(f"BatchInput.frames: expected float tensor, got {self.frames.dtype}")
        if check_range and self.frames.numel() > 0:
            lo, hi = float(self.frames.min()), float(self.frames.max())
            if lo < 0.0 or hi > 1.0:
                raise ValueError(
                    f"BatchInput.frames: expected range [0, 1] before backbone "
                    f"normalization, got [{lo:.4f}, {hi:.4f}]"
                )

        expect_shape(self.label_cls, (self.batch_size,), "BatchInput.label_cls")
        if self.label_cls.is_floating_point():
            raise TypeError(
                f"BatchInput.label_cls: expected integer tensor, got {self.label_cls.dtype}"
            )
        if self.label_cls.numel() > 0:
            lo, hi = int(self.label_cls.min()), int(self.label_cls.max())
            if lo < 0 or hi >= NUM_CLASSES:
                raise ValueError(
                    f"BatchInput.label_cls: values must be in [0, {NUM_CLASSES - 1}], got [{lo}, {hi}]"
                )

        if self.label_spatial is not None:
            expect_shape(
                self.label_spatial,
                (self.batch_size, NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE),
                "BatchInput.label_spatial",
            )
        if self.label_temporal is not None:
            expect_shape(
                self.label_temporal,
                (self.batch_size, NUM_FRAMES),
                "BatchInput.label_temporal",
            )
        return self

    def to(self, device: torch.device | str, non_blocking: bool = True) -> "BatchInput":
        """Return a copy with every tensor moved to device."""
        def move(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.to(device, non_blocking=non_blocking)

        return BatchInput(
            frames=move(self.frames),
            label_cls=move(self.label_cls),
            label_spatial=move(self.label_spatial),
            label_temporal=move(self.label_temporal),
        )


# ---------------------------------------------------------------------------
# 5. BackboneFeatures - produced by dinov3.py (A) and vjepa.py (B)
# ---------------------------------------------------------------------------


@dataclass
class BackboneFeatures:
    """Backbone output. Either branch may be None for unimodal baselines."""

    spatial: torch.Tensor | None = None  # (B, 16, 196, 384)
    temporal: torch.Tensor | None = None  # (B, 16, 196, 768)

    def validate(self) -> "BackboneFeatures":
        if self.spatial is None and self.temporal is None:
            raise ValueError("BackboneFeatures: at least one of spatial/temporal must be set")
        if self.spatial is not None:
            expect_shape(
                self.spatial,
                (None, NUM_FRAMES, NUM_PATCHES, D_SPATIAL),
                "BackboneFeatures.spatial",
            )
            if not self.spatial.is_floating_point():
                raise TypeError(
                    f"BackboneFeatures.spatial: expected float tensor, got {self.spatial.dtype}"
                )
        if self.temporal is not None:
            expect_shape(
                self.temporal,
                (None, NUM_FRAMES, NUM_PATCHES, D_TEMPORAL),
                "BackboneFeatures.temporal",
            )
            if not self.temporal.is_floating_point():
                raise TypeError(
                    f"BackboneFeatures.temporal: expected float tensor, got {self.temporal.dtype}"
                )
        if self.spatial is not None and self.temporal is not None:
            if self.spatial.shape[0] != self.temporal.shape[0]:
                raise ValueError(
                    "BackboneFeatures: spatial/temporal batch size mismatch "
                    f"({self.spatial.shape[0]} vs {self.temporal.shape[0]})"
                )
        return self

    def to(self, device: torch.device | str, non_blocking: bool = True) -> "BackboneFeatures":
        def move(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.to(device, non_blocking=non_blocking)

        return BackboneFeatures(spatial=move(self.spatial), temporal=move(self.temporal))


# ---------------------------------------------------------------------------
# 6. ModelOutputs - produced by heads.py (Orang C)
# ---------------------------------------------------------------------------


@dataclass
class ModelOutputs:
    """Predictions from MultiTaskHeads. Masks are raw logits (pre-sigmoid)."""

    logits_cls: torch.Tensor  # (B, 3)
    mask_spatial: torch.Tensor | None = None  # (B, 16, 224, 224) logits
    mask_temporal: torch.Tensor | None = None  # (B, 16) logits

    def validate(self) -> "ModelOutputs":
        expect_shape(self.logits_cls, (None, NUM_CLASSES), "ModelOutputs.logits_cls")
        if self.mask_spatial is not None:
            expect_shape(
                self.mask_spatial,
                (None, NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE),
                "ModelOutputs.mask_spatial",
            )
        if self.mask_temporal is not None:
            expect_shape(
                self.mask_temporal,
                (None, NUM_FRAMES),
                "ModelOutputs.mask_temporal",
            )
        return self

    def to(self, device: torch.device | str, non_blocking: bool = True) -> "ModelOutputs":
        def move(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.to(device, non_blocking=non_blocking)

        return ModelOutputs(
            logits_cls=move(self.logits_cls),
            mask_spatial=move(self.mask_spatial),
            mask_temporal=move(self.mask_temporal),
        )


# ---------------------------------------------------------------------------
# 7. Introspection
# ---------------------------------------------------------------------------


def contract_summary() -> dict[str, object]:
    """Flat dict for logging into TensorBoard / WandB at run start."""
    return {
        "contract_version": CONTRACT_VERSION,
        "num_frames": NUM_FRAMES,
        "image_size": IMAGE_SIZE,
        "patch_size": PATCH_SIZE,
        "grid_size": GRID_SIZE,
        "num_patches": NUM_PATCHES,
        "num_classes": NUM_CLASSES,
        "d_spatial": D_SPATIAL,
        "d_temporal": D_TEMPORAL,
        "d_hidden": D_HIDDEN,
        "spatial_backbone": DINOV3_VITS16.name,
        "temporal_backbone": VJEPA21_VITB16.name,
        "vjepa_tubelet_size": VJEPA_TUBELET_SIZE,
        "vjepa_raw_t_out": VJEPA_RAW_T_OUT,
        "class_names": list(CLASS_NAMES),
    }


__all__ = [
    "CONTRACT_VERSION",
    "NUM_FRAMES",
    "IMAGE_SIZE",
    "PATCH_SIZE",
    "GRID_SIZE",
    "NUM_PATCHES",
    "CHANNELS",
    "NUM_CLASSES",
    "D_HIDDEN",
    "D_SPATIAL",
    "D_TEMPORAL",
    "VJEPA_TUBELET_SIZE",
    "VJEPA_RAW_T_OUT",
    "CLASS_NAMES",
    "CLASS_TO_INDEX",
    "BackboneSpec",
    "DINOV3_VITS16",
    "VJEPA21_VITB16",
    "BatchInput",
    "BackboneFeatures",
    "ModelOutputs",
    "expect_shape",
    "contract_summary",
]
