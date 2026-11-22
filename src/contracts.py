"""STAF-Net | src/contracts.py

Owner   : Person 3 maintains, Person 1 / Person 2 consume
Status  : FROZEN v0.2 - shapes locked, model implementations not required.

Single source of truth for tensor shapes, layouts, and label semantics
exchanged between data, branch, fusion, and evaluation modules. Change here
FIRST, then tell the team - never silently diverge inside a module.

Geometry is PER BRANCH
----------------------
DINOv3 ViT-S/16   : 4 frames, 224x224, dim 384   layout (B, T, C, H, W)
V-JEPA 2.1 ViT-B/16: 16 frames, 384x384, dim 768  layout (B, C, T, H, W)

The two branches do NOT share resolution, frame count, or channel order. This
is the single most common source of silent shape bugs in this project, so every
validator below checks the exact 5-D layout per branch.

Task
----
Binary video-level classification on DFDC: label 0 = real, 1 = fake.
There are no spatial or temporal mask targets. DFDC ships video-level labels
only, so any frame-level manipulation interval is unavailable by design.

Stages (plan sections 4-8)
--------------------------
Stage I  : train each branch independently (frozen baseline, then LoRA),
           export pooled per-video embeddings for four variants.
Stage II : train fusion / single-branch classifiers on the fixed cached
           embeddings; run ablations A1-A8.

Feature cache
-------------
Variants: dino_frozen, dino_lora, vjepa_frozen, vjepa_lora.
Schema and invariants: docs/cache_format.md
Open questions: docs/open_decisions.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Sequence

import torch

CONTRACT_VERSION: Final[str] = "0.2"

# ---------------------------------------------------------------------------
# 1. Per-branch geometry (frozen)
# ---------------------------------------------------------------------------
DINO_FRAMES: Final[int] = 4
DINO_SIZE: Final[int] = 224
DINO_DIM: Final[int] = 384

VJEPA_FRAMES: Final[int] = 16
VJEPA_SIZE: Final[int] = 384
VJEPA_DIM: Final[int] = 768

# ---------------------------------------------------------------------------
# 2. Fusion / head geometry (plan 7.1-7.4)
# ---------------------------------------------------------------------------
D_PROJ: Final[int] = 256
HEAD_HIDDEN: Final[int] = 128
DROPOUT: Final[float] = 0.2
NUM_LOGITS: Final[int] = 1

# ---------------------------------------------------------------------------
# 3. Enumerations
# ---------------------------------------------------------------------------
BRANCHES: Final[tuple[str, ...]] = ("dino", "vjepa")
FEATURE_DIM: Final[dict[str, int]] = {"dino": DINO_DIM, "vjepa": VJEPA_DIM}

VARIANTS: Final[tuple[str, ...]] = (
    "dino_frozen",
    "dino_lora",
    "vjepa_frozen",
    "vjepa_lora",
)

SPLITS: Final[tuple[str, ...]] = ("train", "val", "test")

LABEL_NAMES: Final[dict[int, str]] = {0: "real", 1: "fake"}

# Pooling definition used by both branches. Open decision D5: if the branches
# end up pooling differently, replace this constant with a per-branch mapping
# and update docs/open_decisions.md instead of hardcoding inside a wrapper.
POOLING: Final[str] = "mean"


def variant_branch(variant: str) -> str:
    """Map a cache variant name to its branch (e.g. 'dino_lora' -> 'dino')."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")
    return variant.split("_", 1)[0]


def variant_dim(variant: str) -> int:
    """Feature width of a cache variant."""
    return FEATURE_DIM[variant_branch(variant)]


def variant_is_lora(variant: str) -> bool:
    """True when the variant came from a LoRA-adapted branch."""
    return variant.endswith("_lora")


# ---------------------------------------------------------------------------
# 4. Shape helpers
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


def expect_float(tensor: torch.Tensor, name: str) -> None:
    if not tensor.is_floating_point():
        raise TypeError(f"{name}: expected float tensor, got {tensor.dtype}")


def expect_unit_range(tensor: torch.Tensor, name: str) -> None:
    if tensor.numel() == 0:
        return
    lo, hi = float(tensor.min()), float(tensor.max())
    if lo < 0.0 or hi > 1.0:
        raise ValueError(
            f"{name}: expected values in [0, 1] before backbone normalization, "
            f"got [{lo:.4f}, {hi:.4f}]"
        )


def expect_binary_label(tensor: torch.Tensor, name: str) -> None:
    expect_shape(tensor, (None,), name)
    expect_float(tensor, name)
    if tensor.numel() == 0:
        return
    unique = torch.unique(tensor)
    bad = [float(v) for v in unique if float(v) not in (0.0, 1.0)]
    if bad:
        raise ValueError(f"{name}: expected binary labels {{0.0, 1.0}}, found {bad}")


# ---------------------------------------------------------------------------
# 5. ManifestRow - written by src/data/build_manifest.py
# ---------------------------------------------------------------------------


@dataclass
class ManifestRow:
    """One DFDC video selected for the 1500-video subset."""

    video_id: str
    filepath: str
    label: int  # 0 real, 1 fake
    split: str  # train | val | test
    source_group_id: str  # group of a fake and its originals, split atomically
    original_id: str | None = None
    clip_start_sec: float = 0.0

    def validate(self) -> "ManifestRow":
        if not self.video_id:
            raise ValueError("ManifestRow.video_id: must be non-empty")
        if self.label not in LABEL_NAMES:
            raise ValueError(f"ManifestRow.label: expected 0 or 1, got {self.label!r}")
        if self.split not in SPLITS:
            raise ValueError(f"ManifestRow.split: expected one of {SPLITS}, got {self.split!r}")
        if not self.source_group_id:
            raise ValueError(
                "ManifestRow.source_group_id: required - splits are assigned per "
                "source group, never per video"
            )
        if self.clip_start_sec < 0.0:
            raise ValueError(f"ManifestRow.clip_start_sec: must be >= 0, got {self.clip_start_sec}")
        return self


# ---------------------------------------------------------------------------
# 6. ClipBatch - yielded by src/data/video_dataset.py (clip mode)
# ---------------------------------------------------------------------------


@dataclass
class ClipBatch:
    """Raw clip batch for Stage I branch training.

    Frames are float32 in [0.0, 1.0]; normalization is the branch wrapper's job.
    Either branch tensor may be None for single-branch training (A1-A4).
    """

    label: torch.Tensor  # (B,) float32 in {0.0, 1.0}
    dino_frames: torch.Tensor | None = None  # (B, 4, 3, 224, 224)
    vjepa_frames: torch.Tensor | None = None  # (B, 3, 16, 384, 384)
    video_id: tuple[str, ...] = field(default_factory=tuple)

    @property
    def batch_size(self) -> int:
        anchor = self.dino_frames if self.dino_frames is not None else self.vjepa_frames
        if anchor is None:
            raise ValueError("ClipBatch: both dino_frames and vjepa_frames are None")
        return int(anchor.shape[0])

    def validate(self, *, check_range: bool = True) -> "ClipBatch":
        if self.dino_frames is None and self.vjepa_frames is None:
            raise ValueError("ClipBatch: at least one branch tensor must be set")

        if self.dino_frames is not None:
            expect_shape(
                self.dino_frames,
                (None, DINO_FRAMES, 3, DINO_SIZE, DINO_SIZE),
                "ClipBatch.dino_frames",
            )
            expect_float(self.dino_frames, "ClipBatch.dino_frames")
            if check_range:
                expect_unit_range(self.dino_frames, "ClipBatch.dino_frames")

        if self.vjepa_frames is not None:
            expect_shape(
                self.vjepa_frames,
                (None, 3, VJEPA_FRAMES, VJEPA_SIZE, VJEPA_SIZE),
                "ClipBatch.vjepa_frames",
            )
            expect_float(self.vjepa_frames, "ClipBatch.vjepa_frames")
            if check_range:
                expect_unit_range(self.vjepa_frames, "ClipBatch.vjepa_frames")

        expect_binary_label(self.label, "ClipBatch.label")
        if self.label.shape[0] != self.batch_size:
            raise ValueError(
                f"ClipBatch.label: batch mismatch ({self.label.shape[0]} labels vs "
                f"{self.batch_size} clips)"
            )

        if self.video_id and len(self.video_id) != self.batch_size:
            raise ValueError(
                f"ClipBatch.video_id: expected {self.batch_size} ids, got {len(self.video_id)}"
            )
        return self

    def to(self, device: torch.device | str, non_blocking: bool = True) -> "ClipBatch":
        def move(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.to(device, non_blocking=non_blocking)

        return ClipBatch(
            label=move(self.label),
            dino_frames=move(self.dino_frames),
            vjepa_frames=move(self.vjepa_frames),
            video_id=self.video_id,
        )


# ---------------------------------------------------------------------------
# 7. BranchFeatures - produced by dino_branch.py and vjepa_branch.py
# ---------------------------------------------------------------------------


@dataclass
class BranchFeatures:
    """Pooled per-video embedding from a single branch."""

    z: torch.Tensor  # (B, 384) for dino, (B, 768) for vjepa
    branch: str  # 'dino' | 'vjepa'

    def validate(self) -> "BranchFeatures":
        if self.branch not in FEATURE_DIM:
            raise ValueError(
                f"BranchFeatures.branch: expected one of {BRANCHES}, got {self.branch!r}"
            )
        expect_shape(self.z, (None, FEATURE_DIM[self.branch]), "BranchFeatures.z")
        expect_float(self.z, "BranchFeatures.z")
        return self

    def to(self, device: torch.device | str, non_blocking: bool = True) -> "BranchFeatures":
        return BranchFeatures(z=self.z.to(device, non_blocking=non_blocking), branch=self.branch)


# ---------------------------------------------------------------------------
# 8. FusionInput - yielded by src/data/video_dataset.py (feature mode)
# ---------------------------------------------------------------------------


@dataclass
class FusionInput:
    """Cached embeddings for Stage II. Either branch may be None (A1-A4)."""

    label: torch.Tensor  # (B,) float32 in {0.0, 1.0}
    dino: torch.Tensor | None = None  # (B, 384)
    vjepa: torch.Tensor | None = None  # (B, 768)
    video_id: tuple[str, ...] = field(default_factory=tuple)

    @property
    def batch_size(self) -> int:
        anchor = self.dino if self.dino is not None else self.vjepa
        if anchor is None:
            raise ValueError("FusionInput: both dino and vjepa are None")
        return int(anchor.shape[0])

    def validate(self) -> "FusionInput":
        if self.dino is None and self.vjepa is None:
            raise ValueError("FusionInput: at least one branch embedding must be set")
        expect_binary_label(self.label, "FusionInput.label")

        if self.dino is not None:
            expect_shape(self.dino, (None, DINO_DIM), "FusionInput.dino")
            expect_float(self.dino, "FusionInput.dino")
            if self.dino.shape[0] != self.label.shape[0]:
                raise ValueError("FusionInput: dino batch size does not match label")
        if self.vjepa is not None:
            expect_shape(self.vjepa, (None, VJEPA_DIM), "FusionInput.vjepa")
            expect_float(self.vjepa, "FusionInput.vjepa")
            if self.vjepa.shape[0] != self.label.shape[0]:
                raise ValueError("FusionInput: vjepa batch size does not match label")
        if self.dino is not None and self.vjepa is not None:
            if self.dino.shape[0] != self.vjepa.shape[0]:
                raise ValueError(
                    "FusionInput: branch batch sizes differ "
                    f"({self.dino.shape[0]} vs {self.vjepa.shape[0]})"
                )
        return self

    def to(self, device: torch.device | str, non_blocking: bool = True) -> "FusionInput":
        def move(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.to(device, non_blocking=non_blocking)

        return FusionInput(
            label=move(self.label),
            dino=move(self.dino),
            vjepa=move(self.vjepa),
            video_id=self.video_id,
        )


# ---------------------------------------------------------------------------
# 9. STAFNetOutput - produced by src/models/stafnet.py
# ---------------------------------------------------------------------------


@dataclass
class STAFNetOutput:
    """Binary logits plus the fusion gate for analysis. Logit is pre-sigmoid."""

    logit: torch.Tensor  # (B, 1)
    gate: torch.Tensor | None = None  # (B, 256) gated fusion only

    def validate(self) -> "STAFNetOutput":
        expect_shape(self.logit, (None, NUM_LOGITS), "STAFNetOutput.logit")
        expect_float(self.logit, "STAFNetOutput.logit")
        if self.gate is not None:
            expect_shape(self.gate, (None, D_PROJ), "STAFNetOutput.gate")
            expect_float(self.gate, "STAFNetOutput.gate")
            if self.gate.numel() > 0:
                lo, hi = float(self.gate.min()), float(self.gate.max())
                if lo < 0.0 or hi > 1.0:
                    raise ValueError(
                        f"STAFNetOutput.gate: expected sigmoid output in [0, 1], got [{lo:.4f}, {hi:.4f}]"
                    )
            if self.gate.shape[0] != self.logit.shape[0]:
                raise ValueError("STAFNetOutput: gate batch size does not match logit")
        return self

    def to(self, device: torch.device | str, non_blocking: bool = True) -> "STAFNetOutput":
        def move(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.to(device, non_blocking=non_blocking)

        return STAFNetOutput(logit=move(self.logit), gate=move(self.gate))


# ---------------------------------------------------------------------------
# 10. FeatureRecord - one row of the cached feature store
# ---------------------------------------------------------------------------


@dataclass
class FeatureRecord:
    """One video's cached embeddings, keyed by variant name."""

    video_id: str
    label: int
    split: str
    source_group_id: str
    features: dict[str, torch.Tensor] = field(default_factory=dict)

    def validate(self, required: Sequence[str] = VARIANTS) -> "FeatureRecord":
        if self.label not in LABEL_NAMES:
            raise ValueError(f"FeatureRecord.label: expected 0 or 1, got {self.label!r}")
        if self.split not in SPLITS:
            raise ValueError(f"FeatureRecord.split: expected one of {SPLITS}, got {self.split!r}")
        for variant in required:
            if variant not in self.features:
                raise ValueError(
                    f"FeatureRecord[{self.video_id}]: missing variant {variant!r} "
                    f"(have {sorted(self.features)})"
                )
        for variant, tensor in self.features.items():
            expect_shape(tensor, (variant_dim(variant),), f"FeatureRecord[{self.video_id}].{variant}")
            expect_float(tensor, f"FeatureRecord[{self.video_id}].{variant}")
        return self


# ---------------------------------------------------------------------------
# 11. Introspection
# ---------------------------------------------------------------------------


def contract_summary() -> dict[str, object]:
    """Flat dict logged at the start of every run."""
    return {
        "contract_version": CONTRACT_VERSION,
        "task": "binary real/fake (DFDC)",
        "dino_frames": DINO_FRAMES,
        "dino_size": DINO_SIZE,
        "dino_dim": DINO_DIM,
        "vjepa_frames": VJEPA_FRAMES,
        "vjepa_size": VJEPA_SIZE,
        "vjepa_dim": VJEPA_DIM,
        "d_proj": D_PROJ,
        "head_hidden": HEAD_HIDDEN,
        "dropout": DROPOUT,
        "pooling": POOLING,
        "variants": list(VARIANTS),
        "splits": list(SPLITS),
        "labels": LABEL_NAMES,
    }


__all__ = [
    "CONTRACT_VERSION",
    "DINO_FRAMES",
    "DINO_SIZE",
    "DINO_DIM",
    "VJEPA_FRAMES",
    "VJEPA_SIZE",
    "VJEPA_DIM",
    "D_PROJ",
    "HEAD_HIDDEN",
    "DROPOUT",
    "NUM_LOGITS",
    "BRANCHES",
    "FEATURE_DIM",
    "VARIANTS",
    "SPLITS",
    "LABEL_NAMES",
    "POOLING",
    "ManifestRow",
    "ClipBatch",
    "BranchFeatures",
    "FusionInput",
    "STAFNetOutput",
    "FeatureRecord",
    "expect_shape",
    "expect_float",
    "expect_unit_range",
    "expect_binary_label",
    "variant_branch",
    "variant_dim",
    "variant_is_lora",
    "contract_summary",
]
