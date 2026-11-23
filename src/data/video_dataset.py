"""STAF-Net | src/data/video_dataset.py

Owner   : Person 1 + Person 2 (shared). Owner of record: Person 1.
Status  : IMPLEMENTED (clip + feature modes) - unit-testable without DFDC by
          passing a synthetic manifest against cached clip tensors.

Responsibility
--------------
- ClipDataset : reads cached per-clip tensors produced by face_tracking.py and
                yields src.contracts.ClipBatch for Stage I branch training.
- FeatureDataset : reads cached embeddings for Stage II fusion / ablations and
                yields src.contracts.FusionInput.

Contract
--------
Clip mode
    dino_frames   (B, 4, 3, 224, 224)   layout (B, T, C, H, W)
    vjepa_frames  (B, 3, 16, 384, 384)  layout (B, C, T, H, W)  <- different
    label         (B,) float32 in {0.0, 1.0}
    video_id      tuple[str, ...]

Feature mode
    dino   (B, 384)  and/or  vjepa (B, 768), label (B,), video_id

Determinism
-----------
val/test use TransformConfig(train=False) and a fixed frame selection; clip
augmentation is only enabled for the train split.

On-disk clip layout
-------------------
    data/clips/<split>/<video_id>.pt   ->  {"frames": (16, 3, H, W) uint8 [0,255],
                                            "clip_start_sec": float,
                                            "preprocessing_version": str}

Storing the 16 decoded crops once means Stage I never decodes video again, and
both branches read the same tensor, which is what makes A1 vs A3 a fair
comparison.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.contracts import (
    DINO_DIM,
    VJEPA_DIM,
    ClipBatch,
    FusionInput,
    ManifestRow,
    variant_dim,
)
from src.data.transforms import (
    TransformConfig,
    build_dino_transform,
    build_vjepa_transform,
    select_dino_frames,
)

__all__ = [
    "read_manifest",
    "ClipDataset",
    "FeatureDataset",
    "clip_collate",
    "feature_collate",
]


def read_manifest(path: str | Path, split: str | None = None) -> list[ManifestRow]:
    """Load manifest.csv, optionally filtered to one split. Every row is validated."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"read_manifest: {path} does not exist. Build it first:\n"
            f"  python -m src.data.build_manifest --config configs/dataset.yaml"
        )
    rows: list[ManifestRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = ManifestRow(
                video_id=raw["video_id"],
                filepath=raw["filepath"],
                label=int(raw["label"]),
                split=raw["split"],
                source_group_id=raw["source_group_id"],
                original_id=raw.get("original_id") or None,
                clip_start_sec=float(raw.get("clip_start_sec") or 0.0),
            ).validate()
            if split is None or row.split == split:
                rows.append(row)
    if not rows:
        raise ValueError(f"read_manifest: no rows for split={split!r} in {path}")
    return rows


@dataclass
class ClipDataset(Dataset):
    """Stage I dataset over cached 16-frame face crops."""

    rows: list[ManifestRow]
    clips_dir: str | Path
    train: bool = False
    emit_dino: bool = True
    emit_vjepa: bool = True
    augmentation: TransformConfig | None = None
    preprocessing_version: str = "clips-v1"

    def __post_init__(self) -> None:
        self.clips_dir = Path(self.clips_dir)
        if not (self.emit_dino or self.emit_vjepa):
            raise ValueError("ClipDataset: at least one branch must be emitted")
        self._dino = build_dino_transform(self.augmentation or TransformConfig(train=self.train))
        self._vjepa = build_vjepa_transform(self.augmentation or TransformConfig(train=self.train))

    def __len__(self) -> int:
        return len(self.rows)

    def _clip_path(self, row: ManifestRow) -> Path:
        return self.clips_dir / row.split / f"{row.video_id}.pt"

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        path = self._clip_path(row)
        if not path.exists():
            raise FileNotFoundError(
                f"ClipDataset: missing clip cache {path}. Run:\n"
                f"  python -m src.data.face_tracking --config configs/dataset.yaml"
            )
        payload = torch.load(path, map_location="cpu")
        frames = payload["frames"]
        if frames.shape[0] != 16:
            raise ValueError(
                f"ClipDataset[{row.video_id}]: expected 16 cached frames, got {frames.shape[0]}"
            )
        if frames.dtype == torch.uint8:
            frames = frames.float().div_(255.0)
        else:
            frames = frames.float()
        frames = frames.clamp(0.0, 1.0)

        item: dict[str, object] = {
            "video_id": row.video_id,
            "label": float(row.label),
        }
        if self.emit_dino:
            item["dino_frames"] = self._dino(select_dino_frames(frames))
        if self.emit_vjepa:
            item["vjepa_frames"] = self._vjepa(frames.permute(1, 0, 2, 3))
        return item


def clip_collate(items: list[dict[str, object]]) -> ClipBatch:
    """Collate per-item dicts into a validated ClipBatch."""
    if not items:
        raise ValueError("clip_collate: empty batch")
    batch = ClipBatch(
        label=torch.tensor([float(it["label"]) for it in items], dtype=torch.float32),
        dino_frames=(
            torch.stack([it["dino_frames"] for it in items])  # type: ignore[arg-type]
            if "dino_frames" in items[0]
            else None
        ),
        vjepa_frames=(
            torch.stack([it["vjepa_frames"] for it in items])  # type: ignore[arg-type]
            if "vjepa_frames" in items[0]
            else None
        ),
        video_id=tuple(str(it["video_id"]) for it in items),
    )
    return batch.validate()


@dataclass
class FeatureDataset(Dataset):
    """Stage II dataset over cached pooled embeddings.

    Reads data/features/<variant>/{index.csv, <split>/shard-*.pt} as specified in
docs/cache_format.md. Pass the variants a given ablation needs (A1 -> dino only,
A8 -> both LoRA variants).
    """

    rows: list[ManifestRow]
    features_dir: str | Path = "data/features"
    dino_variant: str | None = "dino_frozen"
    vjepa_variant: str | None = "vjepa_frozen"

    def __post_init__(self) -> None:
        if self.dino_variant is None and self.vjepa_variant is None:
            raise ValueError("FeatureDataset: select at least one variant")
        self.features_dir = Path(self.features_dir)
        for variant in (self.dino_variant, self.vjepa_variant):
            if variant is None:
                continue
            expected = variant_dim(variant)
            if expected not in (DINO_DIM, VJEPA_DIM):
                raise ValueError(f"FeatureDataset: variant {variant!r} has unknown width {expected}")
        self._cache: dict[tuple[str, str], dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _load_variant(self, variant: str, split: str) -> dict[str, torch.Tensor]:
        key = (variant, split)
        if key in self._cache:
            return self._cache[key]

        root = self.features_dir / variant
        index_path = root / "index.csv"
        if not index_path.exists():
            raise FileNotFoundError(
                f"FeatureDataset: {index_path} missing. Extract the cache first:\n"
                f"  python -m src.extract_features --variant {variant}"
            )

        store: dict[str, torch.Tensor] = {}
        with index_path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                if raw["split"] != split:
                    continue
                shard_path = root / raw["shard"]
                if shard_path.exists():
                    payload = torch.load(shard_path, map_location="cpu")
                    for record in payload["records"]:
                        store[record["video_id"]] = record["z"].float()
        if not store:
            raise ValueError(f"FeatureDataset: no cached features in {root} for split={split}")
        self._cache[key] = store
        return store

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        item: dict[str, object] = {"video_id": row.video_id, "label": float(row.label)}
        if self.dino_variant is not None:
            store = self._load_variant(self.dino_variant, row.split)
            if row.video_id not in store:
                raise KeyError(
                    f"FeatureDataset: {row.video_id} missing from {self.dino_variant}/{row.split}"
                )
            item["dino"] = store[row.video_id]
        if self.vjepa_variant is not None:
            store = self._load_variant(self.vjepa_variant, row.split)
            if row.video_id not in store:
                raise KeyError(
                    f"FeatureDataset: {row.video_id} missing from {self.vjepa_variant}/{row.split}"
                )
            item["vjepa"] = store[row.video_id]
        return item


def feature_collate(items: list[dict[str, object]]) -> FusionInput:
    """Collate per-item dicts into a validated FusionInput."""
    if not items:
        raise ValueError("feature_collate: empty batch")
    batch = FusionInput(
        label=torch.tensor([float(it["label"]) for it in items], dtype=torch.float32),
        dino=torch.stack([it["dino"] for it in items]) if "dino" in items[0] else None,  # type: ignore[arg-type]
        vjepa=torch.stack([it["vjepa"] for it in items]) if "vjepa" in items[0] else None,  # type: ignore[arg-type]
        video_id=tuple(str(it["video_id"]) for it in items),
    )
    return batch.validate()
