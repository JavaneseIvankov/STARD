"""STAF-Net | src/extract_features.py

Owner   : Person 1 + Person 2 (shared)
Status  : IMPLEMENTED - extracts pooled per-video embeddings for train/val/test
          according to docs/cache_format.md v0.2.

Usage
-----
    python -m src.extract_features --variant dino_frozen --weights ~/weights/dinov3-vits16
    python -m src.extract_features --variant dino_lora --weights ~/weights/dinov3-vits16 --checkpoint checkpoints/dino_lora.pt
    python -m src.extract_features --variant vjepa_frozen --weights ~/weights/vjepa2.1-vitb16/hf
    python -m src.extract_features --variant vjepa_lora --weights ~/weights/vjepa2.1-vitb16/hf --checkpoint checkpoints/vjepa_lora.pt
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.contracts import (
    DINO_FRAMES,
    DINO_SIZE,
    POOLING,
    SPLITS,
    VARIANTS,
    VJEPA_FRAMES,
    VJEPA_SIZE,
    expect_shape,
    variant_branch,
    variant_dim,
    variant_is_lora,
)
from src.data.video_dataset import ClipDataset, clip_collate, read_manifest
from src.models.dino_branch import DinoBranch
from src.models.vjepa_branch import VJepaBranch

__all__ = ["extract_variant"]

FORMAT_VERSION = "0.2"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _load_branch(
    variant: str,
    *,
    weights: str | None,
    checkpoint: str | None,
    pretrained: bool = True,
) -> torch.nn.Module:
    branch = variant_branch(variant)
    use_lora = variant_is_lora(variant)

    ckpt_payload = None
    lora_rank, lora_alpha, lora_dropout, lora_targets = 8, 16, 0.05, None
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"extract_features: checkpoint not found: {ckpt_path}")
        ckpt_payload = torch.load(ckpt_path, map_location="cpu")
        lora_cfg = ckpt_payload.get("lora") or {}
        lora_rank = int(lora_cfg.get("rank", 8))
        lora_alpha = int(lora_cfg.get("alpha", 16))
        lora_dropout = float(lora_cfg.get("dropout", 0.05))
        if lora_cfg.get("targets"):
            lora_targets = tuple(lora_cfg["targets"])

    if branch == "dino":
        model: torch.nn.Module = DinoBranch(
            pretrained_name=weights,
            pretrained=pretrained,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_targets=lora_targets,
        )
    else:
        model = VJepaBranch(
            pretrained_name=weights,
            pretrained=pretrained,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_targets=lora_targets,
        )

    if ckpt_payload is not None:
        state = ckpt_payload.get("state_dict", ckpt_payload)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"[warn] load_state_dict({checkpoint}): missing={missing[:5]} unexpected={unexpected[:5]}",
                flush=True,
            )
    return model


@torch.no_grad()
def extract_variant(
    variant: str,
    *,
    manifest: str | Path = "data/manifests/manifest.csv",
    clips_dir: str | Path = "data/clips",
    out_root: str | Path = "data/features",
    weights: str | None = None,
    checkpoint: str | None = None,
    batch_size: int = 8,
    num_workers: int = 4,
    shard_size: int = 500,
    amp_dtype: str = "bf16",
    preprocessing_version: str = "clips-v1",
    pretrained: bool = True,
    overwrite: bool = False,
) -> Path:
    """Extract and cache pooled video embeddings across train/val/test."""
    branch = variant_branch(variant)
    dim = variant_dim(variant)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(out_root) / variant
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_id = f"{branch}+{variant}" + (f"@{Path(checkpoint).name}" if checkpoint else "")

    rows = read_manifest(manifest)
    by_split = {sp: [r for r in rows if r.split == sp] for sp in SPLITS}

    model = _load_branch(variant, weights=weights, checkpoint=checkpoint, pretrained=pretrained).to(device)
    model.eval()

    use_amp = device.type == "cuda" and amp_dtype != "none"
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

    index_rows: list[dict[str, object]] = []
    for split in SPLITS:
        split_rows = by_split[split]
        if not split_rows:
            continue
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        row_lookup = {r.video_id: r for r in split_rows}

        for shard_idx, start in enumerate(range(0, len(split_rows), shard_size)):
            chunk = split_rows[start : start + shard_size]
            shard_rel = f"{split}/shard-{shard_idx:05d}.pt"
            shard_path = root / shard_rel

            # Resume-safe check (docs/cache_format.md invariant #6)
            if shard_path.exists() and not overwrite:
                existing = torch.load(shard_path, map_location="cpu")
                if (
                    existing.get("format_version") == FORMAT_VERSION
                    and existing.get("checkpoint_id") == checkpoint_id
                    and len(existing.get("records", [])) == len(chunk)
                ):
                    for pos, rec in enumerate(existing["records"]):
                        index_rows.append(
                            {
                                "video_id": rec["video_id"],
                                "source_group_id": rec["source_group_id"],
                                "split": split,
                                "shard": shard_rel,
                                "shard_index": pos,
                                "label": int(rec["label"]),
                                "dim": dim,
                            }
                        )
                    print(f"[skip] {variant}/{shard_rel} ({len(chunk)} cached)", flush=True)
                    continue

            ds = ClipDataset(
                rows=chunk,
                clips_dir=clips_dir,
                train=False,
                emit_dino=(branch == "dino"),
                emit_vjepa=(branch == "vjepa"),
            )
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=clip_collate,
                pin_memory=torch.cuda.is_available(),
            )

            records: list[dict[str, object]] = []
            for batch in loader:
                frames = (batch.dino_frames if branch == "dino" else batch.vjepa_frames).to(
                    device, non_blocking=True
                )
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                    z = model.encode(frames).z
                z = z.float().cpu()
                expect_shape(z, (len(batch.video_id), dim), f"extract[{variant}]")
                for i, vid in enumerate(batch.video_id):
                    r = row_lookup[vid]
                    pos = len(records)
                    records.append(
                        {
                            "video_id": vid,
                            "label": int(r.label),
                            "z": z[i],
                            "source_group_id": r.source_group_id,
                        }
                    )
                    index_rows.append(
                        {
                            "video_id": vid,
                            "source_group_id": r.source_group_id,
                            "split": split,
                            "shard": shard_rel,
                            "shard_index": pos,
                            "label": int(r.label),
                            "dim": dim,
                        }
                    )

            torch.save(
                {
                    "format_version": FORMAT_VERSION,
                    "variant": variant,
                    "checkpoint_id": checkpoint_id,
                    "records": records,
                },
                shard_path,
                _use_new_zipfile_serialization=True,
            )
            print(f"[ ok ] {variant}/{shard_rel} ({len(records)} videos)", flush=True)

    # Write index.csv
    index_path = root / "index.csv"
    fieldnames = ["video_id", "source_group_id", "split", "shard", "shard_index", "label", "dim"]
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    # Write MANIFEST.json
    manifest_payload = {
        "format_version": FORMAT_VERSION,
        "variant": variant,
        "backbone": "dinov3-vits16" if branch == "dino" else "vjepa2.1-vitb16",
        "checkpoint_id": checkpoint_id,
        "preprocessing_version": preprocessing_version,
        "num_frames": DINO_FRAMES if branch == "dino" else VJEPA_FRAMES,
        "resolution": DINO_SIZE if branch == "dino" else VJEPA_SIZE,
        "pooling": POOLING,
        "dim": dim,
        "num_records": len(index_rows),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    print(f"[done] {variant}: {len(index_rows)} total vectors -> {root}", flush=True)
    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract pooled video embeddings for Stage II.")
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--manifest", default="data/manifests/manifest.csv")
    parser.add_argument("--clips-dir", default="data/clips")
    parser.add_argument("--out-root", default="data/features")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--amp", default="bf16", choices=("bf16", "fp16", "none"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--untrained-backbone", action="store_true")
    args = parser.parse_args(argv)

    extract_variant(
        args.variant,
        manifest=args.manifest,
        clips_dir=args.clips_dir,
        out_root=args.out_root,
        weights=args.weights,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shard_size=args.shard_size,
        amp_dtype=args.amp,
        pretrained=not args.untrained_backbone,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
