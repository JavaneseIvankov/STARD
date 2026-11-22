"""STAF-Net | src/train_dino.py

Owner   : Person 1 (Spatial branch)
Status  : IMPLEMENTED (not yet run on DFDC).

Responsibility
--------------
Stage I training for the DINOv3 spatial branch:

  frozen run  -> backbone frozen, only the temporary head trains (A1)
  LoRA run    -> LoRA adapters + head (A3)

plus export of the pooled z_D vectors for Stage II.

Contract
--------
    python -m src.train_dino --frozen --epochs 1
    python -m src.train_dino --epochs 2 --export-variant dino_lora

Outputs
-------
    checkpoints/dino_{frozen,lora}.pt          state_dict + config + val metrics
    data/features/<variant>/index.csv + shards  (docs/cache_format.md)
    results/metrics/train_dino_<mode>.json

Selection
---------
Best validation AUROC. The test split is touched only by src/evaluate.py after a
checkpoint is frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.video_dataset import ClipDataset, clip_collate, read_manifest
from src.models.dino_branch import DinoBranch
from src.utils.metrics import binary_metrics

__all__ = ["TrainConfig", "train", "export_features"]


class TrainConfig:
    """Stage I hyperparameters (plan 5.2)."""

    def __init__(
        self,
        *,
        manifest: str = "data/manifests/manifest.csv",
        clips_dir: str = "data/clips",
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_targets: tuple[str, ...] = ("q_proj", "v_proj"),
        lora_lr: float = 1e-4,
        head_lr: float = 5e-4,
        weight_decay: float = 0.01,
        batch_size: int = 8,
        epochs: int = 1,
        amp_dtype: str = "bf16",
        num_workers: int = 4,
        seed: int = 42,
        benchmark_steps: int = 0,
        patience: int = 2,
        pretrained: bool = True,
        out_dir: str = "checkpoints",
        features_dir: str = "data/features",
        metrics_dir: str = "results/metrics",
    ) -> None:
        self.manifest = manifest
        self.clips_dir = clips_dir
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_targets = lora_targets
        self.lora_lr = lora_lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.amp_dtype = amp_dtype
        self.num_workers = num_workers
        self.seed = seed
        self.benchmark_steps = benchmark_steps
        self.patience = patience
        self.pretrained = pretrained
        self.out_dir = out_dir
        self.features_dir = features_dir
        self.metrics_dir = metrics_dir

    @property
    def mode(self) -> str:
        return "lora" if self.use_lora else "frozen"

    @property
    def variant(self) -> str:
        return f"dino_{self.mode}"


def _amp_context(cfg: TrainConfig, device: torch.device):
    if device.type != "cuda" or cfg.amp_dtype == "none":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _make_loader(rows, cfg: TrainConfig, *, split: str, shuffle: bool) -> DataLoader:
    dataset = ClipDataset(
        rows=rows,
        clips_dir=cfg.clips_dir,
        train=shuffle,
        emit_dino=True,
        emit_vjepa=False,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=clip_collate,
        pin_memory=True,
        drop_last=False,
    )


@torch.no_grad()
def _validate(model: DinoBranch, loader: DataLoader, device: torch.device, cfg: TrainConfig) -> dict[str, float]:
    model.eval()
    scores: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch in loader:
        frames = batch.dino_frames.to(device, non_blocking=True)
        with _amp_context(cfg, device):
            logits = model.predict_logit(frames)
        scores.append(torch.sigmoid(logits.float().cpu()).squeeze(1))
        labels.append(batch.label.cpu())
    return binary_metrics(torch.cat(scores), torch.cat(labels))


@torch.no_grad()
def export_features(
    model: DinoBranch,
    loader: DataLoader,
    device: torch.device,
    variant: str,
    checkpoint_id: str,
    out_root: str | Path = "data/features",
    preprocessing_version: str = "clips-v1",
    shard_size: int = 500,
) -> Path:
    """Write the pooled z_D cache for Stage II (docs/cache_format.md)."""
    root = Path(out_root) / variant
    root.mkdir(parents=True, exist_ok=True)

    model.eval()
    records: list[dict[str, object]] = []
    for batch in loader:
        frames = batch.dino_frames.to(device, non_blocking=True)
        z = model.encode(frames).z.float().cpu()
        for i, video_id in enumerate(batch.video_id):
            records.append({"video_id": video_id, "z": z[i]})

    shard_rel = f"{variant}/shard-00000.pt"
    torch.save(
        {
            "format_version": "0.2",
            "variant": variant,
            "checkpoint_id": checkpoint_id,
            "records": records,
        },
        root / "shard-00000.pt",
    )
    manifest = {
        "format_version": "0.2",
        "variant": variant,
        "backbone": "dinov3-vits16",
        "checkpoint_id": checkpoint_id,
        "preprocessing_version": preprocessing_version,
        "num_frames": 4,
        "resolution": 224,
        "pooling": "mean",
        "num_records": len(records),
        "shard": shard_rel,
        "shard_size": shard_size,
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def train(cfg: TrainConfig) -> dict[str, object]:
    """Run Stage I training, save the checkpoint, optionally export features."""
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = read_manifest(cfg.manifest)
    train_rows = [r for r in rows if r.split == "train"]
    val_rows = [r for r in rows if r.split == "val"]
    if not train_rows or not val_rows:
        raise ValueError(
            f"train_dino: manifest needs train and val rows, got "
            f"{len(train_rows)} train / {len(val_rows)} val"
        )

    model = DinoBranch(
        pretrained=cfg.pretrained,
        use_lora=cfg.use_lora,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_targets=cfg.lora_targets,
    ).to(device)
    print(model.describe())

    train_loader = _make_loader(train_rows, cfg, split="train", shuffle=True)
    val_loader = _make_loader(val_rows, cfg, split="val", shuffle=False)

    groups = model.lora_parameter_groups()
    optimizer_groups = []
    for group in groups:
        lr = cfg.lora_lr if group["lr_scale"] == "lora" else cfg.head_lr
        optimizer_groups.append({"params": group["params"], "lr": lr})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"dino_{cfg.mode}.pt"

    history: list[dict[str, object]] = []
    best_auroc = -math.inf
    best_epoch = -1
    stale = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        model.backbone.eval()  # keep frozen normalization / dropout behaviour stable
        started = time.perf_counter()
        running = 0.0
        seen = 0

        for step, batch in enumerate(train_loader, start=1):
            frames = batch.dino_frames.to(device, non_blocking=True)
            labels = batch.label.to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            with _amp_context(cfg, device):
                logits = model.predict_logit(frames)
                loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
            optimizer.step()

            running += float(loss.detach())
            seen += 1

            if cfg.benchmark_steps and step == cfg.benchmark_steps:
                elapsed = time.perf_counter() - started
                per_step = elapsed / step
                projected = per_step * math.ceil(len(train_rows) / cfg.batch_size) * cfg.epochs
                print(
                    f"benchmark: {per_step * 1000:.1f} ms/step over {step} steps -> "
                    f"projected {projected / 60:.1f} min for {cfg.epochs} epoch(s)"
                )
                if device.type == "cuda":
                    print(f"peak VRAM: {torch.cuda.max_memory_allocated(device) / 1024 ** 3:.2f} GB")
                raise SystemExit(0)

        epoch_seconds = time.perf_counter() - started
        metrics = _validate(model, val_loader, device, cfg)
        record = {
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "val_auroc": metrics["auroc"],
            "val_f1": metrics["f1"],
            "val_accuracy": metrics["accuracy"],
            "seconds": round(epoch_seconds, 1),
        }
        history.append(record)
        print(json.dumps(record))

        if metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "branch": "dino",
                    "mode": cfg.mode,
                    "lora": {
                        "rank": cfg.lora_rank,
                        "alpha": cfg.lora_alpha,
                        "dropout": cfg.lora_dropout,
                        "targets": list(cfg.lora_targets),
                    },
                    "epoch": epoch,
                    "val_metrics": metrics,
                },
                ckpt_path,
            )
        else:
            stale += 1
            if stale >= cfg.patience:
                print(f"early stop at epoch {epoch} (best {best_epoch})")
                break

    payload: dict[str, object] = {
        "branch": "dino",
        "mode": cfg.mode,
        "checkpoint": str(ckpt_path),
        "best_epoch": best_epoch,
        "best_val_auroc": best_auroc,
        "history": history,
        "device": str(device),
    }

    if device.type == "cuda":
        payload["peak_vram_gb"] = round(torch.cuda.max_memory_allocated(device) / 1024 ** 3, 2)

    metrics_dir = Path(cfg.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"train_dino_{cfg.mode}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    export_loader = _make_loader(train_rows + val_rows, cfg, split="all", shuffle=False)
    root = export_features(
        model,
        export_loader,
        device,
        variant=cfg.variant,
        checkpoint_id=f"dinov3-vits16+{cfg.mode}",
        out_root=cfg.features_dir,
    )
    payload["features"] = str(root)

    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage I training: DINOv3 spatial branch.")
    parser.add_argument("--manifest", default="data/manifests/manifest.csv")
    parser.add_argument("--clips-dir", default="data/clips")
    parser.add_argument("--frozen", action="store_true", help="train the head only (A1)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--amp", default="bf16", choices=("bf16", "fp16", "none"))
    parser.add_argument("--benchmark-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--untrained-backbone",
        action="store_true",
        help="build DINOv3 with random weights (pipeline smoke test only, metrics are meaningless)",
    )
    args = parser.parse_args(argv)

    if not Path(args.manifest).exists():
        print(f"[wait] manifest not found: {args.manifest}")
        print("       run: python -m src.data.build_manifest --metadata <metadata.json>")
        return 2

    cfg = TrainConfig(
        manifest=args.manifest,
        clips_dir=args.clips_dir,
        use_lora=not args.frozen,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_lr=args.lora_lr,
        head_lr=args.head_lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        amp_dtype=args.amp,
        benchmark_steps=args.benchmark_steps,
        seed=args.seed,
        pretrained=not args.untrained_backbone,
    )
    payload = train(cfg)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
