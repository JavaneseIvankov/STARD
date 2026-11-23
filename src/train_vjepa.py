"""STAF-Net | src/train_vjepa.py

Owner   : Person 2 (Video branch)
Status  : IMPLEMENTED - Stage I supervised training for V-JEPA 2.1 ViT-B/16 (384).

Usage
-----
    python -m src.train_vjepa --frozen --epochs 1 --weights ~/weights/vjepa2.1-vitb16/hf
    python -m src.train_vjepa --epochs 2 --weights ~/weights/vjepa2.1-vitb16/hf
    python -m src.train_vjepa --config configs/vjepa_lora.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.video_dataset import ClipDataset, clip_collate, read_manifest
from src.models.vjepa_branch import VJepaBranch
from src.utils.metrics import binary_metrics

__all__ = ["TrainConfig", "train"]


class TrainConfig:
    """Stage I V-JEPA 2.1 hyperparameters (plan §6.2)."""

    def __init__(
        self,
        *,
        manifest: str = "data/manifests/manifest.csv",
        clips_dir: str = "data/clips",
        weights: str | None = None,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_targets: tuple[str, ...] | None = None,
        lora_lr: float = 5e-5,
        head_lr: float = 5e-4,
        weight_decay: float = 0.01,
        batch_size: int = 2,
        grad_accum: int = 8,
        epochs: int = 1,
        amp_dtype: str = "bf16",
        gradient_checkpointing: bool = True,
        num_workers: int = 4,
        seed: int = 42,
        benchmark_steps: int = 0,
        patience: int = 2,
        pretrained: bool = True,
        out_dir: str = "checkpoints",
        metrics_dir: str = "results/metrics",
    ) -> None:
        self.manifest = manifest
        self.clips_dir = clips_dir
        self.weights = weights
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_targets = lora_targets
        self.lora_lr = lora_lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.grad_accum = max(1, grad_accum)
        self.epochs = epochs
        self.amp_dtype = amp_dtype
        self.gradient_checkpointing = gradient_checkpointing
        self.num_workers = num_workers
        self.seed = seed
        self.benchmark_steps = benchmark_steps
        self.patience = patience
        self.pretrained = pretrained
        self.out_dir = out_dir
        self.metrics_dir = metrics_dir

    @property
    def mode(self) -> str:
        return "lora" if self.use_lora else "frozen"

    @property
    def variant(self) -> str:
        return f"vjepa_{self.mode}"


def _amp_context(cfg: TrainConfig, device: torch.device):
    if device.type != "cuda" or cfg.amp_dtype == "none":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _make_loader(rows, cfg: TrainConfig, *, shuffle: bool) -> DataLoader:
    dataset = ClipDataset(
        rows=rows,
        clips_dir=cfg.clips_dir,
        train=shuffle,
        emit_dino=False,
        emit_vjepa=True,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=clip_collate,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def _validate(
    model: VJepaBranch, loader: DataLoader, device: torch.device, cfg: TrainConfig
) -> dict[str, float]:
    model.eval()
    scores: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch in loader:
        frames = batch.vjepa_frames.to(device, non_blocking=True)
        with _amp_context(cfg, device):
            logits = model.predict_logit(frames)
        scores.append(torch.sigmoid(logits.float().cpu()).squeeze(1))
        labels.append(batch.label.cpu())
    return binary_metrics(torch.cat(scores), torch.cat(labels))


def train(cfg: TrainConfig) -> dict[str, object]:
    """Run Stage I V-JEPA 2.1 training and save the best-val-AUROC checkpoint."""
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = read_manifest(cfg.manifest)
    train_rows = [r for r in rows if r.split == "train"]
    val_rows = [r for r in rows if r.split == "val"]
    if not train_rows or not val_rows:
        raise ValueError(
            f"train_vjepa: manifest needs train and val rows, got "
            f"{len(train_rows)} train / {len(val_rows)} val"
        )

    model = VJepaBranch(
        pretrained_name=cfg.weights,
        pretrained=cfg.pretrained,
        use_lora=cfg.use_lora,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_targets=cfg.lora_targets,
        gradient_checkpointing=cfg.gradient_checkpointing and cfg.use_lora,
    ).to(device)
    print(model.describe(), flush=True)

    train_loader = _make_loader(train_rows, cfg, shuffle=True)
    val_loader = _make_loader(val_rows, cfg, shuffle=False)

    groups = model.lora_parameter_groups()
    optimizer_groups = [
        {"params": g["params"], "lr": cfg.lora_lr if g["lr_scale"] == "lora" else cfg.head_lr}
        for g in groups
    ]
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"vjepa_{cfg.mode}.pt"

    history: list[dict[str, object]] = []
    best_auroc = -math.inf
    best_epoch = -1
    stale = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        model.backbone.eval()
        started = time.perf_counter()
        running = 0.0
        seen = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            frames = batch.vjepa_frames.to(device, non_blocking=True)
            labels = batch.label.to(device, non_blocking=True).unsqueeze(1)

            with _amp_context(cfg, device):
                logits = model.predict_logit(frames)
                loss = criterion(logits, labels) / cfg.grad_accum
            loss.backward()

            if step % cfg.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running += float(loss.detach()) * cfg.grad_accum
            seen += 1

            if cfg.benchmark_steps and step == cfg.benchmark_steps:
                elapsed = time.perf_counter() - started
                per_step = elapsed / step
                projected = per_step * math.ceil(len(train_rows) / cfg.batch_size) * cfg.epochs
                print(
                    f"benchmark: {per_step * 1000:.1f} ms/microstep over {step} steps -> "
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
        print(json.dumps(record), flush=True)

        if metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "branch": "vjepa",
                    "mode": cfg.mode,
                    "weights": cfg.weights,
                    "lora": {
                        "rank": cfg.lora_rank,
                        "alpha": cfg.lora_alpha,
                        "dropout": cfg.lora_dropout,
                        "targets": list(cfg.lora_targets or model.lora_targets),
                    },
                    "epoch": epoch,
                    "val_metrics": metrics,
                },
                ckpt_path,
            )
        else:
            stale += 1
            if stale >= cfg.patience:
                print(f"early stop at epoch {epoch} (best {best_epoch})", flush=True)
                break

    payload: dict[str, object] = {
        "branch": "vjepa",
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
    (metrics_dir / f"train_vjepa_{cfg.mode}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def _load_yaml_defaults(path: str | None) -> dict:
    if not path or not Path(path).exists():
        return {}
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage I training: V-JEPA 2.1 video branch.")
    parser.add_argument("--config", default=None, help="optional YAML config (e.g. configs/vjepa_lora.yaml)")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--clips-dir", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--frozen", action="store_true", help="train head only (A2)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-targets", default=None)
    parser.add_argument("--lora-lr", type=float, default=None)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--amp", default=None, choices=("bf16", "fp16", "none"))
    parser.add_argument("--no-grad-ckpt", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--benchmark-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--untrained-backbone", action="store_true")
    args = parser.parse_args(argv)

    y = _load_yaml_defaults(args.config)
    y_lora = y.get("lora") or {}
    y_train = y.get("train") or {}
    y_out = y.get("output") or {}

    manifest = args.manifest or y.get("manifest") or "data/manifests/manifest.csv"
    if not Path(manifest).exists():
        print(f"[wait] manifest not found: {manifest}")
        return 2

    raw_targets = args.lora_targets
    if raw_targets:
        targets = tuple(raw_targets.split(","))
    elif y_lora.get("targets"):
        targets = tuple(y_lora["targets"])
    else:
        targets = None

    cfg = TrainConfig(
        manifest=manifest,
        clips_dir=args.clips_dir or y.get("clips_dir") or "data/clips",
        weights=args.weights or y.get("weights"),
        use_lora=not args.frozen,
        lora_rank=args.lora_rank or int(y_lora.get("rank", 8)),
        lora_alpha=args.lora_alpha or int(y_lora.get("alpha", 16)),
        lora_dropout=float(y_lora.get("dropout", 0.05)),
        lora_targets=targets,
        lora_lr=args.lora_lr or float(y_train.get("lora_lr", 5e-5)),
        head_lr=args.head_lr or float(y_train.get("head_lr", 5e-4)),
        weight_decay=float(y_train.get("weight_decay", 0.01)),
        batch_size=args.batch_size or int(y_train.get("batch_size", 2)),
        grad_accum=args.grad_accum or int(y_train.get("grad_accum", 8)),
        epochs=args.epochs or int(y_train.get("epochs", 1)),
        amp_dtype=args.amp or str(y_train.get("amp_dtype", "bf16")),
        gradient_checkpointing=(not args.no_grad_ckpt) and bool(y_train.get("gradient_checkpointing", True)),
        num_workers=args.num_workers if args.num_workers is not None else int(y_train.get("num_workers", 4)),
        seed=args.seed if args.seed is not None else int(y.get("seed", 42)),
        benchmark_steps=args.benchmark_steps,
        patience=int(y_train.get("patience", 2)),
        pretrained=not args.untrained_backbone,
        out_dir=str(y_out.get("checkpoints_dir", "checkpoints")),
        metrics_dir=str(y_out.get("metrics_dir", "results/metrics")),
    )
    payload = train(cfg)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
