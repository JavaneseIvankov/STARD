"""STAF-Net | src/train_fusion.py

Owner   : Person 3 (Fusion / evaluation)
Status  : IMPLEMENTED - Stage II training & full A1-A8 ablation matrix driver.

Usage
-----
    python -m src.train_fusion --config configs/fusion.yaml --experiment all
    python -m src.train_fusion --config configs/fusion.yaml --experiment A8
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.contracts import DROPOUT, D_PROJ, HEAD_HIDDEN
from src.data.video_dataset import FeatureDataset, feature_collate, read_manifest
from src.models.stafnet import STAFNet
from src.utils.metrics import (
    best_threshold_from_val,
    binary_metrics,
    confusion_matrix,
    format_ablation_markdown,
    group_bootstrap_auroc,
)

__all__ = ["EXPERIMENTS", "ExperimentSpec", "run_experiment", "run_ablations"]


@dataclass(frozen=True)
class ExperimentSpec:
    exp_id: str
    label: str
    dino_variant: str | None
    vjepa_variant: str | None
    mode: str


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "A1": ExperimentSpec("A1", "A1 Frozen DINO", "dino_frozen", None, "single_dino"),
    "A2": ExperimentSpec("A2", "A2 Frozen V-JEPA", None, "vjepa_frozen", "single_vjepa"),
    "A3": ExperimentSpec("A3", "A3 DINO LoRA", "dino_lora", None, "single_dino"),
    "A4": ExperimentSpec("A4", "A4 V-JEPA LoRA", None, "vjepa_lora", "single_vjepa"),
    "A5": ExperimentSpec("A5", "A5 Frozen concat", "dino_frozen", "vjepa_frozen", "concat"),
    "A6": ExperimentSpec("A6", "A6 Frozen gated", "dino_frozen", "vjepa_frozen", "gated"),
    "A7": ExperimentSpec("A7", "A7 Adapted concat", "dino_lora", "vjepa_lora", "concat"),
    "A8": ExperimentSpec("A8", "A8 STAF-Net (LoRA gated)", "dino_lora", "vjepa_lora", "gated"),
}


@torch.no_grad()
def _collect_predictions(
    model: STAFNet, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor | None]:
    model.eval()
    scores_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    video_ids: list[str] = []
    gates_list: list[torch.Tensor] = []

    for batch in loader:
        batch_dev = batch.to(device)
        out = model(batch_dev)
        probs = torch.sigmoid(out.logit.float().cpu()).squeeze(1)
        scores_list.append(probs)
        labels_list.append(batch.label.cpu())
        video_ids.extend(batch.video_id)
        if out.gate is not None:
            gates_list.append(out.gate.float().cpu())

    gates = torch.cat(gates_list, dim=0) if gates_list else None
    return torch.cat(scores_list), torch.cat(labels_list), video_ids, gates


def run_experiment(
    exp_id: str,
    *,
    manifest: str | Path = "data/manifests/manifest.csv",
    features_dir: str | Path = "data/features",
    checkpoints_dir: str | Path = "checkpoints",
    metrics_dir: str | Path = "results/metrics",
    d_proj: int = D_PROJ,
    hidden_dim: int = HEAD_HIDDEN,
    dropout: float = DROPOUT,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    batch_size: int = 64,
    max_epochs: int = 30,
    patience: int = 5,
    seed: int = 42,
) -> dict[str, object]:
    """Train one Stage II ablation (A1..A8), pick threshold on val, eval once on test."""
    if exp_id not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {exp_id!r}, expected one of {sorted(EXPERIMENTS)}")
    spec = EXPERIMENTS[exp_id]

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = read_manifest(manifest)
    train_rows = [r for r in rows if r.split == "train"]
    val_rows = [r for r in rows if r.split == "val"]
    test_rows = [r for r in rows if r.split == "test"]
    group_by_vid = {r.video_id: r.source_group_id for r in rows}

    def _loader(split_rows, shuffle: bool) -> DataLoader:
        ds = FeatureDataset(
            rows=split_rows,
            features_dir=features_dir,
            dino_variant=spec.dino_variant,
            vjepa_variant=spec.vjepa_variant,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=feature_collate,
        )

    train_loader = _loader(train_rows, shuffle=True)
    val_loader = _loader(val_rows, shuffle=False)
    test_loader = _loader(test_rows, shuffle=False) if test_rows else None

    model = STAFNet(
        mode=spec.mode,
        d_proj=d_proj,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_auroc = -math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float]] = []

    t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            batch_dev = batch.to(device)
            labels = batch_dev.label.unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch_dev)
            loss = criterion(out.logit, labels)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
            seen += 1

        val_scores, val_labels, _, _ = _collect_predictions(model, val_loader, device)
        val_m = binary_metrics(val_scores, val_labels, threshold=0.5)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": running / max(seen, 1),
                "val_auroc": val_m["auroc"],
                "val_f1": val_m["f1"],
            }
        )

        if val_m["auroc"] > best_auroc:
            best_auroc = val_m["auroc"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    train_seconds = time.perf_counter() - t0
    assert best_state is not None
    model.load_state_dict(best_state)

    # Validation threshold selection (never on test!)
    val_scores, val_labels, _, _ = _collect_predictions(model, val_loader, device)
    val_threshold = best_threshold_from_val(val_scores, val_labels)
    val_metrics = binary_metrics(val_scores, val_labels, threshold=val_threshold)

    ckpt_dir = Path(checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"fusion_{exp_id}.pt"
    torch.save(
        {
            "experiment": exp_id,
            "label": spec.label,
            "mode": spec.mode,
            "dino_variant": spec.dino_variant,
            "vjepa_variant": spec.vjepa_variant,
            "model_kwargs": {
                "mode": spec.mode,
                "d_proj": d_proj,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
            },
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "val_threshold": val_threshold,
            "val_metrics": val_metrics,
            "train_seconds": round(train_seconds, 2),
        },
        ckpt_path,
    )

    result: dict[str, object] = {
        "experiment": exp_id,
        "label": spec.label,
        "mode": spec.mode,
        "dino_variant": spec.dino_variant,
        "vjepa_variant": spec.vjepa_variant,
        "checkpoint": str(ckpt_path),
        "best_epoch": best_epoch,
        "train_seconds": round(train_seconds, 2),
        "val_threshold": val_threshold,
        "val_metrics": val_metrics,
        "history": history,
    }

    if test_loader is not None:
        test_scores, test_labels, test_vids, test_gates = _collect_predictions(model, test_loader, device)
        test_metrics = binary_metrics(test_scores, test_labels, threshold=val_threshold)
        test_cm = confusion_matrix(test_scores, test_labels, threshold=val_threshold)
        test_groups = [group_by_vid[v] for v in test_vids]
        boot = group_bootstrap_auroc(test_scores, test_labels, test_groups, seed=seed)
        result["test_metrics"] = test_metrics
        result["test_confusion"] = test_cm
        result["test_auroc_bootstrap_ci"] = boot
        if test_gates is not None:
            real_mask = test_labels == 0
            fake_mask = test_labels == 1
            result["gate_stats"] = {
                "overall_mean": float(test_gates.mean()),
                "overall_std": float(test_gates.std()),
                "real_mean": float(test_gates[real_mask].mean()) if real_mask.any() else None,
                "fake_mean": float(test_gates[fake_mask].mean()) if fake_mask.any() else None,
            }

    m_dir = Path(metrics_dir)
    m_dir.mkdir(parents=True, exist_ok=True)
    (m_dir / f"{exp_id}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_ablations(
    experiments: list[str],
    **kwargs,
) -> list[dict[str, object]]:
    """Run a list of ablations (e.g. A1..A8) and write summary JSON + Markdown table."""
    results: list[dict[str, object]] = []
    table_rows: list[dict[str, object]] = []
    for exp_id in experiments:
        res = run_experiment(exp_id, **kwargs)
        results.append(res)
        tm = res.get("test_metrics") or res["val_metrics"]
        table_rows.append(
            {
                "experiment": exp_id,
                "label": res["label"],
                "auroc": tm["auroc"],
                "f1": tm["f1"],
                "accuracy": tm["accuracy"],
                "eer": tm["eer"],
                "threshold": res["val_threshold"],
                "train_seconds": res["train_seconds"],
            }
        )
        print(
            f"[{exp_id}] val_auroc={res['val_metrics']['auroc']:.4f} "
            f"test_auroc={tm['auroc']:.4f} f1={tm['f1']:.4f} acc={tm['accuracy']:.4f} "
            f"eer={tm['eer']:.4f} ({res['train_seconds']:.1f}s)",
            flush=True,
        )

    m_dir = Path(kwargs.get("metrics_dir", "results/metrics"))
    m_dir.mkdir(parents=True, exist_ok=True)
    md = format_ablation_markdown(table_rows)
    (m_dir / "ablation_table.md").write_text(md, encoding="utf-8")
    (m_dir / "ablation_summary.json").write_text(json.dumps(table_rows, indent=2), encoding="utf-8")
    print("\n" + md)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage II: STAF-Net fusion & A1-A8 ablations.")
    parser.add_argument("--config", default="configs/fusion.yaml")
    parser.add_argument("--experiment", default="all", help="A1..A8, comma-separated, or 'all'")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--features-dir", default=None)
    parser.add_argument("--checkpoints-dir", default=None)
    parser.add_argument("--metrics-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    y = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) if Path(args.config).exists() else {}
    y_model = y.get("model") or {}
    y_train = y.get("train") or {}
    y_out = y.get("output") or {}

    if args.experiment.lower() == "all":
        exp_list = list(EXPERIMENTS.keys())
    else:
        exp_list = [e.strip().upper() for e in args.experiment.split(",") if e.strip()]

    run_ablations(
        exp_list,
        manifest=args.manifest or y.get("manifest") or "data/manifests/manifest.csv",
        features_dir=args.features_dir or y.get("features_dir") or "data/features",
        checkpoints_dir=args.checkpoints_dir or y_out.get("checkpoints_dir") or "checkpoints",
        metrics_dir=args.metrics_dir or y_out.get("metrics_dir") or "results/metrics",
        d_proj=int(y_model.get("d_proj", D_PROJ)),
        hidden_dim=int(y_model.get("hidden_dim", HEAD_HIDDEN)),
        dropout=float(y_model.get("dropout", DROPOUT)),
        lr=float(y_train.get("lr", 1e-3)),
        weight_decay=float(y_train.get("weight_decay", 0.01)),
        batch_size=int(y_train.get("batch_size", 64)),
        max_epochs=int(y_train.get("max_epochs", 30)),
        patience=int(y_train.get("early_stopping_patience", 5)),
        seed=args.seed if args.seed is not None else int(y.get("seed", 42)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
