"""STAF-Net | src/evaluate.py

Owner   : Person 3
Status  : IMPLEMENTED - held-out test evaluation, ROC/confusion/gate plots,
          bootstrap 95% CIs, and FP/FN error analysis export.

Usage
-----
    python -m src.evaluate --config configs/fusion.yaml --all --split test
    python -m src.evaluate --config configs/fusion.yaml --checkpoint checkpoints/fusion_A8.pt --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.video_dataset import FeatureDataset, feature_collate, read_manifest  # noqa: E402
from src.models.stafnet import STAFNet  # noqa: E402
from src.train_fusion import EXPERIMENTS, _collect_predictions  # noqa: E402
from src.utils.metrics import (  # noqa: E402
    binary_metrics,
    confusion_matrix,
    format_ablation_markdown,
    group_bootstrap_auroc,
    roc_curve_points,
)

__all__ = ["evaluate_checkpoint"]


def _plot_confusion(cm: dict[str, int], title: str, out_path: Path) -> None:
    mat = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred REAL", "Pred FAKE"])
    ax.set_yticks([0, 1], labels=["True REAL", "True FAKE"])
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(mat[r][c]), ha="center", va="center", fontsize=12, fontweight="bold")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_gate_hist(gates: torch.Tensor, labels: torch.Tensor, title: str, out_path: Path) -> None:
    mean_g = gates.mean(dim=1).numpy()
    lbl = labels.numpy()
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.hist(mean_g[lbl == 0], bins=25, alpha=0.6, label="REAL (y=0)")
    ax.hist(mean_g[lbl == 1], bins=25, alpha=0.6, label="FAKE (y=1)")
    ax.set_xlabel("Mean gate g (higher = more DINOv3, lower = more V-JEPA)")
    ax.set_ylabel("Video count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    manifest: str | Path = "data/manifests/manifest.csv",
    features_dir: str | Path = "data/features",
    split: str = "test",
    metrics_dir: str | Path = "results/metrics",
    plots_dir: str | Path = "results/plots",
    batch_size: int = 64,
    seed: int = 42,
) -> dict[str, object]:
    """Evaluate a saved Stage II fusion checkpoint on the requested split."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    exp_id = str(ckpt["experiment"])
    label = str(ckpt.get("label", exp_id))
    threshold = float(ckpt["val_threshold"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STAFNet(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["state_dict"])

    rows = read_manifest(manifest, split=split)
    group_by_vid = {r.video_id: r.source_group_id for r in rows}
    ds = FeatureDataset(
        rows=rows,
        features_dir=features_dir,
        dino_variant=ckpt["dino_variant"],
        vjepa_variant=ckpt["vjepa_variant"],
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=feature_collate)

    scores, labels, vids, gates = _collect_predictions(model, loader, device)
    metrics = binary_metrics(scores, labels, threshold=threshold)
    cm = confusion_matrix(scores, labels, threshold=threshold)
    fpr, tpr = roc_curve_points(scores, labels)
    boot = group_bootstrap_auroc(scores, labels, [group_by_vid[v] for v in vids], seed=seed)

    # Error analysis: top false positives & false negatives
    fps: list[dict[str, object]] = []
    fns: list[dict[str, object]] = []
    for vid, sc, lb in zip(vids, scores.tolist(), labels.tolist()):
        pred = 1 if sc >= threshold else 0
        if pred == 1 and int(lb) == 0:
            fps.append({"video_id": vid, "score": round(sc, 4), "source_group_id": group_by_vid[vid]})
        elif pred == 0 and int(lb) == 1:
            fns.append({"video_id": vid, "score": round(sc, 4), "source_group_id": group_by_vid[vid]})
    fps.sort(key=lambda x: float(x["score"]), reverse=True)
    fns.sort(key=lambda x: float(x["score"]))

    p_dir = Path(plots_dir)
    p_dir.mkdir(parents=True, exist_ok=True)
    _plot_confusion(cm, f"{label} ({split}) — AUROC {metrics['auroc']:.3f}", p_dir / f"cm_{exp_id}_{split}.png")
    if gates is not None:
        _plot_gate_hist(gates, labels, f"{label} Gate Distribution ({split})", p_dir / f"gate_{exp_id}_{split}.png")

    payload: dict[str, object] = {
        "experiment": exp_id,
        "label": label,
        "split": split,
        "checkpoint": str(checkpoint_path),
        "val_threshold": threshold,
        "train_seconds": ckpt.get("train_seconds"),
        "metrics": metrics,
        "confusion_matrix": cm,
        "auroc_bootstrap_ci": boot,
        "roc": {"fpr": fpr, "tpr": tpr},
        "false_positives": fps[:25],
        "false_negatives": fns[:25],
    }

    m_dir = Path(metrics_dir)
    m_dir.mkdir(parents=True, exist_ok=True)
    (m_dir / f"eval_{exp_id}_{split}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _plot_combined_roc(evals: list[dict[str, object]], split: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for ev in evals:
        roc = ev["roc"]
        auroc_v = ev["metrics"]["auroc"]
        ax.plot(roc["fpr"], roc["tpr"], label=f"{ev['label']} (AUC={auroc_v:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"STAF-Net Ablation ROC Curves ({split})")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate STAF-Net fusion checkpoints.")
    parser.add_argument("--config", default="configs/fusion.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--all", action="store_true", help="evaluate all available A1..A8 checkpoints")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--features-dir", default=None)
    parser.add_argument("--metrics-dir", default=None)
    parser.add_argument("--plots-dir", default=None)
    args = parser.parse_args(argv)

    y = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) if Path(args.config).exists() else {}
    y_out = y.get("output") or {}

    manifest = args.manifest or y.get("manifest") or "data/manifests/manifest.csv"
    features_dir = args.features_dir or y.get("features_dir") or "data/features"
    ckpt_dir = Path(y_out.get("checkpoints_dir") or "checkpoints")
    metrics_dir = Path(args.metrics_dir or y_out.get("metrics_dir") or "results/metrics")
    plots_dir = Path(args.plots_dir or y_out.get("plots_dir") or "results/plots")

    ckpts: list[Path] = []
    if args.checkpoint:
        ckpts.append(Path(args.checkpoint))
    else:
        for exp_id in EXPERIMENTS:
            p = ckpt_dir / f"fusion_{exp_id}.pt"
            if p.exists():
                ckpts.append(p)
    if not ckpts:
        print(f"[wait] no fusion checkpoints found in {ckpt_dir}")
        return 2

    evals: list[dict[str, object]] = []
    table_rows: list[dict[str, object]] = []
    for p in ckpts:
        ev = evaluate_checkpoint(
            p,
            manifest=manifest,
            features_dir=features_dir,
            split=args.split,
            metrics_dir=metrics_dir,
            plots_dir=plots_dir,
        )
        evals.append(ev)
        m = ev["metrics"]
        table_rows.append(
            {
                "experiment": ev["experiment"],
                "label": ev["label"],
                "auroc": m["auroc"],
                "f1": m["f1"],
                "accuracy": m["accuracy"],
                "eer": m["eer"],
                "threshold": ev["val_threshold"],
                "train_seconds": ev.get("train_seconds") or 0.0,
            }
        )

    if len(evals) > 1:
        _plot_combined_roc(evals, args.split, plots_dir / f"roc_ablation_{args.split}.png")

    md = format_ablation_markdown(table_rows)
    (metrics_dir / f"eval_table_{args.split}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
