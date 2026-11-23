"""STAF-Net | src/utils/metrics.py

Owner   : Person 3 + Person 1
Status  : IMPLEMENTED - binary classification metrics, ROC/confusion arrays,
          source-group-level bootstrap CI, and Markdown ablation table formatter.

Contract
--------
- Pure functions over scores + labels. No model, no dataset, no I/O.
- scores are probabilities of FAKE in [0, 1]; labels are 0.0 real / 1.0 fake.
- Threshold selection and checkpoint selection use validation data only.
"""

from __future__ import annotations

import random
from typing import Sequence

import torch

__all__ = [
    "binary_metrics",
    "accuracy",
    "precision_recall_f1",
    "auroc",
    "eer",
    "best_threshold_from_val",
    "roc_curve_points",
    "confusion_matrix",
    "group_bootstrap_auroc",
    "format_ablation_markdown",
]


def _as_vectors(scores: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
    labels = torch.as_tensor(labels, dtype=torch.float32).reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError(
            f"metrics: scores and labels must have the same shape, got "
            f"{tuple(scores.shape)} vs {tuple(labels.shape)}"
        )
    if scores.numel() == 0:
        raise ValueError("metrics: empty input")
    return scores, labels


def accuracy(scores: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    scores, labels = _as_vectors(scores, labels)
    preds = (scores >= threshold).float()
    return float((preds == labels).float().mean())


def precision_recall_f1(
    scores: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5
) -> tuple[float, float, float]:
    scores, labels = _as_vectors(scores, labels)
    preds = (scores >= threshold).float()
    tp = float(((preds == 1) & (labels == 1)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def confusion_matrix(
    scores: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5
) -> dict[str, int]:
    """Return binary confusion counts {tn, fp, fn, tp}."""
    scores, labels = _as_vectors(scores, labels)
    preds = (scores >= threshold).float()
    return {
        "tn": int(((preds == 0) & (labels == 0)).sum().item()),
        "fp": int(((preds == 1) & (labels == 0)).sum().item()),
        "fn": int(((preds == 0) & (labels == 1)).sum().item()),
        "tp": int(((preds == 1) & (labels == 1)).sum().item()),
    }


def _roc_points(scores: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (fpr, tpr) over all thresholds, ascending."""
    order = torch.argsort(scores, descending=True)
    scores, labels = scores[order], labels[order]
    positives = float((labels == 1).sum())
    negatives = float((labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError(
            "metrics: AUROC is undefined with a single class in the batch "
            f"({int(positives)} fake / {int(negatives)} real)"
        )
    tp = torch.cumsum(labels, dim=0)
    fp = torch.cumsum(1.0 - labels, dim=0)
    tpr = torch.cat([torch.zeros(1), tp / positives, torch.ones(1)])
    fpr = torch.cat([torch.zeros(1), fp / negatives, torch.ones(1)])
    return fpr, tpr


def roc_curve_points(scores: torch.Tensor, labels: torch.Tensor) -> tuple[list[float], list[float]]:
    """Return JSON-serializable (fpr, tpr) lists for ROC plotting."""
    scores_v, labels_v = _as_vectors(scores, labels)
    fpr, tpr = _roc_points(scores_v, labels_v)
    return [float(x) for x in fpr.tolist()], [float(x) for x in tpr.tolist()]


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Area under the ROC curve via trapezoidal integration."""
    scores, labels = _as_vectors(scores, labels)
    fpr, tpr = _roc_points(scores, labels)
    return float(torch.trapz(tpr, fpr).abs())


def eer(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Equal error rate: where FPR and (1 - TPR) cross."""
    scores, labels = _as_vectors(scores, labels)
    fpr, tpr = _roc_points(scores, labels)
    fnr = 1.0 - tpr
    diff = (fpr - fnr).abs()
    idx = int(torch.argmin(diff))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def best_threshold_from_val(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Pick the threshold maximising F1 on validation data (never on test)."""
    scores, labels = _as_vectors(scores, labels)
    candidates = torch.unique(scores)
    best_threshold, best_f1 = 0.5, -1.0
    for candidate in candidates.tolist():
        _, _, f1 = precision_recall_f1(scores, labels, threshold=float(candidate))
        if f1 > best_f1:
            best_threshold, best_f1 = float(candidate), f1
    return best_threshold


def binary_metrics(scores: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    """Convenience bundle used by Stage I/II training and evaluation."""
    scores_v, labels_v = _as_vectors(scores, labels)
    precision, recall, f1 = precision_recall_f1(scores_v, labels_v, threshold=threshold)
    return {
        "auroc": auroc(scores_v, labels_v),
        "f1": f1,
        "accuracy": accuracy(scores_v, labels_v, threshold=threshold),
        "precision": precision,
        "recall": recall,
        "eer": eer(scores_v, labels_v),
        "threshold": float(threshold),
        "n": float(scores_v.numel()),
    }


def group_bootstrap_auroc(
    scores: torch.Tensor,
    labels: torch.Tensor,
    group_ids: Sequence[str],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Cluster-bootstrap 95% CI for AUROC by source_group_id (plan §12)."""
    scores_v, labels_v = _as_vectors(scores, labels)
    if len(group_ids) != int(scores_v.numel()):
        raise ValueError("group_bootstrap_auroc: group_ids length mismatch")

    by_group: dict[str, list[int]] = {}
    for idx, gid in enumerate(group_ids):
        by_group.setdefault(str(gid), []).append(idx)
    groups = list(by_group.values())
    if len(groups) < 2:
        point = auroc(scores_v, labels_v)
        return {"mean": point, "ci_low": point, "ci_high": point, "n_valid": 1.0}

    rng = random.Random(seed)
    boot: list[float] = []
    n_groups = len(groups)
    for _ in range(n_bootstrap):
        chosen: list[int] = []
        for _ in range(n_groups):
            chosen.extend(groups[rng.randrange(n_groups)])
        idx_t = torch.tensor(chosen, dtype=torch.long)
        sub_l = labels_v[idx_t]
        if sub_l.min() == sub_l.max():
            continue
        boot.append(auroc(scores_v[idx_t], sub_l))

    if not boot:
        point = auroc(scores_v, labels_v)
        return {"mean": point, "ci_low": point, "ci_high": point, "n_valid": 0.0}
    boot.sort()
    lo = boot[int(0.025 * (len(boot) - 1))]
    hi = boot[int(0.975 * (len(boot) - 1))]
    return {
        "mean": sum(boot) / len(boot),
        "ci_low": lo,
        "ci_high": hi,
        "n_valid": float(len(boot)),
    }


def format_ablation_markdown(rows: Sequence[dict[str, object]]) -> str:
    """Format A1-A8 results into the plan §12 standard Markdown table."""
    lines = [
        "| Experiment | AUROC ↑ | F1 ↑ | Accuracy ↑ | EER ↓ | Val Thr | Training time |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        exp = str(r.get("experiment", ""))
        label = str(r.get("label", exp))
        auroc_v = f"{float(r['auroc']):.4f}" if "auroc" in r else "—"
        f1_v = f"{float(r['f1']):.4f}" if "f1" in r else "—"
        acc_v = f"{float(r['accuracy']):.4f}" if "accuracy" in r else "—"
        eer_v = f"{float(r['eer']):.4f}" if "eer" in r else "—"
        thr_v = f"{float(r['threshold']):.3f}" if "threshold" in r else "—"
        sec = f"{float(r['train_seconds']):.1f}s" if "train_seconds" in r else "—"
        lines.append(f"| {label} | {auroc_v} | {f1_v} | {acc_v} | {eer_v} | {thr_v} | {sec} |")
    return "\n".join(lines) + "\n"
