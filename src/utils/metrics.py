"""STAF-Net | src/utils/metrics.py

Owner   : Person 3 (fills the rest), Person 1 wrote the binary core needed by
          Stage I branch training.
Status  : PARTIAL - binary classification metrics implemented; bootstrap CI,
          ROC/confusion export and the results-table writer are still TODO.

Contract
--------
- Pure functions over scores + labels. No model, no dataset, no I/O.
- scores are probabilities of FAKE in [0, 1]; labels are 0.0 real / 1.0 fake.
- Threshold selection and checkpoint selection use validation data only.

Implemented here
----------------
    binary_metrics(scores, labels) -> auroc, f1, accuracy, precision, recall, eer
    accuracy / precision / recall / f1_score
    eer(scores, labels)

Not implemented
---------------
- group-level bootstrap confidence intervals (needs source_group_id)
- confusion matrix and ROC curve arrays for plots
- JSON exporter for the standard results table
"""

from __future__ import annotations

import torch

__all__ = [
    "binary_metrics",
    "accuracy",
    "precision_recall_f1",
    "eer",
    "best_threshold_from_val",
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
    """Convenience bundle used by Stage I training and evaluation."""
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
