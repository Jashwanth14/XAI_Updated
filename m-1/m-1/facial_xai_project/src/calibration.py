from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit one scalar temperature on validation logits only."""
    logits = logits.detach().float().cpu()
    labels = labels.detach().long().cpu()

    def objective(log_t):
        temperature = float(np.exp(log_t))
        return float(F.cross_entropy(logits / temperature, labels).item())

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(result.x))


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > left) & (confidence <= right)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def calibration_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    predictions = probabilities.argmax(axis=1)
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    clipped = np.clip(probabilities, 1e-12, 1.0)
    confidence = probabilities.max(axis=1)
    correctness = predictions == labels
    order = np.argsort(-confidence)
    risk_coverage = {}
    for coverage in (0.25, 0.50, 0.75, 1.0):
        k = max(1, int(np.ceil(len(labels) * coverage)))
        risk_coverage[str(coverage)] = float(1.0 - correctness[order[:k]].mean())
    matrix = confusion_matrix(labels, predictions, labels=np.arange(probabilities.shape[1]))
    per_class = {}
    for class_id, row in enumerate(matrix):
        precision = row[class_id] / max(int(matrix[:, class_id].sum()), 1)
        recall = row[class_id] / max(int(row.sum()), 1)
        per_class[str(class_id)] = {
            "precision": float(precision), "recall": float(recall),
            "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
            "support": int(row.sum()),
        }
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "nll": float(-np.log(clipped[np.arange(len(labels)), labels]).mean()),
        "brier_score": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece_15_bins": expected_calibration_error(probabilities, labels),
        "mean_confidence": float(confidence.mean()),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "risk_at_coverage": risk_coverage,
    }

