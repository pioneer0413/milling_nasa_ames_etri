from __future__ import annotations

from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def anomaly_metrics(y_true, scores, threshold: float = 0.5) -> dict:
    pred = scores >= threshold
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "threshold": threshold,
    }
