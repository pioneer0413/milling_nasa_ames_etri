from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def classification_metrics(y_true, y_pred, metrics: list[str]) -> dict:
    out = {}
    if "accuracy" in metrics:
        out["accuracy"] = float(accuracy_score(y_true, y_pred))
    if "f1_macro" in metrics or "f1" in metrics:
        out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if "precision_macro" in metrics or "precision" in metrics:
        out["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    if "recall_macro" in metrics or "recall" in metrics:
        out["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    return out
