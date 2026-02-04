from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BinaryMetricResult:
    auc: float
    tpr_at_fpr: float


def compute_binary_metrics(
    y_true: Iterable[int],
    y_score: Iterable[float],
    *,
    fpr_target: float,
) -> BinaryMetricResult:
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("compute_binary_metrics requires scikit-learn.") from exc

    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_score_arr = np.asarray(list(y_score), dtype=float)
    auc = float(roc_auc_score(y_true_arr, y_score_arr))
    fpr, tpr, _ = roc_curve(y_true_arr, y_score_arr)
    mask = fpr <= fpr_target
    if np.any(mask):
        tpr_at = float(np.max(tpr[mask]))
    else:
        tpr_at = float(tpr[0]) if len(tpr) else 0.0
    return BinaryMetricResult(auc=auc, tpr_at_fpr=tpr_at)


def mean_metric_results(results: Iterable[BinaryMetricResult]) -> BinaryMetricResult:
    items = list(results)
    if not items:
        raise ValueError("results must be non-empty")
    auc = float(np.mean([r.auc for r in items]))
    tpr_at = float(np.mean([r.tpr_at_fpr for r in items]))
    return BinaryMetricResult(auc=auc, tpr_at_fpr=tpr_at)
