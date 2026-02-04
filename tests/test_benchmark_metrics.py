import pytest

from tgefd.benchmarks import compute_binary_metrics, mean_metric_results, BinaryMetricResult


def test_compute_binary_metrics_separable():
    pytest.importorskip("sklearn")
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    metrics = compute_binary_metrics(y_true, y_score, fpr_target=0.5)
    assert metrics.auc == pytest.approx(1.0)
    assert 0.9 <= metrics.tpr_at_fpr <= 1.0


def test_mean_metric_results():
    items = [BinaryMetricResult(auc=0.8, tpr_at_fpr=0.6), BinaryMetricResult(auc=0.6, tpr_at_fpr=0.4)]
    mean = mean_metric_results(items)
    assert mean.auc == pytest.approx(0.7)
    assert mean.tpr_at_fpr == pytest.approx(0.5)
