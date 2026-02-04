from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class Regression:
    kind: str
    metric: str
    label: str | None
    index: int | None
    n_value: int | None
    baseline: float | None
    candidate: float | None
    drop: float | None


def load_curves(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "metrics" not in data:
        raise ValueError("Invalid curves JSON: missing metrics")
    return data


def resolve_metric_name(curves: Mapping[str, Any], metric: str | None) -> str:
    if metric:
        return metric
    default_metric = curves.get("default_metric")
    if isinstance(default_metric, str) and default_metric:
        return default_metric
    metrics = curves.get("metrics", {})
    if isinstance(metrics, dict) and metrics:
        return next(iter(metrics.keys()))
    raise ValueError("No metrics available in curves JSON")


def extract_metric_series(curves: Mapping[str, Any], metric: str) -> dict[str, list[float]]:
    metrics = curves.get("metrics", {})
    if not isinstance(metrics, dict) or metric not in metrics:
        raise ValueError(f"Metric {metric!r} not present in curves JSON")
    series = metrics[metric]
    if not isinstance(series, list):
        raise ValueError(f"Metric {metric!r} has invalid series data")
    mapping: dict[str, list[float]] = {}
    for item in series:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        values = item.get("values")
        if isinstance(label, str) and isinstance(values, list):
            mapping[label] = [float(v) for v in values]
    if not mapping:
        raise ValueError(f"Metric {metric!r} has no labeled series")
    return mapping


def compare_curves(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    metric: str | None = None,
    max_drop: float = 0.0,
) -> list[Regression]:
    metric_name = resolve_metric_name(baseline, metric)
    candidate_metric = resolve_metric_name(candidate, metric)
    if metric_name != candidate_metric and metric is None:
        metric_name = candidate_metric

    base_series = extract_metric_series(baseline, metric_name)
    cand_series = extract_metric_series(candidate, metric_name)
    base_n = baseline.get("n_values")
    cand_n = candidate.get("n_values")
    regressions: list[Regression] = []

    if base_n != cand_n:
        regressions.append(
            Regression(
                kind="n_values_mismatch",
                metric=metric_name,
                label=None,
                index=None,
                n_value=None,
                baseline=None,
                candidate=None,
                drop=None,
            )
        )
    n_values = base_n if isinstance(base_n, list) else None

    for label, base_values in base_series.items():
        cand_values = cand_series.get(label)
        if cand_values is None:
            regressions.append(
                Regression(
                    kind="missing_label",
                    metric=metric_name,
                    label=label,
                    index=None,
                    n_value=None,
                    baseline=None,
                    candidate=None,
                    drop=None,
                )
            )
            continue
        if len(base_values) != len(cand_values):
            regressions.append(
                Regression(
                    kind="length_mismatch",
                    metric=metric_name,
                    label=label,
                    index=None,
                    n_value=None,
                    baseline=None,
                    candidate=None,
                    drop=None,
                )
            )
            continue
        for idx, (base_val, cand_val) in enumerate(zip(base_values, cand_values)):
            drop = float(base_val) - float(cand_val)
            if drop > max_drop:
                n_value = None
                if isinstance(n_values, list) and idx < len(n_values):
                    try:
                        n_value = int(n_values[idx])
                    except Exception:
                        n_value = None
                regressions.append(
                    Regression(
                        kind="metric_drop",
                        metric=metric_name,
                        label=label,
                        index=idx,
                        n_value=n_value,
                        baseline=float(base_val),
                        candidate=float(cand_val),
                        drop=drop,
                    )
                )
    return regressions


def format_regressions(regressions: list[Regression]) -> list[str]:
    lines: list[str] = []
    for reg in regressions:
        if reg.kind == "metric_drop":
            location = f"index {reg.index}"
            if reg.n_value is not None:
                location = f"N={reg.n_value}"
            lines.append(
                f"metric_drop {reg.metric} label={reg.label} {location} "
                f"baseline={reg.baseline:.4f} candidate={reg.candidate:.4f} drop={reg.drop:.4f}"
            )
        elif reg.kind == "missing_label":
            lines.append(f"missing_label {reg.metric} label={reg.label}")
        elif reg.kind == "length_mismatch":
            lines.append(f"length_mismatch {reg.metric} label={reg.label}")
        elif reg.kind == "n_values_mismatch":
            lines.append(f"n_values_mismatch {reg.metric}")
        else:
            lines.append(f"{reg.kind} {reg.metric} label={reg.label}")
    return lines
