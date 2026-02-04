from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping
import urllib.parse
import urllib.request

import numpy as np


@dataclass(frozen=True)
class DatasetPayload:
    x: np.ndarray
    y: np.ndarray
    metadata: dict[str, Any]
    dataset_id: str | None
    provenance: dict[str, Any]


def _hash_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _hash_arrays(x: np.ndarray, y: np.ndarray, metadata: Mapping[str, Any] | None) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(x, dtype=float).tobytes())
    h.update(np.asarray(y, dtype=float).tobytes())
    if metadata:
        h.update(json.dumps(dict(metadata), sort_keys=True).encode("utf-8"))
    return f"sha256:{h.hexdigest()}"


def _coerce_x(values: Any) -> list[list[float]]:
    if isinstance(values, list) and values and all(isinstance(v, (int, float)) for v in values):
        return [[float(v)] for v in values]
    return [[float(cell) for cell in row] for row in values]


def _parse_structured(payload: Mapping[str, Any]) -> tuple[list[list[float]], list[float], dict[str, Any], str | None]:
    metadata = payload.get("metadata") or {}
    dataset_id = payload.get("id")
    x = payload.get("x")
    y = payload.get("y")
    if isinstance(x, Mapping) and "values" in x:
        x = x.get("values")
    if isinstance(y, Mapping) and "values" in y:
        y = y.get("values")
    if x is None or y is None:
        raise ValueError("dataset file must include x and y")
    x_values = _coerce_x(x)
    y_values = [float(v) for v in y]
    return x_values, y_values, dict(metadata), str(dataset_id) if dataset_id is not None else None


def _is_number(value: str) -> bool:
    try:
        float(value)
    except Exception:
        return False
    return True


def _parse_csv(text: str, delimiter: str) -> tuple[list[list[float]], list[float]]:
    rows = []
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    for row in reader:
        if not row:
            continue
        rows.append([cell.strip() for cell in row if cell is not None])
    if not rows:
        raise ValueError("csv dataset is empty")
    if any(not _is_number(cell) for cell in rows[0] if cell != ""):
        rows = rows[1:]
    if not rows:
        raise ValueError("csv dataset has no numeric rows")
    x_values: list[list[float]] = []
    y_values: list[float] = []
    for row in rows:
        if len(row) < 2:
            raise ValueError("csv dataset must have at least 2 columns")
        values = [float(cell) for cell in row]
        x_values.append(values[:-1])
        y_values.append(values[-1])
    return x_values, y_values


def _parse_dataset_bytes(payload: bytes, *, suffix: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any], str | None, str]:
    suffix = suffix.lower()
    if suffix in {".json"}:
        data = json.loads(payload.decode("utf-8"))
        x_values, y_values, metadata, dataset_id = _parse_structured(data)
        return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float), metadata, dataset_id, "json"
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise ImportError("YAML dataset requires PyYAML.") from exc
        data = yaml.safe_load(payload.decode("utf-8"))
        x_values, y_values, metadata, dataset_id = _parse_structured(data)
        return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float), metadata, dataset_id, "yaml"
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        x_values, y_values = _parse_csv(payload.decode("utf-8"), delimiter=delimiter)
        return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float), {}, None, "csv"
    raise ValueError(f"unsupported dataset format '{suffix}'")


def _validate_dataset_arrays(
    x: np.ndarray,
    y: np.ndarray,
    metadata: Mapping[str, Any],
    *,
    max_points: int,
    max_features: int,
    max_total_values: int,
    max_metadata_entries: int,
    max_metadata_value_chars: int,
) -> None:
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim not in (1, 2):
        raise ValueError("dataset.x must be 1D or 2D")
    if y.ndim != 1:
        raise ValueError("dataset.y must be 1D")
    if x.shape[0] != y.shape[0]:
        raise ValueError("dataset.x and dataset.y must have matching length")
    if x.shape[0] < 1:
        raise ValueError("dataset must contain at least one point")
    feature_count = 1 if x.ndim == 1 else x.shape[1]
    n_points = int(x.shape[0])
    if n_points > max_points:
        raise ValueError(f"dataset has {n_points} points; max_points is {max_points}")
    if feature_count > max_features:
        raise ValueError(f"dataset has {feature_count} features; max_features is {max_features}")
    total_values = n_points * feature_count + int(y.shape[0])
    if total_values > max_total_values:
        raise ValueError(f"dataset has {total_values} values; max_total_values is {max_total_values}")
    if not np.isfinite(x).all():
        raise ValueError("dataset.x must contain only finite numeric values")
    if not np.isfinite(y).all():
        raise ValueError("dataset.y must contain only finite numeric values")

    meta = dict(metadata or {})
    if len(meta) > max_metadata_entries:
        raise ValueError(
            f"dataset.metadata has {len(meta)} entries; max_metadata_entries is {max_metadata_entries}"
        )
    for key, value in meta.items():
        if not isinstance(key, str):
            raise ValueError("dataset.metadata keys must be strings")
        if len(str(value)) > max_metadata_value_chars:
            raise ValueError(
                "dataset.metadata value is too long; "
                f"max_metadata_value_chars is {max_metadata_value_chars}"
            )


def load_dataset_from_file(
    path: str,
    *,
    limits: Mapping[str, int],
) -> DatasetPayload:
    resolved = Path(path).expanduser().resolve()
    raw = resolved.read_bytes()
    x, y, metadata, dataset_id, fmt = _parse_dataset_bytes(raw, suffix=resolved.suffix)
    _validate_dataset_arrays(
        x,
        y,
        metadata,
        max_points=int(limits["max_points"]),
        max_features=int(limits["max_features"]),
        max_total_values=int(limits["max_total_values"]),
        max_metadata_entries=int(limits["max_metadata_entries"]),
        max_metadata_value_chars=int(limits["max_metadata_value_chars"]),
    )
    provenance = {
        "source": "file",
        "location": path,
        "resolved_path": str(resolved),
        "format": fmt,
        "sha256": _hash_bytes(raw),
        "size_bytes": len(raw),
    }
    return DatasetPayload(x=x, y=y, metadata=metadata, dataset_id=dataset_id, provenance=provenance)


def load_dataset_from_uri(
    uri: str,
    *,
    limits: Mapping[str, int],
) -> DatasetPayload:
    if uri.startswith("file://"):
        path = uri[len("file://") :]
        return load_dataset_from_file(path, limits=limits)
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("dataset uri must be http(s) or file://")
    with urllib.request.urlopen(uri, timeout=5) as response:
        raw = response.read()
    suffix = Path(parsed.path).suffix or ".json"
    x, y, metadata, dataset_id, fmt = _parse_dataset_bytes(raw, suffix=suffix)
    _validate_dataset_arrays(
        x,
        y,
        metadata,
        max_points=int(limits["max_points"]),
        max_features=int(limits["max_features"]),
        max_total_values=int(limits["max_total_values"]),
        max_metadata_entries=int(limits["max_metadata_entries"]),
        max_metadata_value_chars=int(limits["max_metadata_value_chars"]),
    )
    provenance = {
        "source": "uri",
        "location": uri,
        "format": fmt,
        "sha256": _hash_bytes(raw),
        "size_bytes": len(raw),
    }
    return DatasetPayload(x=x, y=y, metadata=metadata, dataset_id=dataset_id, provenance=provenance)


def provenance_from_arrays(
    x: np.ndarray,
    y: np.ndarray,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "source": "inline",
        "sha256": _hash_arrays(x, y, metadata),
        "size_bytes": int(np.asarray(x, dtype=float).nbytes + np.asarray(y, dtype=float).nbytes),
    }
