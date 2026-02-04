from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Literal, Mapping, Sequence

import yaml

from .api_v1 import DiscoverResponse


@dataclass(frozen=True)
class ArtifactInfo:
    artifact_dir: Path
    artifact_id: str
    config_hash: str


def canonical_config_yaml(config: Mapping[str, object]) -> str:
    return yaml.safe_dump(config, sort_keys=True)


def canonical_config_hash(config: Mapping[str, object]) -> str:
    dumped = canonical_config_yaml(config)
    digest = hashlib.sha256(dumped.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def artifact_id_from_hash(config_hash: str) -> str:
    return config_hash.replace(":", "-")


def artifact_path(base_dir: str | Path, config_hash: str, version: str = "v1") -> Path:
    base = Path(base_dir)
    return base / version / artifact_id_from_hash(config_hash) / "artifact"


def write_artifact_bundle(
    config: Mapping[str, object],
    response: DiscoverResponse,
    base_dir: str | Path,
    version: str = "v1",
    artifact_uri: str | None = None,
    if_exists: Literal["error", "reuse", "overwrite"] = "error",
    dataset_provenance: Mapping[str, object] | None = None,
) -> ArtifactInfo:
    config_hash = canonical_config_hash(config)
    if response.reproducibility.config_hash != config_hash:
        response = replace(
            response,
            reproducibility=replace(response.reproducibility, config_hash=config_hash),
        )
    artifact_dir = artifact_path(base_dir, config_hash, version=version)
    artifact_id = artifact_id_from_hash(config_hash)
    if artifact_dir.exists():
        if if_exists == "reuse":
            return ArtifactInfo(
                artifact_dir=artifact_dir,
                artifact_id=artifact_id,
                config_hash=config_hash,
            )
        if if_exists == "overwrite":
            _remove_artifact_dir(artifact_dir)
        else:
            raise RuntimeError(f"Artifact already exists at {artifact_dir}")

    parent_dir = artifact_dir.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = parent_dir / f".tmp-{artifact_id}-{os.getpid()}-{time.time_ns()}"
    staging_dir.mkdir(parents=False, exist_ok=False)

    try:
        (staging_dir / "topology").mkdir(parents=True, exist_ok=True)
        (staging_dir / "models").mkdir(parents=True, exist_ok=True)
        (staging_dir / "logs").mkdir(parents=True, exist_ok=True)

        config_yaml = canonical_config_yaml(config)
        (staging_dir / "config.yaml").write_text(config_yaml, encoding="utf-8")
        (staging_dir / "config.hash").write_text(config_hash, encoding="utf-8")

        if artifact_uri is not None:
            response = replace(
                response,
                artifacts=replace(
                    response.artifacts,
                    artifact_id=artifact_id,
                    artifact_uri=artifact_uri,
                ),
            )
        results_path = staging_dir / "results.json"
        results_path.write_text(response.to_json(), encoding="utf-8")

        evidence_payload = {
            "decision": response.decision.to_dict(),
            "summary": response.summary.to_dict(),
            "topology_report": response.topology_report.to_dict(),
            "noise_report": [r.to_dict() for r in response.noise_report],
        }
        (staging_dir / "evidence.json").write_text(
            json.dumps(evidence_payload, indent=2), encoding="utf-8"
        )

        decision_payload = {
            "status": response.status,
            "accepted": response.decision.accepted,
            "reason": response.decision.reason,
            "method_version": response.reproducibility.method_version,
        }
        (staging_dir / "decision.json").write_text(
            json.dumps(decision_payload, indent=2), encoding="utf-8"
        )
        negative_result_path: str | None = None
        if not response.decision.accepted:
            negative_payload = {
                "status": response.status,
                "reason": response.decision.reason,
                "summary": response.summary.to_dict(),
                "topology_report": response.topology_report.to_dict(),
                "noise_report": [r.to_dict() for r in response.noise_report],
                "method_version": response.reproducibility.method_version,
            }
            negative_result_path = "negative_result.json"
            (staging_dir / negative_result_path).write_text(
                json.dumps(negative_payload, indent=2), encoding="utf-8"
            )

        library_versions_path: str | None = None
        if _library_versions_enabled(config):
            library_versions = _collect_library_versions()
            library_versions_path = "libraries.json"
            (staging_dir / library_versions_path).write_text(
                json.dumps(library_versions, indent=2, sort_keys=True),
                encoding="utf-8",
            )

        metrics = {
            "status": response.status,
            "decision_reason": response.decision.reason,
            "stable_components": response.summary.stable_components,
            "significant_H0": response.summary.significant_H0,
            "significant_H1": response.summary.significant_H1,
            "stability_score": response.summary.stability_score,
            "noise_report": [r.to_dict() for r in response.noise_report],
        }
        (staging_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        topology_summary = {
            "H0": response.topology_report.H0,
            "H1": response.topology_report.H1,
        }
        if response.topology_report.PI_energy is not None:
            topology_summary["pi_energy_mean"] = response.topology_report.PI_energy.mean
            topology_summary["pi_energy_std"] = response.topology_report.PI_energy.std
        (staging_dir / "topology" / "summary.json").write_text(
            json.dumps(topology_summary, indent=2), encoding="utf-8"
        )

        _write_models(response.models, staging_dir / "models")

        manifest = _build_manifest(
            config_hash,
            response,
            topology_summary,
            config,
            artifact_uri,
            negative_result_path=negative_result_path,
            library_versions_path=library_versions_path,
            dataset_provenance=dataset_provenance,
        )
        artifact_hash = hash_artifact_dir(staging_dir)
        manifest["integrity"] = {
            "hash_algorithm": "sha256",
            "artifact_hash": f"sha256:{artifact_hash}",
        }
        (staging_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (staging_dir / "artifact.hash").write_text(f"sha256:{artifact_hash}", encoding="utf-8")

        _lock_artifact(staging_dir)
        _publish_artifact_dir(staging_dir, artifact_dir, if_exists=if_exists)
    except Exception:
        _remove_artifact_dir(staging_dir)
        raise

    return ArtifactInfo(
        artifact_dir=artifact_dir,
        artifact_id=artifact_id,
        config_hash=config_hash,
    )


def _write_models(models: Sequence, models_dir: Path) -> None:
    for idx, model in enumerate(models, 1):
        family_id = model.family_id or f"fam-{idx:03d}"
        coeffs = model.representative_model.coefficients
        coeff_order = sorted(coeffs.keys())
        coeff_values = [coeffs[k] for k in coeff_order]
        payload = {
            "family_id": family_id,
            "representative_model": model.representative_model.to_dict(),
            "invariant_terms": list(model.invariant_terms),
            "unstable_terms": list(model.unstable_terms),
            "error": model.error.to_dict(),
            "coeff_order": coeff_order,
        }
        (models_dir / f"{family_id}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        if coeff_order:
            import numpy as np

            np.save(models_dir / f"{family_id}.coeffs.npy", np.asarray(coeff_values, dtype=float))


def _build_manifest(
    config_hash: str,
    response: DiscoverResponse,
    topology_summary: dict,
    config: Mapping[str, object],
    artifact_uri: str | None,
    negative_result_path: str | None = None,
    library_versions_path: str | None = None,
    dataset_provenance: Mapping[str, object] | None = None,
) -> dict:
    mode = "discover"
    run = config.get("run", {})
    if isinstance(run, Mapping) and "mode" in run:
        mode = str(run.get("mode"))
    created_at = datetime.now(timezone.utc).isoformat()
    return {
        "artifact_version": "1.0",
        "created_at": created_at,
        "config": {"hash": config_hash, "path": "config.yaml"},
        "run": {
            "id": response.run_id,
            "mode": mode,
            "seed": response.reproducibility.seed,
            "deterministic": response.reproducibility.deterministic,
            "method_version": response.reproducibility.method_version,
        },
        "dataset": dataset_provenance or {},
        "result": {
            "status": response.status,
            "accepted": response.decision.accepted,
            "reason": response.decision.reason,
            "stable_components": response.summary.stable_components,
            "stability_score": response.summary.stability_score,
        },
        "topology": topology_summary,
        "artifacts": {
            "artifact_uri": artifact_uri,
            "config": "config.yaml",
            "results": "results.json",
            "evidence": "evidence.json",
            "decision": "decision.json",
            "negative_result": negative_result_path,
        },
        "reproducibility": {
            "version": response.reproducibility.version,
            "config_hash": response.reproducibility.config_hash,
            "library_versions": library_versions_path,
        },
    }


def _library_versions_enabled(config: Mapping[str, object]) -> bool:
    reproducibility = config.get("reproducibility")
    if not isinstance(reproducibility, Mapping):
        return False
    value = reproducibility.get("library_versions", False)
    return bool(value)


def _safe_version(pkg: str) -> str:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"
    except Exception:
        return "unknown"


def _collect_library_versions() -> dict[str, object]:
    packages = [
        "numpy",
        "pydantic",
        "PyYAML",
        "ripser",
        "persim",
        "fastapi",
        "uvicorn",
    ]
    pkg_versions = {pkg: _safe_version(pkg) for pkg in packages}
    return {
        "python": sys.version.split()[0],
        "packages": pkg_versions,
    }


def hash_artifact_dir(path: Path) -> str:
    h = hashlib.sha256()
    for file in sorted(path.rglob("*")):
        if file.is_dir():
            continue
        if file.name in ("_LOCK", "manifest.json", "artifact.hash"):
            continue
        h.update(file.read_bytes())
    return h.hexdigest()


def verify_artifact(path: Path) -> bool:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("manifest.json missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("integrity", {}).get("artifact_hash")
    if not expected:
        raise RuntimeError("artifact_hash missing from manifest")
    expected = expected.replace("sha256:", "")
    actual = hash_artifact_dir(path)
    return actual == expected


def _lock_artifact(path: Path) -> None:
    lock_path = path / "_LOCK"
    lock_path.write_text("immutable", encoding="utf-8")
    for root, dirs, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            try:
                os.chmod(file_path, 0o444)
            except OSError:
                pass
        for name in dirs:
            dir_path = Path(root) / name
            try:
                os.chmod(dir_path, 0o555)
            except OSError:
                pass


def _ensure_mutable(path: Path) -> None:
    if path.exists() and (path / "_LOCK").exists():
        raise RuntimeError("Artifact is immutable")


def _publish_artifact_dir(
    staging_dir: Path,
    artifact_dir: Path,
    *,
    if_exists: Literal["error", "reuse", "overwrite"],
) -> None:
    if not staging_dir.exists():
        raise RuntimeError("staging artifact directory missing before publish")
    try:
        staging_dir.replace(artifact_dir)
        return
    except OSError as exc:
        if artifact_dir.exists():
            if if_exists == "reuse":
                _remove_artifact_dir(staging_dir)
                return
            if if_exists == "overwrite":
                _remove_artifact_dir(artifact_dir)
                staging_dir.replace(artifact_dir)
                return
            raise RuntimeError(f"Artifact already exists at {artifact_dir}") from exc
        raise


def _remove_artifact_dir(path: Path) -> None:
    if not path.exists():
        return
    for root, dirs, files in os.walk(path):
        for name in files:
            try:
                os.chmod(Path(root) / name, 0o666)
            except OSError:
                pass
        for name in dirs:
            try:
                os.chmod(Path(root) / name, 0o777)
            except OSError:
                pass
    shutil.rmtree(path)
