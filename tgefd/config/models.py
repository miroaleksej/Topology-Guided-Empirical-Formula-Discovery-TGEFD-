from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
import re

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator, field_validator, PrivateAttr

from ..api_v1 import (
    ComputeBudget,
    Dataset,
    DiscoverRequest,
    HypothesisSpace,
    NoiseProfile,
    PHConfig,
    PersistenceImageConfig as ApiPersistenceImageConfig,
    TopologyConfig as ApiTopologyConfig,
)
from ..evaluation import AcceptancePolicy, EvaluationPolicy, StabilityScorePolicy
from ..adaptive import AdaptiveHypercubeConfig
from ..storage import ArtifactStoreConfig as StorageConfig
from ..dataset_io import load_dataset_from_file, load_dataset_from_uri, provenance_from_arrays


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Optional[str] = None
    mode: Literal["discover", "evaluate", "validate"]
    seed: int
    deterministic: bool = True

    @field_validator("seed")
    @classmethod
    def _seed_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("seed must be non-negative")
        return value

    @field_validator("deterministic")
    @classmethod
    def _deterministic_only(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("run.deterministic must be true (deterministic mode is standardized)")
        return value


class DatasetX(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["array", "table"]
    values: List[List[float]]

    @field_validator("values", mode="before")
    @classmethod
    def _coerce_values(cls, value):
        if isinstance(value, list) and value and all(isinstance(v, (int, float)) for v in value):
            return [[float(v)] for v in value]
        return value

    @field_validator("values")
    @classmethod
    def _validate_values(cls, value: List[List[float]]) -> List[List[float]]:
        if not value:
            raise ValueError("dataset.x.values must be non-empty")
        width = None
        for row in value:
            if not row:
                raise ValueError("dataset.x.values rows must be non-empty")
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise ValueError("dataset.x.values rows must have consistent width")
            for cell in row:
                if not np.isfinite(float(cell)):
                    raise ValueError("dataset.x.values must contain only finite numbers")
        return value


class DatasetY(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: List[float]

    @field_validator("values")
    @classmethod
    def _validate_values(cls, value: List[float]) -> List[float]:
        if not value:
            raise ValueError("dataset.y.values must be non-empty")
        for cell in value:
            if not np.isfinite(float(cell)):
                raise ValueError("dataset.y.values must contain only finite numbers")
        return value


class DatasetLimitsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_points: int = Field(50_000, ge=1)
    max_features: int = Field(256, ge=1)
    max_total_values: int = Field(5_000_000, ge=1)
    max_metadata_entries: int = Field(64, ge=0)
    max_metadata_value_chars: int = Field(2048, ge=1)


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["inline", "file", "uri"]
    x: Optional[DatasetX] = None
    y: Optional[DatasetY] = None
    path: Optional[str] = None
    uri: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    limits: DatasetLimitsConfig = Field(default_factory=DatasetLimitsConfig)
    id: Optional[str] = None

    @model_validator(mode="after")
    def _check_dimensions(self) -> "DatasetConfig":
        if self.source == "inline":
            if self.x is None or self.y is None:
                raise ValueError("dataset.x and dataset.y are required when dataset.source='inline'")

            if len(self.x.values) != len(self.y.values):
                raise ValueError("dataset.x and dataset.y must have matching length")
            n_points = len(self.x.values)
            n_features = len(self.x.values[0]) if self.x.values else 0
            total_values = n_points * n_features + len(self.y.values)
            if n_points > self.limits.max_points:
                raise ValueError(
                    f"dataset has {n_points} points; max_points is {self.limits.max_points}"
                )
            if n_features > self.limits.max_features:
                raise ValueError(
                    f"dataset has {n_features} features; max_features is {self.limits.max_features}"
                )
            if total_values > self.limits.max_total_values:
                raise ValueError(
                    f"dataset has {total_values} values; max_total_values is {self.limits.max_total_values}"
                )
        elif self.source == "file":
            if not self.path:
                raise ValueError("dataset.path is required when dataset.source='file'")
            if self.x is not None or self.y is not None:
                raise ValueError("dataset.x and dataset.y must be omitted for file datasets")
        elif self.source == "uri":
            if not self.uri:
                raise ValueError("dataset.uri is required when dataset.source='uri'")
            if self.x is not None or self.y is not None:
                raise ValueError("dataset.x and dataset.y must be omitted for uri datasets")
        else:
            raise ValueError("dataset.source must be one of: inline, file, uri")

        metadata = self.metadata or {}
        if len(metadata) > self.limits.max_metadata_entries:
            raise ValueError(
                f"dataset.metadata has {len(metadata)} entries; max_metadata_entries is {self.limits.max_metadata_entries}"
            )
        for key, value in metadata.items():
            if not isinstance(key, str):
                raise ValueError("dataset.metadata keys must be strings")
            rendered = str(value)
            if len(rendered) > self.limits.max_metadata_value_chars:
                raise ValueError(
                    "dataset.metadata value is too long; "
                    f"max_metadata_value_chars is {self.limits.max_metadata_value_chars}"
                )
        return self


class FeatureSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    expression: str
    parameters: Optional[List[str]] = None


class ParameterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["float", "int"]
    values: List[float] = Field(..., min_length=1)


class RegularizationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    lambda_: List[float] = Field(..., alias="lambda", min_length=1)


class AdaptiveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    refine_factor: float = Field(2.0, gt=1.0)
    top_k: int = Field(5, ge=1)
    min_component_size: int = Field(2, ge=1)
    h0_threshold: Optional[float] = None


class HypothesisSpaceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: List[FeatureSpec]
    parameters: Dict[str, ParameterSpec]
    regularization: RegularizationSpec
    adaptive: Optional[AdaptiveConfig] = None

    @model_validator(mode="after")
    def _validate_feature_parameters(self) -> "HypothesisSpaceConfig":
        for feature in self.features:
            params = feature.parameters
            if not params:
                params = _infer_params_from_expression(feature.expression)
            for param in params:
                if param not in self.parameters:
                    raise ValueError(f"Feature '{feature.name}' uses undefined parameter '{param}'")
        return self


class PersistentHomologyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_dim: int = Field(..., ge=0)
    metric: Literal["euclidean"]


class PIWeight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["persistence"]
    params: Dict[str, float]


class PIKernel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["gaussian", "uniform"]
    params: Dict[str, Any]


class PersistenceImageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    birth_range: List[float]
    pers_range: List[float]
    pixel_size: float
    weight: PIWeight
    kernel: PIKernel


class TopologyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    persistent_homology: PersistentHomologyConfig
    persistence_image: PersistenceImageConfig
    stability_threshold: float = Field(..., gt=0.0)
    h1_threshold: Optional[float] = None
    min_component_size: int = Field(5, ge=1)
    point_scale: Literal["none", "standard"] = "standard"


class NoiseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    type: Literal["gaussian"]
    levels: List[float]
    trials_per_level: int = Field(..., ge=1)


class AcceptanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_stable_components: int = Field(..., ge=0)
    require_h1: bool
    coeff_tol: float = Field(1e-6, gt=0.0)


class StabilityScoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["pi_energy"]
    aggregation: Literal["mean", "median"]
    tolerance: float = Field(..., gt=0.0)
    min_pi_energy: float = Field(0.0, ge=0.0)


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    acceptance: AcceptanceConfig
    stability_score: StabilityScoreConfig
    rejection_reasons: List[str]


class ArtifactStoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["local", "s3", "gcs"] = "local"
    base_dir: Optional[str] = None
    bucket: Optional[str] = None
    prefix: str = "tgefd/runs"
    endpoint_url: Optional[str] = None
    region: Optional[str] = None
    upload_timeout_sec: float = Field(30.0, gt=0.0)
    retry_max_attempts: int = Field(3, ge=1)
    retry_backoff_sec: float = Field(0.2, ge=0.0)
    retry_max_backoff_sec: float = Field(2.0, ge=0.0)


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verbosity: Literal["minimal", "summary", "full"]
    save_artifacts: bool
    formats: List[Literal["json", "yaml"]]
    paths: Dict[str, str]
    store: Optional["ArtifactStoreConfig"] = None


class ReproducibilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tgefd_version: str
    library_versions: bool
    hash_algorithm: Literal["sha256"]


class ComputeBudgetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_hypotheses: int = Field(50_000, ge=1)
    max_noise_trials: int = Field(32, ge=1)
    max_total_runs: int = Field(256, ge=1)
    max_model_evals: int = Field(2_000_000, ge=1)
    symbolic_regression_timeout_sec: float = Field(10.0, gt=0.0)
    ph_timeout_sec: float = Field(10.0, gt=0.0)
    max_request_wall_time_sec: float = Field(60.0, gt=0.0)
    max_point_cloud_points: int = Field(5_000, ge=1)


class TGEFDConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    _dataset_provenance: dict[str, Any] | None = PrivateAttr(default=None)

    tgefd_config_version: Literal["1.0"]
    run: RunConfig
    dataset: DatasetConfig
    hypothesis_space: HypothesisSpaceConfig
    topology: TopologyConfig
    noise: Optional[NoiseConfig]
    evaluation: EvaluationConfig
    output: OutputConfig
    reproducibility: ReproducibilityConfig
    compute_budget: ComputeBudgetConfig = Field(default_factory=ComputeBudgetConfig)

    @model_validator(mode="after")
    def _check_mode_constraints(self) -> "TGEFDConfig":
        mode = self.run.mode
        if mode == "discover" and self.noise is None:
            raise ValueError("Noise config is required in discover mode")
        if mode == "evaluate" and self.noise is not None and self.noise.enabled:
            raise ValueError("Noise config must be disabled in evaluate mode")
        return self

    def to_request(self) -> DiscoverRequest:
        if self.dataset.source == "inline":
            if self.dataset.x is None or self.dataset.y is None:
                raise ValueError("dataset.x and dataset.y are required for request conversion")
            x = np.asarray(self.dataset.x.values, dtype=float)
            y = np.asarray(self.dataset.y.values, dtype=float)
            metadata = self.dataset.metadata or {}
            dataset_id = self.dataset.id
            self._dataset_provenance = provenance_from_arrays(x, y, metadata)
        elif self.dataset.source == "file":
            limits = self.dataset.limits.model_dump()
            payload = load_dataset_from_file(self.dataset.path or "", limits=limits)
            x = payload.x
            y = payload.y
            metadata = payload.metadata
            dataset_id = payload.dataset_id
            self._dataset_provenance = payload.provenance
        elif self.dataset.source == "uri":
            limits = self.dataset.limits.model_dump()
            payload = load_dataset_from_uri(self.dataset.uri or "", limits=limits)
            x = payload.x
            y = payload.y
            metadata = payload.metadata
            dataset_id = payload.dataset_id
            self._dataset_provenance = payload.provenance
        else:
            raise ValueError("dataset.source must be one of: inline, file, uri")

        merged_metadata = dict(metadata)
        if self.dataset.metadata:
            merged_metadata.update(self.dataset.metadata)
        if len(merged_metadata) > self.dataset.limits.max_metadata_entries:
            raise ValueError(
                f"dataset.metadata has {len(merged_metadata)} entries; "
                f"max_metadata_entries is {self.dataset.limits.max_metadata_entries}"
            )
        for key, value in merged_metadata.items():
            if not isinstance(key, str):
                raise ValueError("dataset.metadata keys must be strings")
            rendered = str(value)
            if len(rendered) > self.dataset.limits.max_metadata_value_chars:
                raise ValueError(
                    "dataset.metadata value is too long; "
                    f"max_metadata_value_chars is {self.dataset.limits.max_metadata_value_chars}"
                )

        dataset = Dataset(
            id=self.dataset.id or dataset_id or self.run.id or "dataset",
            x=x,
            y=y,
            metadata=merged_metadata,
        )

        parameters = {name: spec.values for name, spec in self.hypothesis_space.parameters.items()}
        hypothesis = HypothesisSpace(
            features=[f.expression for f in self.hypothesis_space.features],
            parameters=parameters,
            regularization={"lambda": self.hypothesis_space.regularization.lambda_},
        )

        topology = ApiTopologyConfig(
            ph=PHConfig(
                max_dim=self.topology.persistent_homology.max_dim,
                metric=self.topology.persistent_homology.metric,
            ),
            persistence_image=ApiPersistenceImageConfig(
                birth_range=tuple(self.topology.persistence_image.birth_range),
                pers_range=tuple(self.topology.persistence_image.pers_range),
                pixel_size=self.topology.persistence_image.pixel_size,
                weight=self.topology.persistence_image.weight.type,
                weight_params=self.topology.persistence_image.weight.params,
                kernel=self.topology.persistence_image.kernel.type,
                kernel_params=self.topology.persistence_image.kernel.params,
            ),
            stability_threshold=self.topology.stability_threshold,
            h1_threshold=self.topology.h1_threshold,
            min_component_size=self.topology.min_component_size,
            point_scale=self.topology.point_scale,
        )

        noise = None
        if self.noise is not None and self.noise.enabled and self.run.mode == "discover":
            noise = NoiseProfile(
                levels=self.noise.levels,
                trials=self.noise.trials_per_level,
                type=self.noise.type,
            )

        if noise is None:
            noise = NoiseProfile(levels=[0.0], trials=1, type="gaussian")

        return DiscoverRequest(
            dataset=dataset,
            hypothesis_space=hypothesis,
            topology=topology,
            noise=noise,
            seed=self.run.seed,
        )

    def dataset_provenance(self) -> dict[str, Any] | None:
        return self._dataset_provenance

    def to_evaluation_policy(self) -> EvaluationPolicy:
        return EvaluationPolicy(
            acceptance=AcceptancePolicy(
                min_stable_components=self.evaluation.acceptance.min_stable_components,
                require_h1=self.evaluation.acceptance.require_h1,
                coeff_tol=self.evaluation.acceptance.coeff_tol,
            ),
            stability_score=StabilityScorePolicy(
                method=self.evaluation.stability_score.method,
                aggregation=self.evaluation.stability_score.aggregation,
                tolerance=self.evaluation.stability_score.tolerance,
                min_pi_energy=self.evaluation.stability_score.min_pi_energy,
            ),
        )

    def to_adaptive_config(self):
        if self.hypothesis_space.adaptive is None:
            return None
        adaptive = self.hypothesis_space.adaptive
        return AdaptiveHypercubeConfig(
            enabled=adaptive.enabled,
            refine_factor=adaptive.refine_factor,
            top_k=adaptive.top_k,
            min_component_size=adaptive.min_component_size,
            h0_threshold=adaptive.h0_threshold,
        )

    def to_store_config(self) -> StorageConfig:
        base_dir = self.output.paths.get("base_dir", "./runs")
        if self.output.store is None:
            return StorageConfig(backend="local", base_dir=base_dir)
        store = self.output.store
        return StorageConfig(
            backend=store.backend,
            base_dir=store.base_dir or base_dir,
            bucket=store.bucket,
            prefix=store.prefix,
            endpoint_url=store.endpoint_url,
            region=store.region,
            upload_timeout_sec=store.upload_timeout_sec,
            retry_max_attempts=store.retry_max_attempts,
            retry_backoff_sec=store.retry_backoff_sec,
            retry_max_backoff_sec=store.retry_max_backoff_sec,
        )

    def to_compute_budget(self) -> ComputeBudget:
        budget = self.compute_budget
        return ComputeBudget(
            max_hypotheses=budget.max_hypotheses,
            max_noise_trials=budget.max_noise_trials,
            max_total_runs=budget.max_total_runs,
            max_model_evals=budget.max_model_evals,
            symbolic_regression_timeout_sec=budget.symbolic_regression_timeout_sec,
            ph_timeout_sec=budget.ph_timeout_sec,
            max_request_wall_time_sec=budget.max_request_wall_time_sec,
            max_point_cloud_points=budget.max_point_cloud_points,
        )


_PARAM_RE = re.compile(r"\^([A-Za-z_][A-Za-z0-9_]*)")


def _infer_params_from_expression(expr: str) -> List[str]:
    return _PARAM_RE.findall(expr or "")


__all__ = [
    "AcceptanceConfig",
    "AdaptiveConfig",
    "ArtifactStoreConfig",
    "ComputeBudgetConfig",
    "DatasetConfig",
    "DatasetLimitsConfig",
    "DatasetX",
    "DatasetY",
    "EvaluationConfig",
    "FeatureSpec",
    "HypothesisSpaceConfig",
    "NoiseConfig",
    "OutputConfig",
    "ParameterSpec",
    "PersistentHomologyConfig",
    "PIKernel",
    "PIWeight",
    "PersistenceImageConfig",
    "RegularizationSpec",
    "ReproducibilityConfig",
    "RunConfig",
    "StabilityScoreConfig",
    "TGEFDConfig",
    "TopologyConfig",
    "ValidationError",
]
