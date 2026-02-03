from __future__ import annotations

import json
from typing import Mapping

from pydantic import ValidationError

from .models import (
    AcceptanceConfig,
    AdaptiveConfig,
    ArtifactStoreConfig,
    ComputeBudgetConfig,
    DatasetConfig,
    DatasetLimitsConfig,
    DatasetX,
    DatasetY,
    EvaluationConfig,
    FeatureSpec,
    HypothesisSpaceConfig,
    NoiseConfig,
    OutputConfig,
    ParameterSpec,
    PersistentHomologyConfig,
    PIKernel,
    PIWeight,
    PersistenceImageConfig,
    RegularizationSpec,
    ReproducibilityConfig,
    RunConfig,
    StabilityScoreConfig,
    TGEFDConfig,
    TopologyConfig,
)

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
    "load_config",
    "parse_config",
    "config_to_request",
]


def load_config(path: str) -> dict:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    if path.endswith(".yaml") or path.endswith(".yml"):
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise ImportError(
                "YAML config requires PyYAML. Install with `pip install pyyaml`."
            ) from exc
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    raise ValueError("Config file must be .json or .yaml")


def parse_config(data: Mapping[str, object]) -> TGEFDConfig:
    return TGEFDConfig.model_validate(data)


def config_to_request(data: Mapping[str, object]):
    return parse_config(data).to_request()
