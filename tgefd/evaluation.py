from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

METHOD_VERSION = "tgefd-method-v1"

# Rejection reasons are explicit policy outcomes, not runtime errors.
REJECTION_REASON_DEFINITIONS = {
    "no_stable_components": "Stable component count is below policy threshold.",
    "no_h1": "H1 support is required by policy but absent.",
    "pi_energy_unavailable": "Persistence image energy could not be computed.",
    "pi_energy_below_threshold": "Mean persistence image energy is below policy threshold.",
    "topological_instability": "Energy variation across trials exceeds policy tolerance.",
}


@dataclass(frozen=True)
class AcceptancePolicy:
    min_stable_components: int = 1
    require_h1: bool = False
    coeff_tol: float = 1e-6


@dataclass(frozen=True)
class StabilityScorePolicy:
    method: Literal["pi_energy"] = "pi_energy"
    aggregation: Literal["mean", "median"] = "mean"
    tolerance: float = 0.15
    min_pi_energy: float = 0.0


@dataclass(frozen=True)
class EvaluationPolicy:
    acceptance: AcceptancePolicy = AcceptancePolicy()
    stability_score: StabilityScorePolicy = StabilityScorePolicy()


def decision_reason(
    stable_components: int,
    significant_h1: int,
    energy_cv: float | None,
    energy_mean: float | None,
    policy: EvaluationPolicy,
) -> tuple[bool, str]:
    if stable_components < policy.acceptance.min_stable_components:
        return False, "no_stable_components"
    if policy.acceptance.require_h1 and significant_h1 <= 0:
        return False, "no_h1"
    if energy_cv is None:
        return False, "pi_energy_unavailable"
    if energy_mean is None:
        return False, "pi_energy_unavailable"
    if energy_mean < policy.stability_score.min_pi_energy:
        return False, "pi_energy_below_threshold"
    if energy_cv > policy.stability_score.tolerance:
        return False, "topological_instability"
    return True, "topological stability under noise"


def decision_reason_per_noise(
    stable_components: int,
    significant_h1: int,
    energy_mean: float | None,
    policy: EvaluationPolicy,
) -> str:
    if stable_components < policy.acceptance.min_stable_components:
        return "no_stable_components"
    if policy.acceptance.require_h1 and significant_h1 <= 0:
        return "no_h1"
    if energy_mean is None:
        return "pi_energy_unavailable"
    if energy_mean < policy.stability_score.min_pi_energy:
        return "pi_energy_below_threshold"
    return "ok"
