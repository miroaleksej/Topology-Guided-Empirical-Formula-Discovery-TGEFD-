from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import queue as queue_module
import traceback
import re
import hashlib
import json
import time
from typing import Iterable, Mapping, Sequence
import uuid

import numpy as np

from .models import HypothesisParams, ModelResult
from .adaptive import AdaptiveHypercubeConfig, adaptive_search_models
from .evaluation import METHOD_VERSION, EvaluationPolicy, decision_reason, decision_reason_per_noise
from .search import hypothesis_hypercube, rank_results, search_models
from .features import format_feature_expression, required_params_from_features
from .topology import (
    build_point_cloud,
    persistent_components,
    persistent_homology,
    select_stable_models,
    significant_features,
)
from .topo_vectors import image_energy, persistent_image

_DEFAULT_FEATURES = ("1", "x", "x^p", "sin(x)", "sin(x)^q", "exp(-x)")
_NUM_RE = re.compile(r"^[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?$")
_DATASET_HARD_MAX_POINTS = 200_000
_DATASET_HARD_MAX_FEATURES = 2_048


class BudgetExceededError(ValueError):
    """Raised when compute budget or timeout guardrails are violated."""


def _is_supported_feature(expr: str) -> bool:
    expr = expr.replace(" ", "")
    if not expr:
        return False
    factors = expr.split("*")
    for factor in factors:
        if not _is_supported_factor(factor):
            return False
    return True


def _is_supported_factor(token: str) -> bool:
    if "^" in token:
        base, exp = token.split("^", 1)
        if not exp or not (_NUM_RE.match(exp) or exp.isidentifier()):
            return False
        return _is_supported_base(base)
    return _is_supported_base(token)


def _is_supported_base(token: str) -> bool:
    token = token.replace(" ", "")
    if token in {"1", "const"}:
        return True
    if token == "x":
        return True
    if re.fullmatch(r"x\d+", token):
        return True
    if token.startswith("sin(") and token.endswith(")"):
        return _is_supported_base(token[4:-1])
    if token.startswith("cos(") and token.endswith(")"):
        return _is_supported_base(token[4:-1])
    if token.startswith("exp(") and token.endswith(")"):
        inner = token[4:-1]
        if inner.startswith("-"):
            return _is_supported_base(inner[1:])
        return False
    return False


@dataclass(frozen=True)
class Dataset:
    id: str
    x: np.ndarray
    y: np.ndarray
    metadata: Mapping[str, str] | None = None

    def validate(self) -> None:
        x = np.asarray(self.x)
        y = np.asarray(self.y)
        if x.ndim not in (1, 2):
            raise ValueError("dataset.x must be 1D or 2D")
        if y.ndim != 1:
            raise ValueError("dataset.y must be 1D")
        if x.shape[0] != y.shape[0]:
            raise ValueError("dataset.x and dataset.y must have matching length")
        if x.shape[0] < 1:
            raise ValueError("dataset must contain at least one point")
        feature_count = 1 if x.ndim == 1 else x.shape[1]
        if x.shape[0] > _DATASET_HARD_MAX_POINTS:
            raise ValueError(
                f"dataset has {x.shape[0]} points; hard safety limit is {_DATASET_HARD_MAX_POINTS}"
            )
        if feature_count > _DATASET_HARD_MAX_FEATURES:
            raise ValueError(
                "dataset has "
                f"{feature_count} features; hard safety limit is {_DATASET_HARD_MAX_FEATURES}"
            )
        if not np.isfinite(x).all():
            raise ValueError("dataset.x must contain only finite numeric values")
        if not np.isfinite(y).all():
            raise ValueError("dataset.y must contain only finite numeric values")

    def to_dict(self) -> dict:
        x = np.asarray(self.x).tolist()
        y = np.asarray(self.y).tolist()
        return {
            "id": self.id,
            "x": x,
            "y": y,
            "metadata": dict(self.metadata) if self.metadata else {},
        }


@dataclass(frozen=True)
class HypothesisSpace:
    features: Sequence[str]
    parameters: Mapping[str, Sequence[float]]
    regularization: Mapping[str, Sequence[float]]

    def validate(self) -> None:
        if not self.features:
            raise ValueError("features must be non-empty")
        required = required_params_from_features(self.features)
        for param in required:
            if param not in self.parameters:
                raise ValueError(f"Feature uses undefined parameter '{param}'")
        for expr in self.features:
            if not _is_supported_feature(expr):
                raise ValueError(f"Unsupported feature expression '{expr}'")
        if "lambda" not in self.regularization:
            raise ValueError("regularization.lambda must be provided")

    def to_dict(self) -> dict:
        return {
            "features": list(self.features),
            "parameters": {k: list(v) for k, v in self.parameters.items()},
            "regularization": {k: list(v) for k, v in self.regularization.items()},
        }

    def hypercube(self) -> list[HypothesisParams]:
        p_range = list(self.parameters.get("p", [1.0]))
        q_range = list(self.parameters.get("q", [1.0]))
        lam_range = list(self.regularization.get("lambda", [1e-3]))
        return hypothesis_hypercube(p_range, q_range, lam_range)


@dataclass(frozen=True)
class PHConfig:
    max_dim: int = 1
    metric: str = "euclidean"

    def to_dict(self) -> dict:
        return {"max_dim": self.max_dim, "metric": self.metric}


@dataclass(frozen=True)
class PersistenceImageConfig:
    birth_range: tuple[float, float] = (0.0, 1.0)
    pers_range: tuple[float, float] = (0.0, 1.0)
    pixel_size: float = 0.05
    weight: str = "persistence"
    weight_params: Mapping[str, float] | None = None
    kernel: str = "gaussian"
    kernel_params: Mapping[str, list[list[float]]] | None = None

    def to_dict(self) -> dict:
        return {
            "birth_range": list(self.birth_range),
            "pers_range": list(self.pers_range),
            "pixel_size": self.pixel_size,
            "weight": self.weight,
            "weight_params": dict(self.weight_params) if self.weight_params else {"n": 2.0},
            "kernel": self.kernel,
            "kernel_params": dict(self.kernel_params)
            if self.kernel_params
            else {"sigma": [[0.05, 0.0], [0.0, 0.05]]},
        }


@dataclass(frozen=True)
class TopologyConfig:
    ph: PHConfig = PHConfig()
    persistence_image: PersistenceImageConfig = PersistenceImageConfig()
    stability_threshold: float = 0.2
    h1_threshold: float | None = None
    min_component_size: int = 5
    point_scale: str = "standard"

    def resolve_h1_threshold(self) -> float:
        if self.h1_threshold is None:
            return self.stability_threshold * 0.5
        return self.h1_threshold

    def to_dict(self) -> dict:
        return {
            "ph": self.ph.to_dict(),
            "persistence_image": self.persistence_image.to_dict(),
            "stability_threshold": self.stability_threshold,
            "h1_threshold": self.h1_threshold,
            "min_component_size": self.min_component_size,
            "point_scale": self.point_scale,
        }


@dataclass(frozen=True)
class NoiseProfile:
    levels: Sequence[float]
    trials: int = 1
    type: str = "gaussian"

    def validate(self) -> None:
        if self.trials < 1:
            raise ValueError("noise.trials must be at least 1")
        if self.type != "gaussian":
            raise ValueError("noise.type must be 'gaussian'")

    def to_dict(self) -> dict:
        return {
            "levels": list(self.levels),
            "trials": self.trials,
            "type": self.type,
        }


@dataclass(frozen=True)
class ComputeBudget:
    max_hypotheses: int = 50_000
    max_noise_trials: int = 32
    max_total_runs: int = 256
    max_model_evals: int = 2_000_000
    symbolic_regression_timeout_sec: float = 10.0
    ph_timeout_sec: float = 10.0
    max_request_wall_time_sec: float = 60.0
    max_point_cloud_points: int = 5_000

    def validate(self) -> None:
        int_limits = {
            "max_hypotheses": self.max_hypotheses,
            "max_noise_trials": self.max_noise_trials,
            "max_total_runs": self.max_total_runs,
            "max_model_evals": self.max_model_evals,
            "max_point_cloud_points": self.max_point_cloud_points,
        }
        for name, value in int_limits.items():
            if value < 1:
                raise ValueError(f"compute budget {name} must be at least 1")
        float_limits = {
            "symbolic_regression_timeout_sec": self.symbolic_regression_timeout_sec,
            "ph_timeout_sec": self.ph_timeout_sec,
            "max_request_wall_time_sec": self.max_request_wall_time_sec,
        }
        for name, value in float_limits.items():
            if value <= 0.0:
                raise ValueError(f"compute budget {name} must be positive")


@dataclass(frozen=True)
class DiscoverRequest:
    dataset: Dataset
    hypothesis_space: HypothesisSpace
    topology: TopologyConfig
    noise: NoiseProfile
    seed: int = 0

    def validate(self) -> None:
        self.dataset.validate()
        self.hypothesis_space.validate()
        self.noise.validate()
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset.to_dict(),
            "hypothesis_space": self.hypothesis_space.to_dict(),
            "topology": self.topology.to_dict(),
            "noise": self.noise.to_dict(),
            "seed": self.seed,
        }


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reason: str

    def to_dict(self) -> dict:
        return {"accepted": self.accepted, "reason": self.reason}


@dataclass(frozen=True)
class ErrorStats:
    mean: float
    std: float

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}


@dataclass(frozen=True)
class RepresentativeModel:
    formula: str
    coefficients: Mapping[str, float]

    def to_dict(self) -> dict:
        return {"formula": self.formula, "coefficients": dict(self.coefficients)}


@dataclass(frozen=True)
class ModelFamily:
    family_id: str
    representative_model: RepresentativeModel
    invariant_terms: Sequence[str]
    unstable_terms: Sequence[str]
    error: ErrorStats

    def to_dict(self) -> dict:
        return {
            "family_id": self.family_id,
            "representative_model": self.representative_model.to_dict(),
            "invariant_terms": list(self.invariant_terms),
            "unstable_terms": list(self.unstable_terms),
            "error": self.error.to_dict(),
        }


@dataclass(frozen=True)
class TopologyEnergy:
    mean: float
    std: float

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}


@dataclass(frozen=True)
class TopologyReport:
    H0: int
    H1: int
    PI_energy: TopologyEnergy | None = None

    def to_dict(self) -> dict:
        return {
            "H0": self.H0,
            "H1": self.H1,
            "PI_energy": self.PI_energy.to_dict() if self.PI_energy else None,
        }


@dataclass(frozen=True)
class DiscoverSummary:
    stable_components: int
    significant_H0: int
    significant_H1: int
    stability_score: float

    def to_dict(self) -> dict:
        return {
            "stable_components": self.stable_components,
            "significant_H0": self.significant_H0,
            "significant_H1": self.significant_H1,
            "stability_score": self.stability_score,
        }


@dataclass(frozen=True)
class NoiseReportEntry:
    noise_level: float
    stable_components: int
    significant_H0: int
    significant_H1: int
    pi_energy: float | None
    decision: str

    def to_dict(self) -> dict:
        return {
            "noise_level": self.noise_level,
            "stable_components": self.stable_components,
            "significant_H0": self.significant_H0,
            "significant_H1": self.significant_H1,
            "pi_energy": self.pi_energy,
            "decision": self.decision,
        }


@dataclass(frozen=True)
class Artifacts:
    persistence_diagrams: str | None = None
    persistence_images: str | None = None
    artifact_id: str | None = None
    artifact_uri: str | None = None

    def to_dict(self) -> dict:
        return {
            "persistence_diagrams": self.persistence_diagrams,
            "persistence_images": self.persistence_images,
            "artifact_id": self.artifact_id,
            "artifact_uri": self.artifact_uri,
        }


@dataclass(frozen=True)
class Reproducibility:
    config_hash: str
    seed: int
    deterministic: bool
    version: str
    method_version: str

    def to_dict(self) -> dict:
        return {
            "config_hash": self.config_hash,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "version": self.version,
            "method_version": self.method_version,
        }


@dataclass(frozen=True)
class DiscoverResponse:
    run_id: str
    status: str
    summary: DiscoverSummary
    decision: Decision
    models: Sequence[ModelFamily]
    noise_report: Sequence[NoiseReportEntry]
    topology_report: TopologyReport
    artifacts: Artifacts
    reproducibility: Reproducibility

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "summary": self.summary.to_dict(),
            "decision": self.decision.to_dict(),
            "models": [m.to_dict() for m in self.models],
            "noise_report": [r.to_dict() for r in self.noise_report],
            "topology_report": self.topology_report.to_dict(),
            "artifacts": self.artifacts.to_dict(),
            "reproducibility": self.reproducibility.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True)


def _config_hash(request: DiscoverRequest) -> str:
    payload = request.to_dict()
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _get_version() -> str:
    try:
        from importlib.metadata import version

        return f"tgefd-{version('tgefd')}"
    except Exception:
        return "tgefd-dev"


def _feature_labels(
    feature_exprs: Sequence[str] | None,
    params: HypothesisParams | None = None,
    substitute: bool = False,
) -> list[str]:
    exprs = list(feature_exprs) if feature_exprs else list(_DEFAULT_FEATURES)
    if not substitute or params is None:
        return exprs
    params_map = {"p": float(params.p), "q": float(params.q)}
    return [format_feature_expression(expr, params_map) for expr in exprs]


def _describe_model(
    params: HypothesisParams,
    coeffs: np.ndarray,
    feature_exprs: Sequence[str] | None,
) -> str:
    labels = _feature_labels(feature_exprs, params=params, substitute=True)
    parts = []
    for name, coef in zip(labels, coeffs):
        if coef == 0:
            continue
        parts.append(f"({coef:+.4g})*{name}")
    if not parts:
        return "0"
    return "y = " + " + ".join(parts)


def _coeff_dict(
    params: HypothesisParams,
    coeffs: np.ndarray,
    feature_exprs: Sequence[str] | None,
) -> dict[str, float]:
    coeffs = np.asarray(coeffs, dtype=float)
    labels = _feature_labels(feature_exprs, params=params, substitute=True)
    return {name: float(c) for name, c in zip(labels, coeffs) if c != 0}


def _invariant_terms(
    models: Sequence[ModelResult],
    feature_exprs: Sequence[str] | None,
    tol: float = 1e-6,
) -> tuple[list[str], list[str]]:
    if not models:
        return [], []
    coeffs = np.vstack([m.coeffs for m in models])
    presence = np.mean(np.abs(coeffs) > tol, axis=0)
    names = _feature_labels(feature_exprs)
    invariant = [name for name, p in zip(names, presence) if p >= 0.8]
    unstable = [name for name, p in zip(names, presence) if 0 < p < 0.8]
    return invariant, unstable


def _topo_energy(diagram: np.ndarray, cfg: PersistenceImageConfig) -> float | None:
    if diagram.size == 0:
        return 0.0
    img = persistent_image(
        diagram,
        birth_range=cfg.birth_range,
        pers_range=cfg.pers_range,
        pixel_size=cfg.pixel_size,
        weight=cfg.weight,
        weight_params=dict(cfg.weight_params) if cfg.weight_params else None,
        kernel=cfg.kernel,
        kernel_params=dict(cfg.kernel_params) if cfg.kernel_params else None,
        skew=True,
    )
    return image_energy(img)


def _run_with_timeout(
    fn,
    *,
    timeout_sec: float,
    stage: str,
    args: tuple,
    kwargs: Mapping[str, object] | None = None,
):
    kwargs = dict(kwargs or {})
    if timeout_sec <= 0:
        return fn(*args, **kwargs)
    try:
        return _run_in_subprocess(
            fn,
            args=args,
            kwargs=kwargs,
            timeout_sec=timeout_sec,
            stage=stage,
        )
    except _TimeoutStartError:
        started = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - started
        if elapsed > timeout_sec:
            raise BudgetExceededError(
                f"budget exceeded: {stage} timeout ({elapsed:.2f}s > {timeout_sec:.2f}s)"
            )
        return result


class _TimeoutStartError(RuntimeError):
    pass


def _run_in_subprocess(
    fn,
    *,
    args: tuple,
    kwargs: Mapping[str, object],
    timeout_sec: float,
    stage: str,
):
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue(maxsize=1)

    def _worker(queue_handle, target, target_args, target_kwargs):
        try:
            result = target(*target_args, **target_kwargs)
            try:
                queue_handle.put(("ok", result))
            except Exception as exc:
                queue_handle.put(("err", RuntimeError(f"result not serializable: {exc}")))
        except Exception as exc:
            try:
                queue_handle.put(("err", exc))
            except Exception:
                queue_handle.put(
                    (
                        "err",
                        RuntimeError(
                            f"{exc.__class__.__name__}: {exc}\n{traceback.format_exc()}"
                        ),
                    )
                )

    proc = ctx.Process(target=_worker, args=(result_queue, fn, args, kwargs))
    proc.daemon = True
    started = time.perf_counter()
    try:
        proc.start()
    except Exception as exc:
        raise _TimeoutStartError(str(exc)) from exc
    proc.join(timeout_sec)
    elapsed = time.perf_counter() - started
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        raise BudgetExceededError(
            f"budget exceeded: {stage} timeout ({elapsed:.2f}s > {timeout_sec:.2f}s)"
        )

    try:
        status, payload = result_queue.get_nowait()
    except queue_module.Empty as exc:
        raise RuntimeError(f"{stage} failed without producing a result") from exc
    if status == "ok":
        return payload
    if isinstance(payload, Exception):
        raise payload
    raise RuntimeError(f"{stage} failed: {payload}")


def _sample_results_for_topology(
    results: Sequence[ModelResult],
    max_points: int,
    rng: np.random.Generator,
) -> Sequence[ModelResult]:
    if max_points < 1:
        return results
    if len(results) <= max_points:
        return results
    indices = rng.choice(len(results), size=max_points, replace=False)
    ordered = sorted(int(i) for i in indices)
    return [results[i] for i in ordered]


def _guard_runtime_budget(
    *,
    budget: ComputeBudget,
    hypercube_size: int,
    noise_trials: int,
    noise_levels: int,
    adaptive: AdaptiveHypercubeConfig | None,
) -> None:
    if hypercube_size > budget.max_hypotheses:
        raise BudgetExceededError(
            f"budget exceeded: hypothesis hypercube size {hypercube_size} > max {budget.max_hypotheses}"
        )
    if noise_trials > budget.max_noise_trials:
        raise BudgetExceededError(
            f"budget exceeded: noise trials {noise_trials} > max {budget.max_noise_trials}"
        )
    total_runs = noise_trials * noise_levels
    if total_runs > budget.max_total_runs:
        raise BudgetExceededError(
            f"budget exceeded: total runs {total_runs} > max {budget.max_total_runs}"
        )
    adaptive_multiplier = 2 if adaptive is not None and adaptive.enabled else 1
    model_evals = hypercube_size * total_runs * adaptive_multiplier
    if model_evals > budget.max_model_evals:
        raise BudgetExceededError(
            f"budget exceeded: model evaluations {model_evals} > max {budget.max_model_evals}"
        )


def _run_trial(
    x: np.ndarray,
    y: np.ndarray,
    hypercube: Iterable[HypothesisParams],
    topo: TopologyConfig,
    noise_level: float,
    rng: np.random.Generator,
    adaptive: AdaptiveHypercubeConfig | None,
    features: Sequence[str] | None,
    budget: ComputeBudget,
    started_at: float,
) -> dict[str, object]:
    elapsed_total = time.perf_counter() - started_at
    if elapsed_total > budget.max_request_wall_time_sec:
        raise BudgetExceededError(
            "budget exceeded: wall-time limit "
            f"({elapsed_total:.2f}s > {budget.max_request_wall_time_sec:.2f}s)"
        )

    if noise_level > 0:
        y = y + noise_level * rng.standard_normal(len(y))

    if adaptive is None:
        results = _run_with_timeout(
            search_models,
            timeout_sec=budget.symbolic_regression_timeout_sec,
            stage="symbolic regression",
            args=(x, y, hypercube),
            kwargs={"features": features},
        )
    else:
        results = _run_with_timeout(
            adaptive_search_models,
            timeout_sec=budget.symbolic_regression_timeout_sec,
            stage="symbolic regression",
            args=(x, y, hypercube, topo, adaptive),
            kwargs={
                "iters": 10,
                "normalize": True,
                "features": list(features) if features is not None else None,
            },
        )
    topology_results = _sample_results_for_topology(
        results, budget.max_point_cloud_points, rng
    )
    X = build_point_cloud(topology_results, scale=topo.point_scale)
    diagrams = _run_with_timeout(
        persistent_homology,
        timeout_sec=budget.ph_timeout_sec,
        stage="persistent homology",
        args=(X,),
        kwargs={"maxdim": topo.ph.max_dim},
    )

    h0_sig = significant_features(diagrams[0], persistence_threshold=topo.stability_threshold)
    h1_sig = []
    if len(diagrams) > 1:
        h1_sig = significant_features(diagrams[1], persistence_threshold=topo.resolve_h1_threshold())

    components = persistent_components(
        X,
        persistence_threshold=topo.stability_threshold,
        min_component_size=topo.min_component_size,
    )
    stable = select_stable_models(topology_results, components)

    energy = None
    if len(diagrams) > 1:
        try:
            energy = _topo_energy(diagrams[1], topo.persistence_image)
        except ImportError:
            energy = None

    return {
        "results": results,
        "stable": stable,
        "topology_size": len(topology_results),
        "components": components,
        "h0_sig": h0_sig,
        "h1_sig": h1_sig,
        "energy": energy,
        "noise_level": noise_level,
    }


def discover(
    request: DiscoverRequest,
    evaluation: EvaluationPolicy | None = None,
    adaptive: AdaptiveHypercubeConfig | None = None,
    budget: ComputeBudget | None = None,
) -> DiscoverResponse:
    request.validate()
    effective_budget = budget or ComputeBudget()
    effective_budget.validate()

    rng = np.random.default_rng(request.seed)
    hypercube = request.hypothesis_space.hypercube()
    feature_exprs = request.hypothesis_space.features
    _guard_runtime_budget(
        budget=effective_budget,
        hypercube_size=len(hypercube),
        noise_trials=request.noise.trials,
        noise_levels=len(request.noise.levels),
        adaptive=adaptive,
    )

    runs = []
    started_at = time.perf_counter()
    for level in request.noise.levels:
        for _ in range(request.noise.trials):
            runs.append(
                _run_trial(
                    np.asarray(request.dataset.x),
                    np.asarray(request.dataset.y),
                    hypercube,
                    request.topology,
                    noise_level=float(level),
                    rng=rng,
                    adaptive=adaptive,
                    features=feature_exprs,
                    budget=effective_budget,
                    started_at=started_at,
                )
            )

    if not runs:
        raise ValueError("noise profile produced no runs")

    component_counts = [len(r["components"]) for r in runs]
    h0_counts = [len(r["h0_sig"]) for r in runs]
    h1_counts = [len(r["h1_sig"]) for r in runs]
    stable_counts = [len(r["stable"]) for r in runs]

    stable_by_noise: dict[float, set[tuple[int, ...]]] = {}
    for level in request.noise.levels:
        level_patterns: list[set[tuple[int, ...]]] = []
        for r in runs:
            if float(r["noise_level"]) != float(level):
                continue
            patterns = {
                _coeff_pattern(m, evaluation.acceptance.coeff_tol if evaluation else 1e-6)
                for m in r["stable"]
            }
            level_patterns.append(patterns)
        if not level_patterns:
            level_set = set()
        elif len(level_patterns) == 1:
            level_set = level_patterns[0]
        else:
            level_set = set.intersection(*level_patterns)
        stable_by_noise[float(level)] = level_set

    stable_intersection: set[tuple[int, ...]] = set()
    if stable_by_noise:
        stable_intersection = set.intersection(*stable_by_noise.values())

    topology_sizes = [
        max(int(r.get("topology_size", len(r["results"]))), 1) for r in runs
    ]
    stable_frac = float(np.mean([c / size for c, size in zip(stable_counts, topology_sizes)]))

    energies = [r["energy"] for r in runs if r["energy"] is not None]
    mean_energy = None
    std_energy = None
    if energies:
        mean_energy = float(np.mean(energies))
        std_energy = float(np.std(energies))
        cv = std_energy / mean_energy if mean_energy > 0 else 1.0
        energy_score = 1.0 / (1.0 + cv)
        stability_score = stable_frac * energy_score
        energy_summary = TopologyEnergy(mean=mean_energy, std=std_energy)
    else:
        stability_score = stable_frac
        energy_summary = None

    summary = DiscoverSummary(
        stable_components=len(stable_intersection),
        significant_H0=min(h0_counts) if h0_counts else 0,
        significant_H1=min(h1_counts) if h1_counts else 0,
        stability_score=stability_score,
    )

    energy_cv = None
    if energies:
        energy_cv = std_energy / mean_energy if mean_energy > 0 else float("inf")

    if evaluation is None:
        accepted = summary.stable_components > 0 and summary.significant_H0 > 0
        reason = "topological stability under noise" if accepted else "no topologically stable model family detected"
    else:
        accepted, reason = decision_reason(
            summary.stable_components, summary.significant_H1, energy_cv, mean_energy, evaluation
        )

    decision = Decision(accepted=accepted, reason=reason)

    candidate_pool: list[ModelResult] = []
    if stable_intersection:
        for r in runs:
            for m in r["stable"]:
                pattern = _coeff_pattern(
                    m, evaluation.acceptance.coeff_tol if evaluation else 1e-6
                )
                if pattern in stable_intersection:
                    candidate_pool.append(m)
    if not candidate_pool:
        top_run = runs[0]
        candidate_pool = top_run["stable"] or top_run["results"]

    top = rank_results(candidate_pool, top_k=1)[0]

    if decision.accepted and stable_intersection:
        invariant_terms, unstable_terms = _invariant_terms(
            candidate_pool,
            feature_exprs,
            tol=evaluation.acceptance.coeff_tol if evaluation else 1e-6,
        )
    else:
        invariant_terms, unstable_terms = [], []

    coeffs = _coeff_dict(top.params, top.coeffs, feature_exprs)
    model_family = ModelFamily(
        family_id="fam-1" if decision.accepted else "candidate-1",
        representative_model=RepresentativeModel(
            formula=_describe_model(top.params, top.coeffs, feature_exprs),
            coefficients=coeffs,
        ),
        invariant_terms=invariant_terms,
        unstable_terms=unstable_terms,
        error=ErrorStats(
            mean=float(np.mean([m.error for m in candidate_pool])),
            std=float(np.std([m.error for m in candidate_pool])),
        ),
    )

    topology_report = TopologyReport(
        H0=summary.significant_H0,
        H1=summary.significant_H1,
        PI_energy=energy_summary,
    )

    noise_report = _build_noise_report(runs, evaluation or EvaluationPolicy())

    response = DiscoverResponse(
        run_id=str(uuid.uuid4()),
        status="accepted" if decision.accepted else "rejected",
        summary=summary,
        decision=decision,
        models=[model_family],
        noise_report=noise_report,
        topology_report=topology_report,
        artifacts=Artifacts(),
        reproducibility=Reproducibility(
            config_hash=_config_hash(request),
            seed=request.seed,
            deterministic=True,
            version=_get_version(),
            method_version=METHOD_VERSION,
        ),
    )
    return response


def _coeff_pattern(model: ModelResult, tol: float) -> tuple[int, ...]:
    return tuple(int(abs(c) > tol) for c in model.coeffs)


def _build_noise_report(
    runs: Sequence[dict[str, object]],
    evaluation: EvaluationPolicy,
) -> list[NoiseReportEntry]:
    by_level: dict[float, list[dict[str, object]]] = {}
    for r in runs:
        level = float(r["noise_level"])
        by_level.setdefault(level, []).append(r)

    report: list[NoiseReportEntry] = []
    for level in sorted(by_level.keys()):
        items = by_level[level]
        h0 = min(len(r["h0_sig"]) for r in items) if items else 0
        h1 = min(len(r["h1_sig"]) for r in items) if items else 0
        energies = [r["energy"] for r in items if r["energy"] is not None]
        energy_mean = float(np.mean(energies)) if energies else None
        patterns_per_trial = []
        for r in items:
            patterns_per_trial.append(
                {
                    _coeff_pattern(m, evaluation.acceptance.coeff_tol)
                    for m in r["stable"]
                }
            )
        if not patterns_per_trial:
            patterns = set()
        elif len(patterns_per_trial) == 1:
            patterns = patterns_per_trial[0]
        else:
            patterns = set.intersection(*patterns_per_trial)
        reason = decision_reason_per_noise(len(patterns), h1, energy_mean, evaluation)
        report.append(
            NoiseReportEntry(
                noise_level=level,
                stable_components=len(patterns),
                significant_H0=h0,
                significant_H1=h1,
                pi_energy=energy_mean,
                decision=reason,
            )
        )
    return report


def evaluate(
    request: DiscoverRequest,
    evaluation: EvaluationPolicy | None = None,
    adaptive: AdaptiveHypercubeConfig | None = None,
    budget: ComputeBudget | None = None,
) -> DiscoverResponse:
    quick = DiscoverRequest(
        dataset=request.dataset,
        hypothesis_space=request.hypothesis_space,
        topology=request.topology,
        noise=NoiseProfile(levels=[0.0], trials=1, type=request.noise.type),
        seed=request.seed,
    )
    return discover(quick, evaluation=evaluation, adaptive=adaptive, budget=budget)
