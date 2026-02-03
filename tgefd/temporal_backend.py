from __future__ import annotations

import asyncio
from datetime import timedelta
from dataclasses import dataclass, replace
import hashlib
import os
import uuid
from typing import Any

from .api_v1 import discover, evaluate
from .config.models import TGEFDConfig
from .artifacts import canonical_config_hash, artifact_id_from_hash, write_artifact_bundle
from .storage import build_store

_TEMPORAL_IMPORT_ERROR: Exception | None = None
try:
    from temporalio import activity, workflow
    from temporalio.client import Client, WorkflowAlreadyStartedError
    from temporalio.common import RetryPolicy
except Exception as exc:  # pragma: no cover - optional dependency
    activity = None
    workflow = None
    Client = None
    WorkflowAlreadyStartedError = None
    RetryPolicy = None
    _TEMPORAL_IMPORT_ERROR = exc


@dataclass(frozen=True)
class TemporalConfig:
    target: str
    namespace: str
    task_queue: str
    workflow_timeout_sec: float
    activity_timeout_sec: float
    activity_max_attempts: int


def temporal_config_from_env() -> TemporalConfig:
    return TemporalConfig(
        target=os.getenv("TGEFD_TEMPORAL_TARGET", "localhost:7233"),
        namespace=os.getenv("TGEFD_TEMPORAL_NAMESPACE", "default"),
        task_queue=os.getenv("TGEFD_TEMPORAL_TASK_QUEUE", "tgefd"),
        workflow_timeout_sec=float(os.getenv("TGEFD_TEMPORAL_WORKFLOW_TIMEOUT_SEC", "3600")),
        activity_timeout_sec=float(os.getenv("TGEFD_TEMPORAL_ACTIVITY_TIMEOUT_SEC", "600")),
        activity_max_attempts=int(os.getenv("TGEFD_TEMPORAL_ACTIVITY_MAX_ATTEMPTS", "3")),
    )


def _require_temporal() -> None:
    if Client is None or workflow is None or activity is None:
        raise ImportError(
            "Temporal backend requires temporalio. Install with `pip install -e \".[temporal]\"`."
        ) from _TEMPORAL_IMPORT_ERROR


_CLIENT: Client | None = None
_CLIENT_LOCK = asyncio.Lock()


async def get_client(cfg: TemporalConfig) -> Client:
    _require_temporal()
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    async with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = await Client.connect(cfg.target, namespace=cfg.namespace)
    return _CLIENT


def workflow_id_from_idempotency(key: str) -> str:
    normalized = (key or "").strip()
    if not normalized:
        return ""
    if len(normalized) <= 64 and normalized.isalnum():
        return f"tgefd-{normalized}"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"tgefd-{digest}"


def _retry_policy(cfg: TemporalConfig):
    if RetryPolicy is None:
        return None
    return RetryPolicy(
        maximum_attempts=cfg.activity_max_attempts,
    )


if activity is None or workflow is None:  # pragma: no cover - optional dependency
    def discover_activity(config_payload: dict[str, Any]) -> dict[str, Any]:
        _require_temporal()
        raise RuntimeError("Temporal is not available")

    def evaluate_activity(config_payload: dict[str, Any]) -> dict[str, Any]:
        _require_temporal()
        raise RuntimeError("Temporal is not available")

    class DiscoverWorkflow:  # noqa: D101 - placeholder when temporal unavailable
        pass

    class EvaluateWorkflow:  # noqa: D101 - placeholder when temporal unavailable
        pass
else:
    @activity.defn
    def discover_activity(config_payload: dict[str, Any]) -> dict[str, Any]:
        cfg = TGEFDConfig.model_validate(config_payload)
        evaluation = cfg.to_evaluation_policy()
        adaptive = cfg.to_adaptive_config()
        budget = cfg.to_compute_budget()
        req = cfg.to_request()
        response = discover(req, evaluation=evaluation, adaptive=adaptive, budget=budget)
        if cfg.output.save_artifacts:
            store_config = cfg.to_store_config()
            store = build_store(store_config)
            config_hash = canonical_config_hash(config_payload)
            artifact_id = artifact_id_from_hash(config_hash)
            artifact_uri = store.uri_for(artifact_id)
            response = replace(
                response,
                artifacts=replace(
                    response.artifacts,
                    artifact_id=artifact_id,
                    artifact_uri=artifact_uri,
                ),
                reproducibility=replace(
                    response.reproducibility,
                    config_hash=config_hash,
                ),
            )
            info = write_artifact_bundle(
                config_payload,
                response,
                base_dir=store_config.base_dir,
                artifact_uri=artifact_uri,
                dataset_provenance=cfg.dataset_provenance(),
            )
            store.store(info.artifact_dir, info.artifact_id)
        return response.to_dict()

    @activity.defn
    def evaluate_activity(config_payload: dict[str, Any]) -> dict[str, Any]:
        cfg = TGEFDConfig.model_validate(config_payload)
        evaluation = cfg.to_evaluation_policy()
        adaptive = cfg.to_adaptive_config()
        budget = cfg.to_compute_budget()
        req = cfg.to_request()
        response = evaluate(req, evaluation=evaluation, adaptive=adaptive, budget=budget)
        return response.to_dict()

    @workflow.defn
    class DiscoverWorkflow:
        @workflow.run
        async def run(self, config_payload: dict[str, Any]) -> dict[str, Any]:
            cfg = temporal_config_from_env()
        return await workflow.execute_activity(
            discover_activity,
            config_payload,
            start_to_close_timeout=timedelta(seconds=cfg.activity_timeout_sec),
            retry_policy=_retry_policy(cfg),
        )

    @workflow.defn
    class EvaluateWorkflow:
        @workflow.run
        async def run(self, config_payload: dict[str, Any]) -> dict[str, Any]:
            cfg = temporal_config_from_env()
        return await workflow.execute_activity(
            evaluate_activity,
            config_payload,
            start_to_close_timeout=timedelta(seconds=cfg.activity_timeout_sec),
            retry_policy=_retry_policy(cfg),
        )


async def submit_workflow(
    config_payload: dict[str, Any],
    *,
    workflow_name: str,
    workflow_id: str | None,
) -> str:
    cfg = temporal_config_from_env()
    client = await get_client(cfg)
    workflow_id = workflow_id or f"tgefd-{uuid.uuid4().hex}"
    try:
        handle = await client.start_workflow(
            workflow_name,
            config_payload,
            id=workflow_id,
            task_queue=cfg.task_queue,
            execution_timeout=timedelta(seconds=cfg.workflow_timeout_sec),
        )
        return handle.id
    except Exception as exc:  # pragma: no cover - temporal optional
        if WorkflowAlreadyStartedError and isinstance(exc, WorkflowAlreadyStartedError):
            return workflow_id
        raise


async def get_workflow_status(job_id: str) -> dict[str, Any]:
    cfg = temporal_config_from_env()
    client = await get_client(cfg)
    handle = client.get_workflow_handle(job_id)
    description = await handle.describe()
    status = getattr(description, "status", None)
    if status is not None and hasattr(status, "name"):
        status_name = status.name.lower()
    else:
        status_name = str(status).lower() if status is not None else "unknown"
    return {"job_id": job_id, "status": status_name}


async def get_workflow_result(job_id: str, *, timeout_sec: float) -> dict[str, Any]:
    cfg = temporal_config_from_env()
    client = await get_client(cfg)
    handle = client.get_workflow_handle(job_id)
    try:
        result = await asyncio.wait_for(handle.result(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        return {"job_id": job_id, "status": "running"}
    return {"job_id": job_id, "status": "completed", "result": result}
