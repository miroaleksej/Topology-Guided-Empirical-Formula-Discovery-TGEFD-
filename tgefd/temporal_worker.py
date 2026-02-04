from __future__ import annotations

import asyncio

from temporalio.worker import Worker

from .temporal_backend import (
    DiscoverWorkflow,
    EvaluateWorkflow,
    discover_activity,
    evaluate_activity,
    get_client,
    temporal_config_from_env,
)


async def _run_worker() -> None:
    cfg = temporal_config_from_env()
    client = await get_client(cfg)
    worker = Worker(
        client,
        task_queue=cfg.task_queue,
        workflows=[DiscoverWorkflow, EvaluateWorkflow],
        activities=[discover_activity, evaluate_activity],
    )
    await worker.run()


def main() -> None:
    asyncio.run(_run_worker())


if __name__ == "__main__":
    main()
