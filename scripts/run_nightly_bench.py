import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tgefd.bench_regressions import load_curves, compare_curves, format_regressions
from tgefd.ond_rng_bench import BenchConfig, load_bench_config, run_benchmark, save_benchmark_result, bench_config_hash


class NightlyRunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: str
    baseline: str | None = None
    metric: str | None = None
    max_drop: float = Field(0.0, ge=0.0)


class NightlyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    output_dir: str = "benchmarks/nightly"
    skip_if_exists: bool = True
    runs: list[NightlyRunSpec]

    @model_validator(mode="after")
    def _check_version(self) -> "NightlyConfig":
        if self.version != "1.0":
            raise ValueError("version must be '1.0'")
        if not self.runs:
            raise ValueError("runs must be non-empty")
        return self


def _run_dir_name(config_id: str, config_hash: str) -> str:
    short = config_hash.split(":", 1)[-1][:12]
    return f"{config_id}-{short}"


def _load_nightly(path: Path) -> NightlyConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return NightlyConfig.model_validate(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run nightly OND-RNG-Bench configs.")
    parser.add_argument("config", help="Path to nightly YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, do not run")
    parser.add_argument("--force", action="store_true", help="Run even if output exists")
    args = parser.parse_args()

    nightly = _load_nightly(Path(args.config))
    output_dir = Path(nightly.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "config": args.config,
        "output_dir": str(output_dir),
        "runs": [],
    }
    exit_code = 0

    for run_spec in nightly.runs:
        raw = load_bench_config(run_spec.config)
        try:
            bench_config = BenchConfig.model_validate(raw)
        except Exception as exc:
            summary["runs"].append(
                {
                    "config": run_spec.config,
                    "status": "invalid_config",
                    "error": str(exc),
                }
            )
            exit_code = 2
            continue

        raw_for_save = raw
        if bench_config.output.base_dir != str(output_dir):
            raw_for_save = dict(raw)
            output_payload = dict(raw.get("output") or {})
            output_payload["base_dir"] = str(output_dir)
            raw_for_save["output"] = output_payload
            bench_config = bench_config.model_copy(
                update={
                    "output": bench_config.output.model_copy(
                        update={"base_dir": str(output_dir)}
                    )
                }
            )
        config_hash = bench_config_hash(bench_config.model_dump(by_alias=True))
        run_dir = output_dir / _run_dir_name(bench_config.id, config_hash)
        if nightly.skip_if_exists and run_dir.exists() and not args.force:
            summary["runs"].append(
                {
                    "config": run_spec.config,
                    "status": "skipped",
                    "output": str(run_dir),
                }
            )
            continue

        if args.dry_run:
            summary["runs"].append(
                {
                    "config": run_spec.config,
                    "status": "dry_run",
                    "output": str(run_dir),
                }
            )
            continue

        try:
            result = run_benchmark(bench_config)
            actual_dir = save_benchmark_result(result, raw_for_save, bench_config.output.base_dir)
            run_entry: dict[str, Any] = {
                "config": run_spec.config,
                "status": "ok",
                "bench_id": result.bench_id,
                "output": str(actual_dir),
            }

            regressions: list[str] = []
            if run_spec.baseline:
                baseline = load_curves(Path(run_spec.baseline))
                candidate = load_curves(actual_dir / "curves.json")
                issues = compare_curves(
                    baseline,
                    candidate,
                    metric=run_spec.metric,
                    max_drop=run_spec.max_drop,
                )
                if issues:
                    regressions = format_regressions(issues)
                    run_entry["regressions"] = regressions
                    exit_code = 1
            summary["runs"].append(run_entry)
        except Exception as exc:
            summary["runs"].append(
                {
                    "config": run_spec.config,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            exit_code = 2

    summary_path = output_dir / "nightly_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}")

    for entry in summary["runs"]:
        status = entry.get("status")
        if status == "ok" and entry.get("regressions"):
            print(f"Regressions in {entry.get('config')}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
