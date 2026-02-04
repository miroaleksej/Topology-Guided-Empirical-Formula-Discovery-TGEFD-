import argparse
from pathlib import Path
import subprocess
import sys

from tgefd.ond_rng_bench import BenchConfig, load_bench_config, run_benchmark, save_benchmark_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OND-RNG-Bench from config.")
    parser.add_argument("config", help="Path to bench config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, do not run")
    args = parser.parse_args()

    raw = load_bench_config(args.config)
    try:
        config = BenchConfig.model_validate(raw)
    except Exception as exc:
        print(f"Config validation failed: {exc}")
        return 2

    if args.dry_run:
        print("Config is valid")
        print(f"protocol: {config.protocol.name}")
        print(f"rng_families: {[spec.name for spec in config.rng_families]}")
        print(f"n_values: {config.n_values}")
        return 0

    result = run_benchmark(config)
    run_dir = save_benchmark_result(result, raw, config.output.base_dir)
    if config.output.render_svg:
        out_dir = config.output.svg_out_dir or "docs/benchmarks"
        cmd = [
            sys.executable,
            "scripts/plot_benchmark_curves.py",
            str(run_dir / "curves.json"),
            "--out-dir",
            out_dir,
        ]
        if config.output.svg_metric:
            cmd.extend(["--metric", config.output.svg_metric])
        subprocess.run(cmd, check=True)
    print(f"bench_id: {result.bench_id}")
    print(f"output: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
