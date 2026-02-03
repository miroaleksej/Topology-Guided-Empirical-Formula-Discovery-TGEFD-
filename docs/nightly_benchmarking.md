# Nightly OND-RNG-Bench + Regression Checks

This guide defines an automated nightly benchmark run and a regression gate.

## What it does
- Executes a fixed set of OND-RNG-Bench configs.
- Writes a summary file (`nightly_summary.json`).
- Compares curves against frozen baselines and fails on regressions.

## Nightly config
See `benchmarks/ond_rng_bench/nightly.yaml` for the canonical schedule.

Fields:
- `output_dir`: where run artifacts go.
- `skip_if_exists`: avoid rerunning if the exact run already exists.
- `runs`: list of benchmark configs plus baselines and thresholds.

Example:
```yaml
version: "1.0"
output_dir: benchmarks/nightly
skip_if_exists: true
runs:
  - config: benchmarks/ond_rng_bench/configs/ecdsa_baseline.yaml
    baseline: benchmarks/ond_rng_bench/baselines/ecdsa_baseline.json
    metric: AUC
    max_drop: 0.05
```

## How to run
```bash
python scripts/run_nightly_bench.py benchmarks/ond_rng_bench/nightly.yaml
```

To re-run even if output exists:
```bash
python scripts/run_nightly_bench.py benchmarks/ond_rng_bench/nightly.yaml --force
```

## Regression checker (standalone)
```bash
python scripts/check_bench_regressions.py \
  benchmarks/ond_rng_bench/baselines/ecdsa_baseline.json \
  benchmarks/nightly/<bench_id>/curves.json \
  --metric AUC \
  --max-drop 0.05
```

## Baseline refresh
1) Run a benchmark config manually.
2) Copy the `curves.json` into `benchmarks/ond_rng_bench/baselines/`.
3) Update `benchmarks/ond_rng_bench/nightly.yaml` to reference the new baseline file.

## CI/cron suggestion
- Run nightly on a dedicated runner (CPU-heavy).
- Persist `benchmarks/nightly/` and `benchmarks/ond_rng_bench/baselines/` as artifacts.
- Fail the job if regressions are reported.
