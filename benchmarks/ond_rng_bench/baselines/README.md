# OND-RNG-Bench Baselines

Store frozen `curves.json` snapshots here for regression checks.

Suggested workflow:
1) Run a benchmark config with `scripts/run_ond_rng_bench.py`.
2) Copy the resulting `curves.json` into this directory with a stable name.
3) Update `benchmarks/ond_rng_bench/nightly.yaml` to reference the file.

Example:
```bash
python scripts/run_ond_rng_bench.py benchmarks/ond_rng_bench/configs/ecdsa_baseline.yaml
cp benchmarks/runs/<bench_id>/curves.json benchmarks/ond_rng_bench/baselines/ecdsa_baseline.json
```
