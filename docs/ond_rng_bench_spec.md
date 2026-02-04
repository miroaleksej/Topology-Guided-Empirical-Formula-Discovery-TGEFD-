# OND-RNG-Bench Specification (v1.0)

This is a production-grade, reproducible benchmark suite for OND. It provides
standardized datasets and evaluation procedures for comparing RNG dynamics across
protocols (ECDSA + PQ) using OND metrics.

## 1) Scope
- Evaluate **observable randomness dynamics** induced by cryptographic protocols.
- Focus on **structural** properties (OND metrics), not cryptanalytic recovery.
- Provide reproducible curves **quality vs N** and class labels I–IV.

## 2) Core concepts
- **Observation map:** `U_i = pi(sigma_i)` for each signature.
- **Dynamics:** `Delta_i = U_{i+1} - U_i`.
- **Metrics:** `H_rank`, `H_sub`, `H_branch`.
- **Null model:** `U_i` i.i.d. uniform on `O`.

## 3) OND classes
- **Class I (IID-like / OND-maximal)**
- **Class II (Low-rank dynamics)**
- **Class III (Structured full-rank)**
- **Class IV (Finite-state / automaton-like)**

## 4) Required outputs per run
- Curves vs N: **AUC** and **TPR@FPR**.
- OND metric vectors vs N (optional but recommended).
- Class label + confidence (bootstrap).
- Full reproducibility metadata.

## 5) Benchmark config schema (v1.0)
See `schemas/ond_rng_bench_v1.json`.

### DRBG reseed parameters (optional)
For DRBG families (`chacha20_drbg`, `ctr_drbg`, `hmac_drbg`) you can pass:
- `reseed_mode`: `"none"` | `"periodic"` | `"burst"` | `"periodic,burst"` (comma-separated)
- `reseed_interval`: integer (periodic interval in generate calls)
- `reseed_prob`: float in [0,1] (burst probability)
Example:
```yaml
  - name: chacha20_drbg
    params:
      reseed_mode: "periodic"
      reseed_interval: 10
```
See `benchmarks/ond_rng_bench/catalog/reseed_profiles.yaml` for ready reseed profiles.

## 6) Reference layout
```
benchmarks/
  ond_rng_bench/
    catalog/
      rng_families.yaml
      reseed_profiles.yaml
      protocols.yaml
    configs/
      ecdsa_baseline.yaml
      dilithium2_baseline.yaml
    baselines/
      ecdsa_baseline.json
    nightly.yaml
  runs/
    <benchmark_id>/
      manifest.json
      curves.json
      metrics.json
      config.yaml
      logs/
```

## 7) Compliance and safety
- **No key recovery** or cryptanalytic claims.
- Tests are diagnostic and observational.
- Reference vs modified RNG is validated via Golden Experiment.

## 8) Tooling
- Runner: `scripts/run_ond_rng_bench.py`
- Plotter: `scripts/plot_benchmark_curves.py`
- Methodology: `docs/ond_methodology.md`
 - Nightly/regressions: `scripts/run_nightly_bench.py` + `scripts/check_bench_regressions.py`

## 8.1) Auto SVG rendering
Set `output.render_svg=true` and `output.svg_out_dir` in the benchmark config
to auto-generate SVG curves after each run.

## 8.2) DRBG families
The benchmark includes ChaCha20-DRBG, CTR-DRBG, and HMAC-DRBG implementations
for reference class I. These are **benchmark implementations** using standard
primitives (HMAC-SHA256, AES-CTR, ChaCha20) and are not FIPS validated.

## 9) Versioning
- `bench_version` identifies schema stability.
- Outputs include `method_version` and dependency versions.
