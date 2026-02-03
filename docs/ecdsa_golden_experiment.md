# ECDSA Golden Experiment (RNG Detectability)

Goal: create a reproducible, synthetic benchmark that measures how well
non-reference RNG families can be detected from ECDSA signatures as the number
of signatures (N) increases.

Safety note: this is a research benchmark for detecting weak randomness. It is
not intended for attacking real systems. Use only synthetic keys and datasets.

## Baseline Setup
- Curve: choose one (e.g., secp256k1 or P-256) and keep it fixed.
- Hash: SHA-256 for message hashes.
- Key: one fixed private key per trial to avoid key mixing.
- Reference RNG: system CSPRNG or standards-compliant DRBG.

## RNG Families (Modified)
Each family is a controllable deviation from the reference RNG. Use multiple
parameter settings per family to span weak to strong deviations.

1) Bounded / Interval RNG
   - Draw k from a limited interval or with fixed high/low bits.
   - Example controls: interval width, fixed prefix length, mask size.

2) Grid / Comb / Finite-State RNG
   - Draw k from a discrete grid or finite set.
   - Example controls: grid step size, set size, state machine period.

3) Recurrent / Low-Rank RNG
   - Generate k via linear recurrence or low-rank transform with small noise.
   - Example controls: recurrence order, noise scale, rank.

## Dataset Generation
For each RNG family + parameter setting:
- For each trial t:
  - Generate N signatures with fixed key and RNG variant.
  - Compute a feature vector from the signature set.
- Label: RNG class or "reference vs non-reference" (binary).

Suggested feature families (non-sensitive):
- Distribution stats: mean/variance/skew of r and s values.
- Collision metrics: repeated r rate, nearest-neighbor distances.
- Temporal metrics: lag-1 autocorrelation of r and s.
- Bit-level stats: Hamming weight distribution of r and s.

## Detection Quality vs N
For N in a range (e.g., 8, 16, 32, 64, 128, 256):
- Train a detector on synthetic trials (holdout for validation).
- Record quality metrics: accuracy, AUC, or TPR at fixed FPR.
- Plot quality vs N for each RNG family.

## Automated Benchmark Runner
Run the real signature generators and auto-update JSON curves:
```bash
python -m pip install -e ".[benchmarks]"
python scripts/run_golden_experiment.py --scheme both --n-values 8,16,32,64,128,256
python scripts/plot_benchmark_curves.py benchmarks/curves/ecdsa_detection_vs_n.json \
  benchmarks/curves/pq_dilithium_detection_vs_n.json
```

Notes:
- ECDSA uses controlled nonces (k) via the `ecdsa` library.
- Dilithium2 uses `dilithium-py` (or `dilithium`) and patches the system RNG for signing.
- Metrics include AUC and TPR@FPR (stored in JSON).

## Methodology Reference (Inverse OND + PQ + OND-ART)
See `docs/ond_methodology.md` for the canonical framing of Inverse OND, the
A/B/C → OND I–IV mapping, and the PQ transfer principle (Dilithium2 pilot).

## How to Use TGEFD (Optional)
If you want formula discovery instead of a standard classifier:
- Define y as a detection target (e.g., reference vs non-reference).
- Use TGEFD to search for parsimonious formulae over the feature set.
- Measure detection quality on holdout trials for each N.

## Reproducibility
- Fix seeds for key generation, message generation, and RNG variants.
- Store config + artifacts per N.
- Keep reference RNG runs as a control curve.
