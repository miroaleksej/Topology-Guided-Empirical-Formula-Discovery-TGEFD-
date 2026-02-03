# OND-RNG-Bench Roadmap

This document defines a concrete, reproducible benchmark suite for OND.
It pairs RNG families with protocols and assigns OND classes I–IV for
calibration and regression testing.

## 1) Goals
- Provide stable, labeled RNG datasets for OND-ART and OND metrics.
- Enable apples-to-apples comparisons across schemes (ECDSA + PQ).
- Calibrate thresholds for `H_rank`, `H_sub`, `H_branch` vs N.

## 2) Canonical OND Classes (I–IV)
- **Class I (IID-like / OND-maximal)**: reference RNG behavior.
- **Class II (Low-rank dynamics)**: linear/recurrent structure.
- **Class III (Structured full-rank)**: bounded/interval, phase restriction.
- **Class IV (Finite-state / automaton-like)**: grid/comb, low branching.

## 3) Benchmark Matrix (minimal baseline)

### A) Synthetic RNG families (must-have)
| RNG family | OND class | Notes |
| --- | --- | --- |
| Reference CSPRNG (OS) | I | baseline reference per protocol |
| Bounded/Interval RNG | III | phase restriction, full rank | 
| Grid/Comb RNG | IV | finite-state / low branching | 
| Linear/Recurrence RNG | II | low-rank dynamics |

### A.1) Concrete DRBGs (reference class I)
| RNG family | OND class | Notes |
| --- | --- | --- |
| ChaCha20-DRBG | I | stream-cipher DRBG baseline |
| CTR-DRBG (AES) | I | counter-mode DRBG baseline |
| HMAC-DRBG | I | hash-based DRBG baseline |

### B) Protocols (pilot set)
| Protocol | Observation map `pi` | Notes |
| --- | --- | --- |
| ECDSA secp256k1 | `(r, s)` or normalized projection | baseline classical |
| Dilithium2 | lattice/vector projection from signature | PQ pilot |

### C) Usage models
| Usage model | Description |
| --- | --- |
| Fixed key, varying RNG | standard golden experiment |
| Fixed RNG, varying message | controls message entropy |
| Controlled noise injection | sensitivity analysis |

### C.1) Reseed / entropy-pool variants (structured)
| Variant | Expected OND class | Notes |
| --- | --- | --- |
| Periodic reseed (short period) | IV | cycle/finite-state artifacts |
| Low-entropy pool (reduced mixing) | III | phase restriction |
| Biased pool (MSB/LSB skew) | III | structured but full-rank |
| Stalled reseed (frozen pool) | II/IV | low-rank or automaton-like |
| Burst reseed (on demand) | IV | regime switching |

#### Expected metric profiles (H_rank / H_sub / H_branch)
- **Periodic reseed (short period):** H_rank ↘, H_sub ↘, H_branch ↘↘ (cyclic/finite-state transitions).
- **Low-entropy pool:** H_rank ~ max, H_sub ↘↘, H_branch ↘ (phase restriction with clustering).
- **Biased pool (MSB/LSB skew):** H_rank ~ max, H_sub ↘, H_branch ↘ (structured but full-rank).
- **Stalled reseed (frozen pool):** H_rank ↘↘, H_sub ↘, H_branch ↘↘ (low-rank + automaton-like).
- **Burst reseed (on demand):** H_rank ~ max, H_sub ↘, H_branch ↘↘ (regime switching, abrupt transitions).

## 4) Expanded catalog (next step)
- **CSPRNG variants:** ChaCha20-based, HMAC-DRBG, CTR-DRBG (reference class I).
- **Seed reuse / reseeding models:** periodic reseed → Class IV / hybrid.
- **Hybrid RNGs:** concatenated RNGs (class IV / mixed dynamics).
- **Hardware RNG drift:** slow bias over time (Class III / nonstationary).
 - **Reseed profiles:** see `benchmarks/ond_rng_bench/catalog/reseed_profiles.yaml`.

## 5) Required outputs per run
- `N` sweep curves (AUC + TPR@FPR)
- OND metric vectors vs N
- Class label + confidence (bootstrapped)
- Reproducibility metadata (seed, protocol, RNG family, version)

## 6) Acceptance criteria (for internal readiness)
- Curves are monotone or near-monotone in N (statistical noise allowed).
- Class profiles are consistent across runs.
- Reference RNG (Class I) stays close to null model.

## 7) Integration points in repo
- Benchmark runner: `scripts/run_golden_experiment.py`
- Curves: `benchmarks/curves/*.json`
- Plots: `docs/benchmarks/*.svg`
- Methodology: `docs/ond_methodology.md`

## 8) Status
- Baseline ECDSA + Dilithium2 pilot implemented.
- Next milestone: populate expanded catalog and freeze baseline curves (nightly regression gate).
