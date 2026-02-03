# OND Methodology Addendum (Inverse OND + PQ + OND-ART)

This note distills actionable methodology from the internal documents
"обратная задача" and "Новые горизонты" and aligns it with the current
TGEFD/OND codebase. It is intentionally non-cryptanalytic and focuses on
observable dynamics only.

## 1) Inverse OND (clear formulation)
**Problem statement:** Inverse OND is not key recovery and not nonce recovery.
It is **structural inference** of the RNG dynamics class from observable
signature dynamics.

**Observation model:**
- Fix a public context (e.g., the same public key `pk`).
- Map each signature to an observation: `U_i = pi(sigma_i)` in a public
  observation space `O`.
- Define observed dynamics: `Delta_i = U_{i+1} - U_i`.
- Interpret all metrics **relative to a null model** (`U_i` i.i.d. uniform on `O`).

**Diagnostic output (not a verdict):**
- OND returns a **profile** (metrics + class + uncertainty), not a “pass/fail”.
- The output is a **structural report**, not a cryptographic claim.

## 2) OND metrics (canonical orientation)
We use the canonical three-metric vector:
- **H_rank** (rank-entropy): effective dimensionality of dynamics (SVD on the
  dynamics matrix).
- **H_sub** (subspace occupancy): how uniformly dynamics fills its subspace
  (binning entropy after projection to span{Delta}).
- **H_branch** (branching index): how many outgoing transitions per cluster in
  observation space (finite-state/automaton signature).

## 3) Mapping RNG defect models to OND classes
We use A/B/C as **defect models** and map them to OND I–IV via expected metric
profiles.

- **A — Bounded/Interval (structured but full-rank)**
  - Expected: `H_rank ~ max`, but `H_sub` and `H_branch` drop due to phase
    restriction / clustering.
  - **OND Class:** III (Structured full-rank).

- **B — Grid/Comb / Finite-state**
  - Expected: strong drop in `H_branch`, pronounced clustering in `O`.
  - **OND Class:** IV (Finite-state / automaton-like).

- **C — Linear/Recurrent (low-rank dynamics)**
  - Expected: drop in `H_rank` from linear dependence / recurrence.
  - **OND Class:** II (Low-rank dynamics).

Class I remains the null model / IID-like baseline.

## 4) PQ framework (Dilithium2 as pilot)
**Key transfer principle:** signatures are observations of hidden randomness
state; the curve is not essential. For PQ schemes, the geometry changes, not the
principle.

- Define `pi: sigma -> U` for PQ signatures (e.g., lattice/vector space
  representation).
- Use the same `Delta_i = U_{i+1} - U_i` and OND metrics.
- Dilithium2 is a natural pilot: signatures live in structured lattice spaces
  where low-rank and branching effects are meaningful.

## 5) OND-ART (positioning and methodology)
OND-ART is a **complement** to classical statistical tests (e.g., NIST SP 800-22):
- It evaluates **structural, dynamical** properties induced by protocol usage.
- It does **not** replace or modify existing standards.
- It is explicitly non-cryptanalytic and does not claim key/nonce recovery.

## 6) OND-RNG-Bench (benchmark roadmap)
A reproducible benchmark suite that pairs:
- RNG families (reference + controlled modifications),
- usage model (protocol + observation map `pi`),
- and OND class labels (I–IV).

Purpose:
- Validate OND-ART metrics on controlled data.
- Provide stable, comparable curves “quality vs N”.
- Calibrate classifiers and thresholds per scheme (ECDSA + Dilithium2).

## 7) Golden Experiment as the validation core
All claims are validated only via **reference vs modified RNG** under a fixed
public key and fixed protocol implementation. No cryptanalytic attempts or
key recovery are allowed by design.

## 8) Integration into the current repo
- **Docs:** this addendum can be referenced from `docs/ecdsa_golden_experiment.md`.
- **Benchmarks:** `scripts/run_golden_experiment.py` already supports ECDSA +
  Dilithium2 and outputs JSON for curves.
- **Value story:** `docs/benchmarks_value_story.md` can cite OND-ART as the
  methodological frame and OND-RNG-Bench as the roadmap.
