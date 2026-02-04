# TGEFD Theory (Draft v1)

This document provides a formal overview of the Topology-Guided Empirical Formula Discovery (TGEFD) method.
It is intended to be self-contained and publication-ready.

## 1. Problem Setting

Given observations $(x, y)$ and a hypothesis library $\Phi(x; \theta)$ parameterized by $\theta$,
we seek a sparse model $y \approx \Phi(x; \theta)\,c$ that is *topologically stable* in hypothesis space.

Each hypothesis produces a point in the *hypothesis landscape*:
$$
z_i = (\theta_i, \varepsilon_i, \|c_i\|_0),
$$
where $\varepsilon_i$ is the approximation error.

## 2. Topological Stability Criterion

We compute persistent homology of the point cloud $\{z_i\}$ (Vietoris–Rips complex).
Short-lived features are treated as noise; long-lived features indicate stable structure.

### Acceptance Criterion (Formal)

A model family is **accepted** if all conditions hold:

1. There exists at least one stable $H_0$ component across all noise levels.
2. If required, at least one stable $H_1$ feature is present.
3. The persistent image energy is above a minimum threshold.
4. The coefficient pattern is consistent across noise (intersection non-empty).

Otherwise the result is **rejected** with a recorded reason.

### Rejection Reason Codes (Formal)

The decision policy records one explicit reason code:

- `no_stable_components`
- `no_h1`
- `pi_energy_unavailable`
- `pi_energy_below_threshold`
- `topological_instability`

These are policy outcomes and are treated as valid run results.

## 3. Algorithm 1 (Coarse→Fine + Topological Filtering)

```
Input: data (x, y), hypercube H, noise levels N
Output: accepted/rejected + stable model families

1. Run coarse search on H → results R
2. Compute PH on hypothesis landscape; detect stable H0 components
3. If adaptive enabled:
   3.1 pick top-k stable models
   3.2 refine hypercube locally → H'
   3.3 run search on H'
4. Aggregate stability across noise levels:
   - intersect stable coefficient patterns
   - compute PI energy statistics
5. Decide accept/reject using formal policy
```

## 4. Negative Results Are First-Class

If no stable family is found, the system **rejects** and certifies
that no stable empirical structure exists in the given hypothesis class.

## 5. Notes on Reproducibility

All decisions are deterministic with fixed seed and config.
Each run is stored as an immutable artifact (config + results + hashes).
Each run also records `method_version` to pin the decision procedure version.
