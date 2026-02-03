# Method and Trust

This document defines the minimum trust guarantees for TGEFD runs.

## 1) Acceptance/rejection logic is formalized

Decision logic is implemented as explicit policy checks in `tgefd/evaluation.py`:

- stable component threshold
- optional H1 requirement
- persistence-image energy availability and threshold
- topological stability tolerance

The output is a deterministic tuple: `(accepted: bool, reason: str)`.

## 2) Rejection is a valid result

Rejected runs return:

- `status: "rejected"`
- `decision.accepted: false`
- `decision.reason: <policy reason code>`

This is a method outcome, not an exception.

## 3) Negative results are first-class artifacts

Artifact bundles always include:

- `results.json`
- `decision.json`

Rejected runs additionally include:

- `negative_result.json`

Negative outcomes are therefore persisted, hashed, and auditable.

## 4) Method version is fixed per run

Every run records `reproducibility.method_version`, and the same value is written to
`manifest.json` under `run.method_version`.

This makes method evolution explicit and traceable across historical runs.

## 5) Computational guardrails are enforced

`compute_budget` enforces hard limits before and during execution:

- maximum hypothesis hypercube size
- maximum noise trials and total runs
- maximum model evaluations
- timeout guardrails for symbolic regression and persistent homology

When a budget is violated, the run fails fast with an explicit explanation:
`budget exceeded: ...`
