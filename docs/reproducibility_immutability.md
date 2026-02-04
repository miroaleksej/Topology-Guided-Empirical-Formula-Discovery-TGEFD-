# Reproducibility and Immutability

This document defines reproducibility guarantees for every TGEFD run.

## Guarantees

1. **Config hash is fixed in each run**
   - Response includes `reproducibility.config_hash`.
   - Artifact stores `config.hash` and `manifest.config.hash`.

2. **Config + Results + Evidence are immutable artifacts**
   - Artifact always contains:
     - `config.yaml`
     - `results.json`
     - `evidence.json`
   - Artifact directory is locked after write (`_LOCK`) and can be integrity-verified.
   - Artifacts are written to a staging directory and atomically published to avoid race-condition partial writes.

3. **Library versions are stored**
   - If `reproducibility.library_versions: true`, artifact includes `libraries.json`.
   - File contains Python version and package versions used for execution context.

4. **Seed and deterministic mode are standardized**
   - `run.seed` is required and non-negative.
   - `run.deterministic` is standardized to `true`.
   - Response includes `reproducibility.seed` and `reproducibility.deterministic`.

5. **Artifact integrity hash is verifiable**
   - Manifest stores `integrity.artifact_hash`.
   - Verify with CLI: `tgefd verify sha256-<hash>`.

6. **Partial failures are handled safely**
   - Local artifact generation cleans up staging directories on failure.
   - S3/GCS upload failures use retries/backoff and attempt rollback of already uploaded objects.
