# Agent Instructions

Project: Topology-Guided Empirical Formula Discovery (TGEFD)

Goals
- Provide a runnable research framework for hypothesis hypercube search, sparse regression, and topology-guided filtering.
- Keep the core lightweight and easy to extend.

Conventions
- Python 3.11+.
- Prefer numpy + scikit-learn only. Add new dependencies only if essential and document why.
- Use dataclasses for structured results.
- Keep files ASCII unless the file already uses non-ASCII.
- Add small, focused tests for new behavior.

Useful commands
- Run tests: `python -m pytest`
- Run example: `python -m tgefd.cli --demo`

Repo layout
- `tgefd/`: core library.
- `examples/`: runnable examples.
- `tests/`: pytest tests.
