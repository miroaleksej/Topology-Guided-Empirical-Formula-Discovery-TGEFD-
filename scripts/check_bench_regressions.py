import argparse
from pathlib import Path
import sys

from tgefd.bench_regressions import load_curves, compare_curves, format_regressions


def _resolve_curves_path(path: str) -> Path:
    target = Path(path)
    if target.is_dir():
        target = target / "curves.json"
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Check OND-RNG-Bench regressions against baseline curves.")
    parser.add_argument("baseline", help="Path to baseline curves.json or run directory")
    parser.add_argument("candidate", help="Path to candidate curves.json or run directory")
    parser.add_argument("--metric", help="Metric name to compare (defaults to curves default)")
    parser.add_argument("--max-drop", type=float, default=0.0, help="Max allowed absolute drop")
    args = parser.parse_args()

    baseline_path = _resolve_curves_path(args.baseline)
    candidate_path = _resolve_curves_path(args.candidate)

    if not baseline_path.exists():
        print(f"Baseline curves not found: {baseline_path}")
        return 2
    if not candidate_path.exists():
        print(f"Candidate curves not found: {candidate_path}")
        return 2

    baseline = load_curves(baseline_path)
    candidate = load_curves(candidate_path)
    regressions = compare_curves(baseline, candidate, metric=args.metric, max_drop=args.max_drop)
    if not regressions:
        print("No regressions detected")
        return 0

    print("Regressions detected:")
    for line in format_regressions(regressions):
        print(f"- {line}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
