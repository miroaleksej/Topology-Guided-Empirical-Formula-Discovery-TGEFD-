from __future__ import annotations

import argparse
import sys
from typing import Iterable

from pydantic import ValidationError
from dataclasses import replace as dc_replace
from pathlib import Path

import numpy as np

from .data import make_demo_data
from .api_v1 import discover, evaluate
from .artifacts import (
    artifact_id_from_hash,
    artifact_path,
    canonical_config_hash,
    verify_artifact,
    write_artifact_bundle,
)
from .storage import build_store
from .config import TGEFDConfig, load_config
from .models import HypothesisParams
from .search import hypothesis_hypercube, rank_results, search_models
from .topology import (
    build_point_cloud,
    persistent_homology,
    persistent_components,
    select_stable_models,
    significant_features,
)

FEATURE_NAMES = ["1", "x", "x^p", "sin(x)", "sin(x)^q", "exp(-x)"]


def _parse_range(value: str, cast=float) -> list[float]:
    value = value.strip()
    if ":" in value:
        parts = value.split(":")
        if len(parts) != 3:
            raise ValueError("Range must be start:stop:step")
        start, stop, step = (cast(p) for p in parts)
        if step == 0:
            raise ValueError("step must be non-zero")
        seq = []
        cur = start
        # include stop with tolerance
        tol = abs(step) * 1e-9
        if step > 0:
            while cur <= stop + tol:
                seq.append(cur)
                cur += step
        else:
            while cur >= stop - tol:
                seq.append(cur)
                cur += step
        return seq
    if "," in value:
        return [cast(v) for v in value.split(",") if v.strip()]
    return [cast(value)]


def _format_coeffs(coeffs: np.ndarray, precision: int = 4) -> str:
    fmt = f"{{:.{precision}g}}"
    return "[" + ", ".join(fmt.format(c) for c in coeffs) + "]"


def _describe_model(params: HypothesisParams, coeffs: np.ndarray) -> str:
    parts = []
    for name, coef in zip(FEATURE_NAMES, coeffs):
        if coef == 0:
            continue
        parts.append(f"({coef:+.4g})*{name}")
    if not parts:
        return "0"
    return " + ".join(parts)


def run(args: argparse.Namespace) -> int:
    if args.demo:
        x, y = make_demo_data(n=args.n, noise=args.noise, seed=args.seed)
    else:
        raise ValueError("Only --demo data is supported in this CLI. Use the library for custom data.")

    p_range = _parse_range(args.p_range, float)
    q_range = _parse_range(args.q_range, float)
    lam_range = _parse_range(args.lam_range, float)

    hypercube = hypothesis_hypercube(p_range, q_range, lam_range)
    results = search_models(x, y, hypercube, iters=args.iters, normalize=args.normalize)

    try:
        X = build_point_cloud(results, scale=args.scale)
        diagrams = persistent_homology(X, maxdim=args.maxdim)
    except ImportError as exc:
        print(str(exc))
        return 3

    h0_sig = significant_features(diagrams[0], args.h0_threshold)
    h1_sig = []
    if len(diagrams) > 1:
        h1_sig = significant_features(diagrams[1], args.h1_threshold)

    components = persistent_components(
        X,
        persistence_threshold=args.h0_threshold,
        min_component_size=args.min_component_size,
    )
    stable = select_stable_models(results, components)

    print(f"total models: {len(results)}")
    print(f"significant H0: {len(h0_sig)} | significant H1: {len(h1_sig)}")
    print(
        "persistent components "
        f"(H0 >= {args.h0_threshold}, min size {args.min_component_size}): {len(components)}"
    )
    print(f"stable models: {len(stable)}")

    ranked = rank_results(stable or results, top_k=args.top_k)

    print("\nTop candidates:")
    for idx, r in enumerate(ranked, 1):
        print(
            f"{idx:02d} p={r.params.p} q={r.params.q} lam={r.params.lam} "
            f"error={r.error:.4g} l0={r.l0} l1={r.l1:.4g}"
        )
        print(f"    coeffs: {_format_coeffs(r.coeffs)}")
        print(f"    model:  {_describe_model(r.params, r.coeffs)}")

    return 0


def _run_validate(path: str) -> int:
    try:
        data = load_config(path)
    except Exception as exc:
        print(f"Config load failed: {exc}")
        return 1

    try:
        cfg = TGEFDConfig.model_validate(data)
    except ValidationError as exc:
        print("Config validation failed\n")
        print(exc.json(indent=2))
        return 1

    print("Config is valid")
    print(f"Mode: {cfg.run.mode}")
    print(f"Seed: {cfg.run.seed}")
    print("Config hash will be computed at runtime")
    return 0


def _artifact_uri_for_store(store_config, artifact_id: str, base_dir: str) -> str | None:
    if store_config.backend == "local":
        return str(Path(base_dir) / "v1" / artifact_id / "artifact")
    if store_config.backend == "s3" and store_config.bucket:
        prefix = store_config.prefix.strip("/")
        return f"s3://{store_config.bucket}/{prefix}/{artifact_id}/artifact"
    if store_config.backend == "gcs" and store_config.bucket:
        prefix = store_config.prefix.strip("/")
        return f"gs://{store_config.bucket}/{prefix}/{artifact_id}/artifact"
    return None


def _run_discover(path: str, base_dir: str | None, reuse: bool, force: bool) -> int:
    try:
        data = load_config(path)
    except Exception as exc:
        print(f"Config load failed: {exc}")
        return 1

    try:
        cfg = TGEFDConfig.model_validate(data)
    except ValidationError as exc:
        print("Config validation failed\n")
        print(exc.json(indent=2))
        return 1

    store_config = cfg.to_store_config()
    if base_dir and store_config.backend == "local":
        store_config = dc_replace(store_config, base_dir=base_dir)
    target_base = base_dir or store_config.base_dir

    if reuse and force:
        print("Invalid flags: --reuse and --force are mutually exclusive.")
        return 2

    config_hash = canonical_config_hash(data)
    artifact_dir = artifact_path(target_base, config_hash)
    artifact_id = artifact_id_from_hash(config_hash)
    artifact_uri = _artifact_uri_for_store(store_config, artifact_id, target_base)

    if artifact_dir.exists():
        if reuse:
            print("Artifact already exists; reusing.")
            print(f"artifact_id: {artifact_id}")
            print(f"path: {artifact_dir}")
            if artifact_uri:
                print(f"artifact_uri: {artifact_uri}")
            print(f"hint: tgefd show {artifact_id}")
            return 0
        if not force:
            print(f"Artifact already exists at {artifact_dir}")
            print(f"hint: tgefd show {artifact_id}")
            print("Use --reuse to reuse or --force to overwrite.")
            return 1

    request = cfg.to_request()
    evaluation = cfg.to_evaluation_policy()
    adaptive = cfg.to_adaptive_config()
    budget = cfg.to_compute_budget()
    if cfg.run.mode == "evaluate":
        response = evaluate(request, evaluation=evaluation, adaptive=adaptive, budget=budget)
    elif cfg.run.mode == "discover":
        response = discover(request, evaluation=evaluation, adaptive=adaptive, budget=budget)
    else:
        print("discover CLI supports only discover/evaluate modes")
        return 2

    store = build_store(store_config)
    artifact_id = artifact_id_from_hash(config_hash)
    artifact_uri = store.uri_for(artifact_id)
    info = write_artifact_bundle(
        data,
        response,
        base_dir=target_base,
        artifact_uri=artifact_uri,
        if_exists="overwrite" if force else "error",
        dataset_provenance=cfg.dataset_provenance(),
    )
    store.store(info.artifact_dir, info.artifact_id)
    print("Artifact created")
    print(f"artifact_id: {info.artifact_id}")
    print(f"path: {info.artifact_dir}")
    if artifact_uri:
        print(f"artifact_uri: {artifact_uri}")
    return 0


def _run_show(artifact_id: str, base_dir: str | None) -> int:
    path = Path(artifact_id)
    if not path.exists():
        if base_dir is None:
            base_dir = "./runs"
        if not artifact_id.startswith("sha256:"):
            artifact_id = artifact_id.replace("-", ":", 1)
        path = artifact_path(base_dir, artifact_id)
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        print(f"manifest.json not found in {path}")
        return 1
    print(manifest_path.read_text(encoding="utf-8"))
    return 0


def _run_verify(artifact_id: str, base_dir: str | None) -> int:
    path = Path(artifact_id)
    if not path.exists():
        if base_dir is None:
            base_dir = "./runs"
        if not artifact_id.startswith("sha256:"):
            artifact_id = artifact_id.replace("-", ":", 1)
        path = artifact_path(base_dir, artifact_id)
    ok = verify_artifact(path)
    if ok:
        print("Artifact integrity OK")
        return 0
    print("Artifact integrity FAILED")
    return 2


def main(argv: Iterable[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) > 0 and argv[0] in {"validate", "discover", "show", "verify"}:
        cmd = argv[0]
        if cmd == "validate":
            parser = argparse.ArgumentParser(description="Validate a TGEFD config")
            parser.add_argument("path", help="Path to config file (.json/.yaml)")
            args = parser.parse_args(argv[1:])
            return _run_validate(args.path)
        if cmd == "discover":
            parser = argparse.ArgumentParser(description="Run discovery and store immutable artifact")
            parser.add_argument("path", help="Path to config file (.json/.yaml)")
            parser.add_argument("--base-dir", default=None, help="Override output base dir")
            parser.add_argument("--reuse", action="store_true", help="Reuse existing artifact if present")
            parser.add_argument("--force", action="store_true", help="Overwrite existing artifact bundle")
            args = parser.parse_args(argv[1:])
            return _run_discover(args.path, args.base_dir, args.reuse, args.force)
        if cmd == "show":
            parser = argparse.ArgumentParser(description="Show manifest for an artifact")
            parser.add_argument("artifact", help="Artifact id or path")
            parser.add_argument("--base-dir", default=None, help="Base dir for artifact lookup")
            args = parser.parse_args(argv[1:])
            return _run_show(args.artifact, args.base_dir)
        if cmd == "verify":
            parser = argparse.ArgumentParser(description="Verify artifact integrity")
            parser.add_argument("artifact", help="Artifact id or path")
            parser.add_argument("--base-dir", default=None, help="Base dir for artifact lookup")
            args = parser.parse_args(argv[1:])
            return _run_verify(args.artifact, args.base_dir)

    parser = argparse.ArgumentParser(description="TGEFD demo CLI")
    parser.add_argument("--demo", action="store_true", help="Run the built-in demo dataset")
    parser.add_argument("--n", type=int, default=200, help="Number of points")
    parser.add_argument("--noise", type=float, default=0.05, help="Noise level")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")

    parser.add_argument("--p-range", default="0.5:3.0:0.5", help="p range, e.g. 0.5:3.0:0.5")
    parser.add_argument("--q-range", default="1:4:1", help="q range, e.g. 1:4:1")
    parser.add_argument("--lam-range", default="0.001,0.005,0.01,0.05", help="lambda list, e.g. 0.001,0.01")

    parser.add_argument("--iters", type=int, default=10, help="STLSQ iterations")
    normalize_group = parser.add_mutually_exclusive_group()
    normalize_group.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        help="Enable feature normalization",
    )
    normalize_group.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable feature normalization",
    )
    parser.set_defaults(normalize=False)

    parser.add_argument("--scale", choices=["none", "standard"], default="none", help="Scale point cloud features")
    parser.add_argument("--maxdim", type=int, default=2, help="Max dimension for persistent homology")
    parser.add_argument("--h0-threshold", type=float, default=0.2, help="Persistence threshold for H0")
    parser.add_argument("--h1-threshold", type=float, default=0.1, help="Persistence threshold for H1")
    parser.add_argument("--min-component-size", type=int, default=5, help="Minimum size for stable component")
    parser.add_argument("--top-k", type=int, default=10, help="Top results to display")

    args = parser.parse_args(argv)

    if not args.demo:
        parser.print_help()
        return 2

    return run(args)


if __name__ == "__main__":
    sys.exit(main())
