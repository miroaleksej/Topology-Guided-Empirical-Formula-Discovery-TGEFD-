import argparse
import hashlib
import json
import os
import secrets
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
import random

import numpy as np

from tgefd.benchmarks import compute_binary_metrics, mean_metric_results


@dataclass
class CurveSeries:
    label: str
    values: list[float]


def _parse_n_values(raw: str) -> list[int]:
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError("n-values must be non-empty")
    return values


def _randbytes(rng: random.Random, n: int) -> bytes:
    if hasattr(rng, "randbytes"):
        return rng.randbytes(n)  # type: ignore[arg-type]
    return bytes(rng.getrandbits(8) for _ in range(n))


class IntRNGFamily:
    def __init__(self, name: str, rng: random.Random, order: int):
        self.name = name
        self.rng = rng
        self.order = order
        self._state = rng.randrange(1, order)

    def next_k(self) -> int:
        if self.name == "reference":
            return self.rng.randrange(1, self.order)
        if self.name == "bounded":
            bound_bits = max(8, self.order.bit_length() // 2)
            bound = min(self.order - 1, 1 << bound_bits)
            return self.rng.randrange(1, bound)
        if self.name == "grid":
            step_bits = max(4, self.order.bit_length() // 6)
            step = 1 << step_bits
            max_steps = max(1, (self.order - 1) // step)
            return step * self.rng.randrange(1, max_steps + 1)
        if self.name == "recurrent":
            a = 1103515245
            c = 12345
            self._state = (a * self._state + c) % self.order
            noise = self.rng.randrange(0, 1 << 12)
            return 1 + (self._state + noise) % (self.order - 1)
        raise ValueError(f"unknown RNG family {self.name}")


class BytesRNGFamily:
    def __init__(self, name: str, rng: random.Random):
        self.name = name
        self.rng = rng
        self._state = rng.randrange(0, 256)

    def randbytes(self, n: int) -> bytes:
        if self.name == "reference":
            return _randbytes(self.rng, n)
        if self.name == "bounded":
            mask_bits = 5
            mask = (1 << mask_bits) - 1
            raw = _randbytes(self.rng, n)
            return bytes((b & mask) for b in raw)
        if self.name == "grid":
            step = 16
            raw = _randbytes(self.rng, n)
            return bytes((b // step) * step for b in raw)
        if self.name == "recurrent":
            out = bytearray()
            a = 57
            c = 101
            for _ in range(n):
                self._state = (a * self._state + c) % 256
                noise = self.rng.randrange(0, 4)
                out.append((self._state + noise) % 256)
            return bytes(out)
        raise ValueError(f"unknown RNG family {self.name}")


@contextmanager
def _patched_system_rng(randbytes_fn: Callable[[int], bytes]):
    orig_urandom = os.urandom
    orig_token = secrets.token_bytes
    os.urandom = lambda n: randbytes_fn(n)  # type: ignore[assignment]
    secrets.token_bytes = lambda n: randbytes_fn(n)  # type: ignore[assignment]
    try:
        yield
    finally:
        os.urandom = orig_urandom
        secrets.token_bytes = orig_token


def _autocorr(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    if np.std(values) == 0:
        return 0.0
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def _collision_rate(values: Iterable[object]) -> float:
    values = list(values)
    if not values:
        return 0.0
    unique = len(set(values))
    return float(1.0 - unique / len(values))


def _ecdsa_features(rs_pairs: list[tuple[int, int]]) -> np.ndarray:
    if not rs_pairs:
        return np.zeros(12, dtype=float)
    r_vals = [r for r, _ in rs_pairs]
    s_vals = [s for _, s in rs_pairs]
    mask = (1 << 32) - 1
    r_mod = np.array([r & mask for r in r_vals], dtype=float)
    s_mod = np.array([s & mask for s in s_vals], dtype=float)
    scale = float(1 << 32)
    r_norm = r_mod / scale
    s_norm = s_mod / scale
    r_bits = np.array([int(r).bit_count() for r in r_vals], dtype=float)
    s_bits = np.array([int(s).bit_count() for s in s_vals], dtype=float)
    features = np.array(
        [
            float(np.mean(r_norm)),
            float(np.std(r_norm)),
            float(np.mean(s_norm)),
            float(np.std(s_norm)),
            float(np.mean(r_bits)),
            float(np.std(r_bits)),
            float(np.mean(s_bits)),
            float(np.std(s_bits)),
            _collision_rate([int(r) for r in r_mod]),
            _collision_rate([int(s) for s in s_mod]),
            _autocorr(r_norm),
            _autocorr(s_norm),
        ],
        dtype=float,
    )
    return features


def _dilithium_features(signatures: list[bytes]) -> np.ndarray:
    if not signatures:
        return np.zeros(8, dtype=float)
    mean_bytes = []
    std_bytes = []
    bit_counts = []
    for sig in signatures:
        arr = np.frombuffer(sig, dtype=np.uint8)
        mean_bytes.append(float(np.mean(arr)))
        std_bytes.append(float(np.std(arr)))
        bit_counts.append(float(int(np.unpackbits(arr).sum())))
    mean_bytes_arr = np.array(mean_bytes, dtype=float)
    std_bytes_arr = np.array(std_bytes, dtype=float)
    bit_counts_arr = np.array(bit_counts, dtype=float)
    features = np.array(
        [
            float(np.mean(mean_bytes_arr)),
            float(np.std(mean_bytes_arr)),
            float(np.mean(std_bytes_arr)),
            float(np.std(std_bytes_arr)),
            float(np.mean(bit_counts_arr)),
            float(np.std(bit_counts_arr)),
            _collision_rate([hashlib.sha256(sig).digest() for sig in signatures]),
            _autocorr(mean_bytes_arr),
        ],
        dtype=float,
    )
    return features


def _train_detector(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("run_golden_experiment requires scikit-learn.") from exc

    idx = np.arange(len(y))
    rng.shuffle(idx)
    split = max(1, int(len(idx) * 0.7))
    train_idx = idx[:split]
    test_idx = idx[split:]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(X[train_idx], y[train_idx])
    scores = model.predict_proba(X[test_idx])[:, 1]
    return y[test_idx], scores


def _ecdsa_signature_set(
    *,
    n_signatures: int,
    rng_family: IntRNGFamily,
    signer,
    order: int,
    seed_prefix: str,
) -> list[tuple[int, int]]:
    from ecdsa.util import sigdecode_string, sigencode_string

    signatures = []
    for i in range(n_signatures):
        msg = f"{seed_prefix}-{i}".encode("utf-8")
        k = rng_family.next_k()
        sig = signer.sign(msg, hashfunc=hashlib.sha256, sigencode=sigencode_string, k=k)
        r, s = sigdecode_string(sig, order)
        signatures.append((int(r), int(s)))
    return signatures


def _dilithium_signature_set(
    *,
    n_signatures: int,
    rng_family: BytesRNGFamily,
    dilithium,
    seed_prefix: str,
) -> list[bytes]:
    signatures: list[bytes] = []
    with _patched_system_rng(rng_family.randbytes):
        keygen = None
        for name in ("keygen", "keypair", "generate_keypair"):
            candidate = getattr(dilithium, name, None)
            if callable(candidate):
                keygen = candidate
                break
        if keygen is None:
            raise RuntimeError("Dilithium2 key generation method not found")
        pk, sk = keygen()
        sign_fn = None
        for name in ("sign", "sign_detached", "sign_message"):
            candidate = getattr(dilithium, name, None)
            if callable(candidate):
                sign_fn = candidate
                break
        if sign_fn is None:
            raise RuntimeError("Dilithium2 sign method not found")
        for i in range(n_signatures):
            msg = f"{seed_prefix}-{i}".encode("utf-8")
            try:
                sig = sign_fn(sk, msg)
            except TypeError:
                sig = sign_fn(msg, sk)
            if isinstance(sig, tuple):
                sig = sig[0]
            signatures.append(sig)
    return signatures


def _run_ecdsa_curve(
    *,
    n_values: list[int],
    trials: int,
    repeats: int,
    fpr_target: float,
    seed: int,
) -> dict:
    try:
        from ecdsa import SECP256k1, SigningKey
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("ECDSA benchmark requires the 'ecdsa' package.") from exc

    curve = SECP256k1
    order = curve.order
    rng_families = ["bounded", "grid", "recurrent"]
    metrics_auc: dict[str, list[float]] = {name: [] for name in rng_families}
    metrics_tpr: dict[str, list[float]] = {name: [] for name in rng_families}

    for n in n_values:
        for family in rng_families:
            family_index = rng_families.index(family)
            runs = []
            for repeat in range(repeats):
                repeat_seed = seed + n * 10_000 + repeat * 101 + family_index * 1000
                rng = random.Random(repeat_seed)
                secret = rng.randrange(1, order)
                signer = SigningKey.from_secret_exponent(secret, curve=curve, hashfunc=hashlib.sha256)

                ref_features = []
                var_features = []
                for trial in range(trials):
                    reference_rng = IntRNGFamily(
                        "reference", random.Random(rng.randrange(1, 2**32)), order
                    )
                    family_rng = IntRNGFamily(
                        family, random.Random(rng.randrange(1, 2**32)), order
                    )
                    rs_ref = _ecdsa_signature_set(
                        n_signatures=n,
                        rng_family=reference_rng,
                        signer=signer,
                        order=order,
                        seed_prefix=f"ref-{repeat}-{trial}",
                    )
                    ref_features.append(_ecdsa_features(rs_ref))

                    rs_var = _ecdsa_signature_set(
                        n_signatures=n,
                        rng_family=family_rng,
                        signer=signer,
                        order=order,
                        seed_prefix=f"{family}-{repeat}-{trial}",
                    )
                    var_features.append(_ecdsa_features(rs_var))

                X = np.vstack([ref_features, var_features])
                y = np.array([0] * len(ref_features) + [1] * len(var_features))
                y_true, scores = _train_detector(X, y, np.random.default_rng(repeat_seed))
                runs.append(compute_binary_metrics(y_true, scores, fpr_target=fpr_target))
            averaged = mean_metric_results(runs)
            metrics_auc[family].append(averaged.auc)
            metrics_tpr[family].append(averaged.tpr_at_fpr)

    return {
        "title": "ECDSA RNG detectability (SECP256k1)",
        "n_values": n_values,
        "metrics": {
            "AUC": [CurveSeries(label=family, values=metrics_auc[family]).__dict__ for family in rng_families],
            f"TPR@FPR={fpr_target:.2f}": [
                CurveSeries(label=family, values=metrics_tpr[family]).__dict__ for family in rng_families
            ],
        },
        "default_metric": "AUC",
        "note": "ECDSA signatures with controlled nonce RNG families",
        "metadata": {
            "curve": "secp256k1",
            "trials": trials,
            "repeats": repeats,
            "seed": seed,
        },
    }


def _resolve_dilithium() -> object:
    try:
        from dilithium_py.dilithium import Dilithium2  # type: ignore
    except Exception:
        Dilithium2 = None  # type: ignore
    if Dilithium2 is not None:
        return Dilithium2
    try:
        import dilithium  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Dilithium2 benchmark requires 'dilithium-py' (preferred) or 'dilithium'."
        ) from exc
    if hasattr(dilithium, "Dilithium2"):
        return dilithium.Dilithium2
    raise ImportError("Dilithium2 class not found in dilithium packages")


def _run_dilithium_curve(
    *,
    n_values: list[int],
    trials: int,
    repeats: int,
    fpr_target: float,
    seed: int,
) -> dict:
    dilithium2 = _resolve_dilithium()
    rng_families = ["bounded", "grid", "recurrent"]
    metrics_auc: dict[str, list[float]] = {name: [] for name in rng_families}
    metrics_tpr: dict[str, list[float]] = {name: [] for name in rng_families}

    for n in n_values:
        for family in rng_families:
            family_index = rng_families.index(family)
            runs = []
            for repeat in range(repeats):
                repeat_seed = seed + n * 10_000 + repeat * 101 + family_index * 1000
                rng = random.Random(repeat_seed)

                ref_features = []
                var_features = []
                for trial in range(trials):
                    reference_rng = BytesRNGFamily(
                        "reference", random.Random(rng.randrange(1, 2**32))
                    )
                    family_rng = BytesRNGFamily(
                        family, random.Random(rng.randrange(1, 2**32))
                    )
                    sig_ref = _dilithium_signature_set(
                        n_signatures=n,
                        rng_family=reference_rng,
                        dilithium=dilithium2,
                        seed_prefix=f"ref-{repeat}-{trial}",
                    )
                    ref_features.append(_dilithium_features(sig_ref))

                    sig_var = _dilithium_signature_set(
                        n_signatures=n,
                        rng_family=family_rng,
                        dilithium=dilithium2,
                        seed_prefix=f"{family}-{repeat}-{trial}",
                    )
                    var_features.append(_dilithium_features(sig_var))

                X = np.vstack([ref_features, var_features])
                y = np.array([0] * len(ref_features) + [1] * len(var_features))
                y_true, scores = _train_detector(X, y, np.random.default_rng(repeat_seed))
                runs.append(compute_binary_metrics(y_true, scores, fpr_target=fpr_target))
            averaged = mean_metric_results(runs)
            metrics_auc[family].append(averaged.auc)
            metrics_tpr[family].append(averaged.tpr_at_fpr)

    return {
        "title": "Dilithium2 RNG detectability",
        "n_values": n_values,
        "metrics": {
            "AUC": [CurveSeries(label=family, values=metrics_auc[family]).__dict__ for family in rng_families],
            f"TPR@FPR={fpr_target:.2f}": [
                CurveSeries(label=family, values=metrics_tpr[family]).__dict__ for family in rng_families
            ],
        },
        "default_metric": "AUC",
        "note": "Dilithium2 signatures with patched RNG source",
        "metadata": {
            "trials": trials,
            "repeats": repeats,
            "seed": seed,
        },
    }


def _write_json(payload: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run golden experiment benchmarks.")
    parser.add_argument(
        "--scheme",
        choices=["ecdsa", "dilithium2", "both"],
        default="both",
        help="Which scheme(s) to run",
    )
    parser.add_argument("--n-values", default="8,16,32,64,128,256")
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fpr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", default="benchmarks/curves")
    args = parser.parse_args()

    n_values = _parse_n_values(args.n_values)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.scheme in ("ecdsa", "both"):
        payload = _run_ecdsa_curve(
            n_values=n_values,
            trials=args.trials,
            repeats=args.repeats,
            fpr_target=args.fpr,
            seed=args.seed,
        )
        _write_json(payload, out_dir / "ecdsa_detection_vs_n.json")

    if args.scheme in ("dilithium2", "both"):
        payload = _run_dilithium_curve(
            n_values=n_values,
            trials=args.trials,
            repeats=args.repeats,
            fpr_target=args.fpr,
            seed=args.seed,
        )
        _write_json(payload, out_dir / "pq_dilithium_detection_vs_n.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
