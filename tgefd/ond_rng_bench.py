from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .benchmarks import compute_binary_metrics


@dataclass(frozen=True)
class CurveSeries:
    label: str
    values: list[float]


@dataclass(frozen=True)
class BenchmarkResult:
    bench_id: str
    config_hash: str
    curves: dict[str, list[CurveSeries]]
    ond_metrics: dict[str, Any]
    metadata: dict[str, Any]


class ProtocolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] | None = None


class ObservationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    map: str | None = None
    note: str | None = None


class RNGFamilySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    label: str | None = None
    class_label: str | None = None
    params: dict[str, Any] | None = None


class UsageSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixed_key: bool = True
    message_model: str = "counter"


class MetricsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fpr_target: float = Field(0.05, ge=0.0, le=1.0)


class OutputSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_dir: str
    save_curves: bool = True
    save_datasets: bool = False
    render_svg: bool = False
    svg_out_dir: str | None = None
    svg_metric: str | None = None


class ReproSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int
    deterministic: bool = True


class BenchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bench_version: str
    id: str
    protocol: ProtocolSpec
    observation: ObservationSpec | None = None
    n_values: list[int]
    rng_families: list[RNGFamilySpec]
    usage: UsageSpec = Field(default_factory=UsageSpec)
    trials: int = Field(ge=1)
    repeats: int = Field(ge=1)
    metrics: MetricsSpec = Field(default_factory=MetricsSpec)
    output: OutputSpec
    reproducibility: ReproSpec

    @model_validator(mode="after")
    def _check_values(self) -> "BenchConfig":
        if self.bench_version != "1.0":
            raise ValueError("bench_version must be '1.0'")
        if not self.n_values:
            raise ValueError("n_values must be non-empty")
        if any(n <= 0 for n in self.n_values):
            raise ValueError("n_values must be positive")
        if not self.rng_families:
            raise ValueError("rng_families must be non-empty")
        return self


def load_bench_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def canonical_bench_yaml(config: Mapping[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=True)


def bench_config_hash(config: Mapping[str, Any]) -> str:
    payload = canonical_bench_yaml(config).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _bench_run_id(config: BenchConfig, config_hash: str) -> str:
    short = config_hash.split(":", 1)[-1][:12]
    return f"{config.id}-{short}"


def _collect_library_versions() -> dict[str, Any]:
    import importlib.metadata
    versions = {
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "packages": {},
    }
    for name in [
        "numpy",
        "scikit-learn",
        "ecdsa",
        "dilithium-py",
        "dilithium",
        "cryptography",
    ]:
        try:
            versions["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _scale(values: np.ndarray, bins: int = 10) -> np.ndarray:
    if values.size == 0:
        return values
    vmin = values.min(axis=0)
    vmax = values.max(axis=0)
    span = np.where(vmax - vmin == 0, 1.0, vmax - vmin)
    scaled = (values - vmin) / span
    return np.clip(scaled, 0.0, 0.999999) * bins


def compute_ond_metrics(u_series: np.ndarray) -> dict[str, float]:
    if u_series.shape[0] < 2:
        return {"H_rank": 0.0, "H_sub": 0.0, "H_branch": 0.0}
    delta = np.diff(u_series, axis=0)
    if delta.ndim == 1:
        delta = delta.reshape(-1, 1)

    # H_rank
    try:
        _, svals, _ = np.linalg.svd(delta, full_matrices=False)
    except np.linalg.LinAlgError:
        svals = np.zeros(min(delta.shape), dtype=float)
    ssum = float(np.sum(svals))
    if ssum == 0.0:
        h_rank = 0.0
    else:
        p = svals / ssum
        h_rank = float(-np.sum(p * np.log(p + 1e-12)) / np.log(len(p) + 1e-12))

    # H_sub
    k = min(3, delta.shape[1])
    if k == 0:
        h_sub = 0.0
    else:
        try:
            _, _, vt = np.linalg.svd(delta, full_matrices=False)
            proj = delta @ vt[:k].T
        except np.linalg.LinAlgError:
            proj = delta[:, :k]
        bins = 10
        scaled = _scale(proj, bins=bins).astype(int)
        _, counts = np.unique(scaled, return_counts=True, axis=0)
        probs = counts / counts.sum()
        h_sub = float(-np.sum(probs * np.log(probs + 1e-12)) / np.log(len(probs) + 1e-12))

    # H_branch
    try:
        from sklearn.cluster import KMeans
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("OND metrics require scikit-learn.") from exc
    n_clusters = min(8, max(2, u_series.shape[0] // 4))
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
    labels = km.fit_predict(u_series)
    transitions = list(zip(labels[:-1], labels[1:]))
    if not transitions:
        h_branch = 0.0
    else:
        counts = {}
        outgoing = {k: [] for k in range(n_clusters)}
        for a, b in transitions:
            counts[(a, b)] = counts.get((a, b), 0) + 1
        for (a, b), c in counts.items():
            outgoing[a].append(c)
        entropies = []
        weights = []
        for k, vals in outgoing.items():
            total = sum(vals)
            if total == 0:
                continue
            probs = np.array(vals, dtype=float) / float(total)
            ent = float(-np.sum(probs * np.log(probs + 1e-12)) / np.log(len(probs) + 1e-12))
            entropies.append(ent)
            weights.append(total)
        if not entropies:
            h_branch = 0.0
        else:
            weights_arr = np.array(weights, dtype=float)
            h_branch = float(np.average(entropies, weights=weights_arr))
    return {
        "H_rank": float(np.clip(h_rank, 0.0, 1.0)),
        "H_sub": float(np.clip(h_sub, 0.0, 1.0)),
        "H_branch": float(np.clip(h_branch, 0.0, 1.0)),
    }


def _randbytes(rng: random.Random, n: int) -> bytes:
    if hasattr(rng, "randbytes"):
        return rng.randbytes(n)  # type: ignore[arg-type]
    return bytes(rng.getrandbits(8) for _ in range(n))


class IntRNGFamily:
    def __init__(self, name: str, rng: random.Random, order: int, params: Mapping[str, Any] | None = None):
        self.name = name
        self.rng = rng
        self.order = order
        self._state = rng.randrange(1, order)
        self._params = params or {}

    def next_k(self) -> int:
        if self.name == "reference":
            return self.rng.randrange(1, self.order)
        if self.name == "bounded":
            bound_bits = int(self._params.get("bound_bits", max(8, self.order.bit_length() // 2)))
            bound = min(self.order - 1, 1 << bound_bits)
            return self.rng.randrange(1, bound)
        if self.name == "grid":
            step_bits = int(self._params.get("step_bits", max(4, self.order.bit_length() // 6)))
            step = 1 << step_bits
            max_steps = max(1, (self.order - 1) // step)
            return step * self.rng.randrange(1, max_steps + 1)
        if self.name == "recurrent":
            a = 1103515245
            c = 12345
            self._state = (a * self._state + c) % self.order
            noise_bits = int(self._params.get("noise_bits", 12))
            noise = self.rng.randrange(0, 1 << max(1, noise_bits))
            return 1 + (self._state + noise) % (self.order - 1)
        raise ValueError(f"unknown RNG family {self.name}")


class BytesRNGFamily:
    def __init__(self, name: str, rng: random.Random, params: Mapping[str, Any] | None = None):
        self.name = name
        self.rng = rng
        self._state = rng.randrange(0, 256)
        self._params = params or {}

    def randbytes(self, n: int) -> bytes:
        if self.name == "reference":
            return _randbytes(self.rng, n)
        if self.name == "bounded":
            mask_bits = int(self._params.get("mask_bits", 5))
            mask = (1 << mask_bits) - 1
            raw = _randbytes(self.rng, n)
            return bytes((b & mask) for b in raw)
        if self.name == "grid":
            step = int(self._params.get("step", 16))
            step = max(1, step)
            raw = _randbytes(self.rng, n)
            return bytes((b // step) * step for b in raw)
        if self.name == "recurrent":
            out = bytearray()
            a = 57
            c = 101
            for _ in range(n):
                self._state = (a * self._state + c) % 256
                noise_bits = int(self._params.get("noise_bits", 2))
                noise = self.rng.randrange(0, 1 << max(1, noise_bits))
                out.append((self._state + noise) % 256)
            return bytes(out)
        raise ValueError(f"unknown RNG family {self.name}")


class HmacDRBG:
    def __init__(
        self,
        entropy: bytes,
        nonce: bytes,
        personalization: bytes = b"",
        *,
        reseed_mode: str = "none",
        reseed_interval: int = 0,
        reseed_prob: float = 0.0,
        reseed_rng: random.Random | None = None,
    ) -> None:
        self._K = b"\x00" * 32
        self._V = b"\x01" * 32
        seed = entropy + nonce + personalization
        self._update(seed)
        self._reseed_mode = reseed_mode
        self._reseed_interval = max(0, int(reseed_interval))
        self._reseed_prob = float(reseed_prob)
        self._reseed_rng = reseed_rng or random.Random()
        self._generate_calls = 0

    def _hmac(self, key: bytes, data: bytes) -> bytes:
        return hmac.new(key, data, hashlib.sha256).digest()

    def _update(self, provided_data: bytes = b"") -> None:
        self._K = self._hmac(self._K, self._V + b"\x00" + provided_data)
        self._V = self._hmac(self._K, self._V)
        if provided_data:
            self._K = self._hmac(self._K, self._V + b"\x01" + provided_data)
            self._V = self._hmac(self._K, self._V)

    def reseed(self, entropy: bytes, additional: bytes = b"") -> None:
        self._update(entropy + additional)

    def _maybe_reseed(self) -> None:
        if "periodic" in self._reseed_mode and self._reseed_interval > 0:
            if self._generate_calls > 0 and self._generate_calls % self._reseed_interval == 0:
                entropy = _randbytes(self._reseed_rng, 32)
                nonce = _randbytes(self._reseed_rng, 16)
                self.reseed(entropy + nonce)
        if "burst" in self._reseed_mode and self._reseed_prob > 0.0:
            if self._reseed_rng.random() < self._reseed_prob:
                entropy = _randbytes(self._reseed_rng, 32)
                nonce = _randbytes(self._reseed_rng, 16)
                self.reseed(entropy + nonce)

    def randbytes(self, n: int, additional: bytes = b"") -> bytes:
        self._maybe_reseed()
        if additional:
            self._update(additional)
        out = bytearray()
        while len(out) < n:
            self._V = self._hmac(self._K, self._V)
            out.extend(self._V)
        self._update(additional)
        self._generate_calls += 1
        return bytes(out[:n])


class CtrDRBG:
    def __init__(
        self,
        entropy: bytes,
        nonce: bytes,
        personalization: bytes = b"",
        *,
        reseed_mode: str = "none",
        reseed_interval: int = 0,
        reseed_prob: float = 0.0,
        reseed_rng: random.Random | None = None,
    ) -> None:
        self._key = b"\x00" * 32
        self._V = b"\x00" * 16
        seed = hashlib.sha512(entropy + nonce + personalization).digest()[:48]
        self._update(seed)
        self._reseed_mode = reseed_mode
        self._reseed_interval = max(0, int(reseed_interval))
        self._reseed_prob = float(reseed_prob)
        self._reseed_rng = reseed_rng or random.Random()
        self._generate_calls = 0

    @staticmethod
    def _inc(v: bytes) -> bytes:
        value = int.from_bytes(v, "big") + 1
        return (value % (1 << 128)).to_bytes(16, "big")

    @staticmethod
    def _xor_bytes(a: bytes, b: bytes) -> bytes:
        return bytes(x ^ y for x, y in zip(a, b))

    def _encrypt_block(self, block: bytes) -> bytes:
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("CTR-DRBG requires cryptography.") from exc
        cipher = Cipher(algorithms.AES(self._key), modes.ECB())
        encryptor = cipher.encryptor()
        return encryptor.update(block) + encryptor.finalize()

    def _update(self, provided_data: bytes = b"") -> None:
        seedlen = 48
        temp = bytearray()
        v = self._V
        while len(temp) < seedlen:
            v = self._inc(v)
            temp.extend(self._encrypt_block(v))
        temp_bytes = bytes(temp[:seedlen])
        if provided_data:
            provided_data = provided_data.ljust(seedlen, b"\x00")[:seedlen]
            temp_bytes = self._xor_bytes(temp_bytes, provided_data)
        self._key = temp_bytes[:32]
        self._V = temp_bytes[32:48]

    def reseed(self, entropy: bytes, additional: bytes = b"") -> None:
        seed = hashlib.sha512(entropy + additional).digest()[:48]
        self._update(seed)

    def _maybe_reseed(self) -> None:
        if "periodic" in self._reseed_mode and self._reseed_interval > 0:
            if self._generate_calls > 0 and self._generate_calls % self._reseed_interval == 0:
                entropy = _randbytes(self._reseed_rng, 32)
                nonce = _randbytes(self._reseed_rng, 16)
                self.reseed(entropy + nonce)
        if "burst" in self._reseed_mode and self._reseed_prob > 0.0:
            if self._reseed_rng.random() < self._reseed_prob:
                entropy = _randbytes(self._reseed_rng, 32)
                nonce = _randbytes(self._reseed_rng, 16)
                self.reseed(entropy + nonce)

    def randbytes(self, n: int, additional: bytes = b"") -> bytes:
        self._maybe_reseed()
        if additional:
            seed = hashlib.sha512(additional).digest()[:48]
            self._update(seed)
        out = bytearray()
        v = self._V
        while len(out) < n:
            v = self._inc(v)
            out.extend(self._encrypt_block(v))
        self._V = v
        self._update(additional if additional else b"")
        self._generate_calls += 1
        return bytes(out[:n])


class ChaCha20DRBG:
    def __init__(
        self,
        entropy: bytes,
        nonce: bytes,
        personalization: bytes = b"",
        *,
        reseed_mode: str = "none",
        reseed_interval: int = 0,
        reseed_prob: float = 0.0,
        reseed_rng: random.Random | None = None,
    ) -> None:
        seed = hashlib.sha512(entropy + nonce + personalization).digest()
        self._key = seed[:32]
        self._nonce = seed[32:48]
        self._reseed_mode = reseed_mode
        self._reseed_interval = max(0, int(reseed_interval))
        self._reseed_prob = float(reseed_prob)
        self._reseed_rng = reseed_rng or random.Random()
        self._generate_calls = 0

    @staticmethod
    def _inc(nonce: bytes) -> bytes:
        value = int.from_bytes(nonce, "big") + 1
        return (value % (1 << 128)).to_bytes(16, "big")

    def reseed(self, entropy: bytes, additional: bytes = b"") -> None:
        seed = hashlib.sha512(entropy + additional).digest()
        self._key = seed[:32]
        self._nonce = seed[32:48]

    def _maybe_reseed(self) -> None:
        if "periodic" in self._reseed_mode and self._reseed_interval > 0:
            if self._generate_calls > 0 and self._generate_calls % self._reseed_interval == 0:
                entropy = _randbytes(self._reseed_rng, 32)
                nonce = _randbytes(self._reseed_rng, 16)
                self.reseed(entropy + nonce)
        if "burst" in self._reseed_mode and self._reseed_prob > 0.0:
            if self._reseed_rng.random() < self._reseed_prob:
                entropy = _randbytes(self._reseed_rng, 32)
                nonce = _randbytes(self._reseed_rng, 16)
                self.reseed(entropy + nonce)

    def randbytes(self, n: int, additional: bytes = b"") -> bytes:
        self._maybe_reseed()
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("ChaCha20-DRBG requires cryptography.") from exc
        if additional:
            self.reseed(additional)
        cipher = Cipher(algorithms.ChaCha20(self._key, self._nonce), mode=None)
        encryptor = cipher.encryptor()
        out = encryptor.update(b"\x00" * n)
        mix = hashlib.sha256(self._key + self._nonce + out[:32]).digest()
        self._key = mix
        self._nonce = self._inc(self._nonce)
        self._generate_calls += 1
        return out


class DRBGIntRNG:
    def __init__(self, drbg, order: int) -> None:
        self._drbg = drbg
        self._order = order
        self._nbytes = max(1, (order.bit_length() + 7) // 8)

    def next_k(self) -> int:
        candidate = int.from_bytes(self._drbg.randbytes(self._nbytes), "big")
        return 1 + (candidate % (self._order - 1))


class DRBGBytesRNG:
    def __init__(self, drbg) -> None:
        self._drbg = drbg

    def randbytes(self, n: int) -> bytes:
        return self._drbg.randbytes(n)


_RNG_ALIAS = {
    "reference": "reference",
    "bounded_interval": "bounded",
    "grid_comb": "grid",
    "recurrent_low_rank": "recurrent",
    "reseed_periodic": "grid",
    "reseed_low_entropy_pool": "bounded",
    "reseed_biased_pool": "bounded",
    "reseed_stalled": "recurrent",
    "reseed_burst": "grid",
}


def _make_drbg(name: str, rng: random.Random, params: Mapping[str, Any] | None):
    entropy = _randbytes(rng, 32)
    nonce = _randbytes(rng, 16)
    params = params or {}
    personalization = str(params.get("personalization") or name).encode("utf-8")
    reseed_mode = str(params.get("reseed_mode", "none"))
    reseed_interval = int(params.get("reseed_interval", 0))
    reseed_prob = float(params.get("reseed_prob", 0.0))
    key = name.lower()
    if key == "hmac_drbg":
        return HmacDRBG(
            entropy,
            nonce,
            personalization,
            reseed_mode=reseed_mode,
            reseed_interval=reseed_interval,
            reseed_prob=reseed_prob,
            reseed_rng=rng,
        )
    if key == "ctr_drbg":
        return CtrDRBG(
            entropy,
            nonce,
            personalization,
            reseed_mode=reseed_mode,
            reseed_interval=reseed_interval,
            reseed_prob=reseed_prob,
            reseed_rng=rng,
        )
    if key == "chacha20_drbg":
        return ChaCha20DRBG(
            entropy,
            nonce,
            personalization,
            reseed_mode=reseed_mode,
            reseed_interval=reseed_interval,
            reseed_prob=reseed_prob,
            reseed_rng=rng,
        )
    raise ValueError(f"unsupported DRBG '{name}'")


def _make_int_rng(name: str, rng: random.Random, order: int, params: Mapping[str, Any] | None = None):
    key = name.strip().lower()
    if key in _RNG_ALIAS:
        key = _RNG_ALIAS[key]
    if key in {"reference", "bounded", "grid", "recurrent"}:
        return IntRNGFamily(key, rng, order, params)
    if key in {"hmac_drbg", "ctr_drbg", "chacha20_drbg"}:
        return DRBGIntRNG(_make_drbg(key, rng, params), order)
    raise ValueError(f"unsupported RNG family '{name}'")


def _make_bytes_rng(name: str, rng: random.Random, params: Mapping[str, Any] | None = None):
    key = name.strip().lower()
    if key in _RNG_ALIAS:
        key = _RNG_ALIAS[key]
    if key in {"reference", "bounded", "grid", "recurrent"}:
        return BytesRNGFamily(key, rng, params)
    if key in {"hmac_drbg", "ctr_drbg", "chacha20_drbg"}:
        return DRBGBytesRNG(_make_drbg(key, rng, params))
    raise ValueError(f"unsupported RNG family '{name}'")


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
            float(1.0 - len(set(r_mod)) / len(r_mod)),
            float(1.0 - len(set(s_mod)) / len(s_mod)),
            float(np.corrcoef(r_norm[:-1], r_norm[1:])[0, 1]) if len(r_norm) > 1 else 0.0,
            float(np.corrcoef(s_norm[:-1], s_norm[1:])[0, 1]) if len(s_norm) > 1 else 0.0,
        ],
        dtype=float,
    )
    return features


def _ecdsa_u_series(rs_pairs: list[tuple[int, int]], order: int) -> np.ndarray:
    if not rs_pairs:
        return np.zeros((0, 2), dtype=float)
    values = [[r / order, s / order] for r, s in rs_pairs]
    return np.asarray(values, dtype=float)


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
            float(1.0 - len({hashlib.sha256(sig).digest() for sig in signatures}) / len(signatures)),
            float(np.corrcoef(mean_bytes_arr[:-1], mean_bytes_arr[1:])[0, 1]) if len(mean_bytes_arr) > 1 else 0.0,
        ],
        dtype=float,
    )
    return features


def _dilithium_u_series(signatures: list[bytes], width: int = 32) -> np.ndarray:
    if not signatures:
        return np.zeros((0, width), dtype=float)
    rows = []
    for sig in signatures:
        arr = np.frombuffer(sig, dtype=np.uint8)
        if arr.size < width:
            padded = np.zeros(width, dtype=np.uint8)
            padded[: arr.size] = arr
            arr = padded
        rows.append(arr[:width] / 255.0)
    return np.vstack(rows)


def _train_detector(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("OND-RNG-Bench requires scikit-learn.") from exc

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


def _patched_system_rng(randbytes_fn):
    class _Context:
        def __enter__(self):
            self._orig_urandom = os.urandom
            self._orig_token = secrets.token_bytes
            os.urandom = lambda n: randbytes_fn(n)  # type: ignore[assignment]
            secrets.token_bytes = lambda n: randbytes_fn(n)  # type: ignore[assignment]
            return self

        def __exit__(self, exc_type, exc, tb):
            os.urandom = self._orig_urandom
            secrets.token_bytes = self._orig_token
            return False

    return _Context()


def _dilithium_keypair(dilithium, rng_family: BytesRNGFamily):
    with _patched_system_rng(rng_family.randbytes):
        keygen = None
        for name in ("keygen", "keypair", "generate_keypair"):
            candidate = getattr(dilithium, name, None)
            if callable(candidate):
                keygen = candidate
                break
        if keygen is None:
            raise RuntimeError("Dilithium2 key generation method not found")
        return keygen()


def _dilithium_signature_set(
    *,
    n_signatures: int,
    rng_family: BytesRNGFamily,
    dilithium,
    sk,
    seed_prefix: str,
) -> list[bytes]:
    signatures: list[bytes] = []
    with _patched_system_rng(rng_family.randbytes):
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


def _run_ecdsa(config: BenchConfig) -> tuple[dict[str, list[CurveSeries]], dict[str, Any]]:
    try:
        from ecdsa import SECP256k1, SigningKey
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("ECDSA benchmark requires the 'ecdsa' package.") from exc

    curve = SECP256k1
    order = curve.order
    rng_specs = [spec for spec in config.rng_families if spec.name]
    rng_labels = [spec.label or spec.name for spec in rng_specs]
    auc_series = {label: [] for label in rng_labels}
    tpr_series = {label: [] for label in rng_labels}
    ond_metrics: dict[str, Any] = {label: {"reference": {}, "variant": {}} for label in rng_labels}

    for n in config.n_values:
        for family_index, spec in enumerate(rng_specs):
            family = spec.name
            label = spec.label or spec.name
            auc_runs = []
            tpr_runs = []
            ond_ref_acc = {"H_rank": [], "H_sub": [], "H_branch": []}
            ond_var_acc = {"H_rank": [], "H_sub": [], "H_branch": []}
            for repeat in range(config.repeats):
                repeat_seed = config.reproducibility.seed + n * 10_000 + repeat * 101 + family_index * 1000
                rng = random.Random(repeat_seed)
                secret = rng.randrange(1, order)
                signer = SigningKey.from_secret_exponent(secret, curve=curve, hashfunc=hashlib.sha256)

                ref_features = []
                var_features = []
                for trial in range(config.trials):
                    reference_rng = _make_int_rng(
                        "reference",
                        random.Random(rng.randrange(1, 2**32)),
                        order,
                    )
                    family_rng = _make_int_rng(
                        family,
                        random.Random(rng.randrange(1, 2**32)),
                        order,
                        spec.params,
                    )

                    rs_ref = _ecdsa_signature_set(
                        n_signatures=n,
                        rng_family=reference_rng,
                        signer=signer,
                        order=order,
                        seed_prefix=f"ref-{repeat}-{trial}",
                    )
                    rs_var = _ecdsa_signature_set(
                        n_signatures=n,
                        rng_family=family_rng,
                        signer=signer,
                        order=order,
                        seed_prefix=f"{family}-{repeat}-{trial}",
                    )

                    ref_features.append(_ecdsa_features(rs_ref))
                    var_features.append(_ecdsa_features(rs_var))

                    ref_u = _ecdsa_u_series(rs_ref, order)
                    var_u = _ecdsa_u_series(rs_var, order)
                    ref_metrics = compute_ond_metrics(ref_u)
                    var_metrics = compute_ond_metrics(var_u)
                    for key in ond_ref_acc:
                        ond_ref_acc[key].append(ref_metrics[key])
                        ond_var_acc[key].append(var_metrics[key])

                X = np.vstack([ref_features, var_features])
                y = np.array([0] * len(ref_features) + [1] * len(var_features))
                y_true, scores = _train_detector(X, y, np.random.default_rng(repeat_seed))
                metrics = compute_binary_metrics(y_true, scores, fpr_target=config.metrics.fpr_target)
                auc_runs.append(metrics.auc)
                tpr_runs.append(metrics.tpr_at_fpr)

            auc_series[label].append(float(np.mean(auc_runs)))
            tpr_series[label].append(float(np.mean(tpr_runs)))
            for key in ond_ref_acc:
                ond_metrics[label]["reference"].setdefault(key, []).append(float(np.mean(ond_ref_acc[key])))
                ond_metrics[label]["variant"].setdefault(key, []).append(float(np.mean(ond_var_acc[key])))

    curves = {
        "AUC": [CurveSeries(label=name, values=auc_series[name]) for name in rng_labels],
        f"TPR@FPR={config.metrics.fpr_target:.2f}": [
            CurveSeries(label=name, values=tpr_series[name]) for name in rng_labels
        ],
    }
    return curves, ond_metrics


def _run_dilithium(config: BenchConfig) -> tuple[dict[str, list[CurveSeries]], dict[str, Any]]:
    dilithium = _resolve_dilithium()
    rng_specs = [spec for spec in config.rng_families if spec.name]
    rng_labels = [spec.label or spec.name for spec in rng_specs]
    auc_series = {label: [] for label in rng_labels}
    tpr_series = {label: [] for label in rng_labels}
    ond_metrics: dict[str, Any] = {label: {"reference": {}, "variant": {}} for label in rng_labels}

    for n in config.n_values:
        for family_index, spec in enumerate(rng_specs):
            family = spec.name
            label = spec.label or spec.name
            auc_runs = []
            tpr_runs = []
            ond_ref_acc = {"H_rank": [], "H_sub": [], "H_branch": []}
            ond_var_acc = {"H_rank": [], "H_sub": [], "H_branch": []}
            for repeat in range(config.repeats):
                repeat_seed = config.reproducibility.seed + n * 10_000 + repeat * 101 + family_index * 1000
                rng = random.Random(repeat_seed)
                reference_rng = _make_bytes_rng("reference", random.Random(rng.randrange(1, 2**32)))
                pk, sk = _dilithium_keypair(dilithium, reference_rng)

                ref_features = []
                var_features = []
                for trial in range(config.trials):
                    reference_rng = _make_bytes_rng("reference", random.Random(rng.randrange(1, 2**32)))
                    family_rng = _make_bytes_rng(
                        family,
                        random.Random(rng.randrange(1, 2**32)),
                        spec.params,
                    )

                    sig_ref = _dilithium_signature_set(
                        n_signatures=n,
                        rng_family=reference_rng,
                        dilithium=dilithium,
                        sk=sk,
                        seed_prefix=f"ref-{repeat}-{trial}",
                    )
                    sig_var = _dilithium_signature_set(
                        n_signatures=n,
                        rng_family=family_rng,
                        dilithium=dilithium,
                        sk=sk,
                        seed_prefix=f"{family}-{repeat}-{trial}",
                    )

                    ref_features.append(_dilithium_features(sig_ref))
                    var_features.append(_dilithium_features(sig_var))

                    ref_u = _dilithium_u_series(sig_ref)
                    var_u = _dilithium_u_series(sig_var)
                    ref_metrics = compute_ond_metrics(ref_u)
                    var_metrics = compute_ond_metrics(var_u)
                    for key in ond_ref_acc:
                        ond_ref_acc[key].append(ref_metrics[key])
                        ond_var_acc[key].append(var_metrics[key])

                X = np.vstack([ref_features, var_features])
                y = np.array([0] * len(ref_features) + [1] * len(var_features))
                y_true, scores = _train_detector(X, y, np.random.default_rng(repeat_seed))
                metrics = compute_binary_metrics(y_true, scores, fpr_target=config.metrics.fpr_target)
                auc_runs.append(metrics.auc)
                tpr_runs.append(metrics.tpr_at_fpr)

            auc_series[label].append(float(np.mean(auc_runs)))
            tpr_series[label].append(float(np.mean(tpr_runs)))
            for key in ond_ref_acc:
                ond_metrics[label]["reference"].setdefault(key, []).append(float(np.mean(ond_ref_acc[key])))
                ond_metrics[label]["variant"].setdefault(key, []).append(float(np.mean(ond_var_acc[key])))

    curves = {
        "AUC": [CurveSeries(label=name, values=auc_series[name]) for name in rng_labels],
        f"TPR@FPR={config.metrics.fpr_target:.2f}": [
            CurveSeries(label=name, values=tpr_series[name]) for name in rng_labels
        ],
    }
    return curves, ond_metrics


def run_benchmark(config: BenchConfig) -> BenchmarkResult:
    if config.protocol.name == "ecdsa_secp256k1":
        curves, ond_metrics = _run_ecdsa(config)
    elif config.protocol.name == "dilithium2":
        curves, ond_metrics = _run_dilithium(config)
    else:
        raise ValueError(f"unsupported protocol '{config.protocol.name}'")

    config_hash = bench_config_hash(config.model_dump(by_alias=True))
    bench_id = _bench_run_id(config, config_hash)
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": config.protocol.name,
        "n_values": config.n_values,
        "rng_families": [
            {
                "name": spec.name,
                "label": spec.label or spec.name,
                "class_label": spec.class_label,
                "params": spec.params or {},
            }
            for spec in config.rng_families
        ],
        "trials": config.trials,
        "repeats": config.repeats,
        "metrics": {"fpr_target": config.metrics.fpr_target},
        "library_versions": _collect_library_versions(),
    }
    return BenchmarkResult(
        bench_id=bench_id,
        config_hash=config_hash,
        curves=curves,
        ond_metrics=ond_metrics,
        metadata=metadata,
    )


def save_benchmark_result(result: BenchmarkResult, config: Mapping[str, Any], base_dir: str | Path) -> Path:
    base = Path(base_dir)
    run_dir = base / result.bench_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    (run_dir / "config.yaml").write_text(canonical_bench_yaml(config), encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "bench_id": result.bench_id,
                "config_hash": result.config_hash,
                "metadata": result.metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    curves_payload = {
        "bench_id": result.bench_id,
        "config_hash": result.config_hash,
        "n_values": result.metadata["n_values"],
        "metrics": {
            name: [series.__dict__ for series in series_list]
            for name, series_list in result.curves.items()
        },
        "default_metric": "AUC",
    }
    (run_dir / "curves.json").write_text(json.dumps(curves_payload, indent=2), encoding="utf-8")

    ond_payload = {
        "bench_id": result.bench_id,
        "config_hash": result.config_hash,
        "n_values": result.metadata["n_values"],
        "families": result.ond_metrics,
    }
    (run_dir / "ond_metrics.json").write_text(json.dumps(ond_payload, indent=2), encoding="utf-8")

    return run_dir
