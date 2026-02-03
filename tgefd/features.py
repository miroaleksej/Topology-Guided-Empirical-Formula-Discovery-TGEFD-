from __future__ import annotations

import re
from typing import Iterable, Mapping, Sequence

import numpy as np

from .models import HypothesisParams

_DEFAULT_FEATURES: list[str] = ["1", "x", "x^p", "sin(x)", "sin(x)^q", "exp(-x)"]


def _safe_power(x: np.ndarray, power: float) -> np.ndarray:
    """Real-valued power for possibly negative x."""
    if np.isclose(power, round(power)):
        p_int = int(round(power))
        if p_int % 2 == 0:
            return np.abs(x) ** p_int
        return np.sign(x) * (np.abs(x) ** p_int)
    return np.sign(x) * (np.abs(x) ** power)


def _normalize_expr(expr: str) -> str:
    return expr.replace(" ", "")


_NUM_RE = re.compile(r"^[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?$")


def _parse_exponent(token: str, params: Mapping[str, float]) -> float:
    if token in params:
        return float(params[token])
    if _NUM_RE.match(token):
        return float(token)
    raise ValueError(f"Unsupported exponent token '{token}'")


def _coerce_x(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim == 2:
        return x
    raise ValueError("x must be 1D or 2D")


def _get_var(x: np.ndarray, token: str) -> np.ndarray:
    if token == "x":
        return x[:, 0]
    if re.fullmatch(r"x\d+", token):
        idx = int(token[1:])
        if idx >= x.shape[1]:
            raise ValueError(f"feature references x{idx} but x has only {x.shape[1]} columns")
        return x[:, idx]
    raise ValueError(f"Unsupported variable token '{token}'")


def _eval_base(token: str, x: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    token = _normalize_expr(token)
    if token in ("1", "const"):
        return np.ones(x.shape[0], dtype=float)
    if token.startswith("sin(") and token.endswith(")"):
        inner = token[4:-1]
        return np.sin(_eval_base(inner, x, params))
    if token.startswith("cos(") and token.endswith(")"):
        inner = token[4:-1]
        return np.cos(_eval_base(inner, x, params))
    if token.startswith("exp(") and token.endswith(")"):
        inner = token[4:-1]
        if not inner.startswith("-"):
            raise ValueError(f"exp() supports only negative argument, got '{token}'")
        return np.exp(-_eval_base(inner[1:], x, params))
    return _get_var(x, token)


def _eval_factor(token: str, x: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    token = _normalize_expr(token)
    if "^" in token:
        base, exp = token.split("^", 1)
        exponent = _parse_exponent(exp, params)
        return _safe_power(_eval_base(base, x, params), exponent)
    return _eval_base(token, x, params)


def _eval_expression(expr: str, x: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    expr = _normalize_expr(expr)
    factors = expr.split("*")
    values = [_eval_factor(factor, x, params) for factor in factors]
    out = values[0]
    for val in values[1:]:
        out = out * val
    return out


def feature_library(
    x: np.ndarray,
    params: HypothesisParams,
    features: Sequence[str] | None = None,
) -> np.ndarray:
    """Build a hypothesis feature library for 1D or 2D x."""
    x = _coerce_x(x)

    if features is None:
        features = _DEFAULT_FEATURES

    params_map = {"p": float(params.p), "q": float(params.q)}
    columns = [_eval_expression(expr, x, params_map) for expr in features]
    return np.column_stack(columns)


def format_feature_expression(
    expr: str,
    params: Mapping[str, float] | None = None,
    precision: int = 4,
) -> str:
    expr = _normalize_expr(expr)
    if "^" not in expr or not params:
        return expr
    base, token = expr.rsplit("^", 1)
    if token not in params:
        return expr
    value = params[token]
    fmt = f"{{:.{precision}g}}"
    return f"{base}^{fmt.format(value)}"


def required_params_from_features(features: Iterable[str]) -> list[str]:
    required: list[str] = []
    for expr in features:
        expr = _normalize_expr(expr)
        if "^" not in expr:
            continue
        _, token = expr.rsplit("^", 1)
        if _NUM_RE.match(token):
            continue
        if token not in required:
            required.append(token)
    return required


__all__ = [
    "feature_library",
    "format_feature_expression",
    "required_params_from_features",
]
