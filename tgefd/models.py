from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class HypothesisParams:
    p: float
    q: float
    lam: float


@dataclass(frozen=True)
class ModelResult:
    params: HypothesisParams
    coeffs: np.ndarray
    error: float
    l0: int
    l1: float

    def active_terms(self) -> Iterable[int]:
        return np.flatnonzero(self.coeffs)
