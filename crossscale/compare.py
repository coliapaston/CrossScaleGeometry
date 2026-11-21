"""Curve-comparison kernels used by the statistics pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr


@dataclass(frozen=True)
class SpearmanResult:
    rho: float
    pvalue: float
    n_obs: int


def _safe_standardize_against_baseline(baseline_values: np.ndarray, seed_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    baseline_values = np.asarray(baseline_values, dtype=float)
    seed_values = np.asarray(seed_values, dtype=float)

    mean = float(np.mean(baseline_values))
    std = float(np.std(baseline_values, ddof=0))
    if np.isclose(std, 0.0):
        return baseline_values - mean, seed_values - mean
    return (baseline_values - mean) / std, (seed_values - mean) / std


def compute_spearman_stats(x: Sequence[float], y: Sequence[float]) -> SpearmanResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"Spearman inputs must have the same shape, got {x.shape} vs {y.shape}")
    if x.size < 2:
        raise ValueError(f"Spearman requires at least 2 observations, got n={x.size}")

    rho, pvalue = spearmanr(x, y)
    return SpearmanResult(rho=float(rho), pvalue=float(pvalue), n_obs=int(x.size))


@dataclass(frozen=True)
class PearsonResult:
    r: float
    pvalue: float
    n_obs: int


def compute_pearson_stats(x: Sequence[float], y: Sequence[float]) -> PearsonResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"Pearson inputs must have the same shape, got {x.shape} vs {y.shape}")
    if x.size < 2:
        raise ValueError(f"Pearson requires at least 2 observations, got n={x.size}")

    rho, pvalue = pearsonr(x, y)
    return PearsonResult(r=float(rho), pvalue=float(pvalue), n_obs=int(x.size))
