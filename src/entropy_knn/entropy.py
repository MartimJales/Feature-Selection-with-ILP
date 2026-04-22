"""Entropy utilities for the Entropy KNN pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def shannon_entropy(values: pd.Series | np.ndarray) -> float:
    """Compute Shannon entropy in bits."""
    arr = np.asarray(values)
    if arr.ndim > 1:
        arr = arr.ravel()

    valid_mask = pd.notna(arr)
    arr = arr[valid_mask]
    if arr.size == 0:
        return 0.0

    _, counts = np.unique(arr, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(np.where(probabilities > 0, probabilities * np.log2(probabilities), 0.0))
    return float(entropy)


def conditional_entropy(target: pd.Series | np.ndarray, feature: pd.Series | np.ndarray) -> float:
    """Compute H(target | feature)."""
    y = np.asarray(target)
    x = np.asarray(feature)

    if y.ndim > 1:
        y = y.ravel()
    if x.ndim > 1:
        x = x.ravel()

    valid_mask = pd.notna(x) & pd.notna(y)
    x = x[valid_mask]
    y = y[valid_mask]

    if x.size == 0:
        return 0.0

    contingency = pd.crosstab(pd.Series(x, name="x"), pd.Series(y, name="y"), dropna=True)
    if contingency.empty:
        return 0.0

    matrix = contingency.to_numpy(dtype=float)
    row_totals = matrix.sum(axis=1)
    total = row_totals.sum()
    if total <= 0.0:
        return 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        probs = matrix / row_totals[:, None]
        log_probs = np.where(probs > 0, np.log2(probs), 0.0)
        row_entropy = -np.sum(probs * log_probs, axis=1)

    weights = row_totals / total
    return float(np.sum(weights * row_entropy))


def entropy_reduction_ratio(target: pd.Series | np.ndarray, feature: pd.Series | np.ndarray) -> float:
    """Compute normalized entropy reduction."""
    base_entropy = shannon_entropy(target)
    if base_entropy <= 0.0:
        return 0.0

    return float((base_entropy - conditional_entropy(target, feature)) / base_entropy)
