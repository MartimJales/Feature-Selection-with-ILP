"""Entropy utilities for the Entropy KNN pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def shannon_entropy(values: pd.Series | np.ndarray) -> float:
    """Compute Shannon entropy in bits."""
    series = pd.Series(values).dropna()
    if series.empty:
        return 0.0

    probabilities = series.value_counts(normalize=True)
    return float(-(probabilities * np.log2(probabilities)).sum())


def conditional_entropy(target: pd.Series | np.ndarray, feature: pd.Series | np.ndarray) -> float:
    """Compute H(target | feature)."""
    y = pd.Series(target).reset_index(drop=True)
    x = pd.Series(feature).reset_index(drop=True)

    valid_mask = x.notna() & y.notna()
    x = x[valid_mask]
    y = y[valid_mask]

    if x.empty:
        return 0.0

    total = len(x)
    conditional = 0.0

    for _, group_index in x.groupby(x).groups.items():
        group_y = y.loc[group_index]
        conditional += (len(group_y) / total) * shannon_entropy(group_y)

    return float(conditional)


def entropy_reduction_ratio(target: pd.Series | np.ndarray, feature: pd.Series | np.ndarray) -> float:
    """Compute normalized entropy reduction."""
    base_entropy = shannon_entropy(target)
    if base_entropy <= 0.0:
        return 0.0

    return float((base_entropy - conditional_entropy(target, feature)) / base_entropy)
