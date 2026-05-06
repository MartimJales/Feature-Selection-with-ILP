"""Heatmap of average rank agreement between filter methods."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from .common import (
    METHODS,
    METHOD_LABELS,
    build_rank_frame,
    build_score_frame,
    list_cluster_json_files,
    load_cluster_json,
)

logger = logging.getLogger(__name__)


def _safe_spearman(values_a: pd.Series, values_b: pd.Series) -> float | None:
    if values_a.nunique(dropna=False) <= 1 or values_b.nunique(dropna=False) <= 1:
        return None

    correlation = spearmanr(values_a, values_b).correlation
    if correlation is None or np.isnan(correlation):
        return None
    return float(correlation)


def generate_spearman_heatmap(cluster_json_dir: Path, output_path: Path) -> pd.DataFrame:
    """Generate a heatmap with the mean Spearman rank correlation between methods."""
    json_files = list_cluster_json_files(cluster_json_dir)
    if not json_files:
        raise FileNotFoundError(f"No cluster_*.json files found in {cluster_json_dir}")

    pairwise_values: dict[tuple[str, str], list[float]] = {(a, b): [] for a in METHODS for b in METHODS}

    for json_file in json_files:
        cluster_data = load_cluster_json(json_file)
        score_frame = build_score_frame(cluster_data)
        rank_frame = build_rank_frame(score_frame)
        if rank_frame.empty:
            continue

        feature_columns = [method for method in METHODS if method in rank_frame.columns]
        if len(feature_columns) < 2:
            continue

        for i, method_a in enumerate(feature_columns):
            for method_b in feature_columns[i:]:
                correlation = _safe_spearman(rank_frame[method_a], rank_frame[method_b])
                if correlation is None:
                    continue
                pairwise_values[(method_a, method_b)].append(correlation)
                if method_a != method_b:
                    pairwise_values[(method_b, method_a)].append(correlation)

    matrix = pd.DataFrame(index=METHODS, columns=METHODS, dtype=float)
    for method_a in METHODS:
        for method_b in METHODS:
            values = pairwise_values.get((method_a, method_b), [])
            if method_a == method_b:
                matrix.loc[method_a, method_b] = 1.0
            else:
                matrix.loc[method_a, method_b] = float(np.nanmean(values)) if values else np.nan

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": "Mean Spearman correlation"},
        xticklabels=[METHOD_LABELS[m] for m in METHODS],
        yticklabels=[METHOD_LABELS[m] for m in METHODS],
    )
    plt.title(f"Mean Spearman rank correlation\n{cluster_json_dir.name}")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Wrote heatmap to %s", output_path)

    return matrix
