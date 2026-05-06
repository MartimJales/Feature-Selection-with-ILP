"""Scatter plots for method score comparisons."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from .common import METHODS, METHOD_LABELS, list_cluster_json_files, load_cluster_json

logger = logging.getLogger(__name__)


def _load_top1_values(cluster_json_dir: Path) -> pd.DataFrame:
    rows = []
    for json_file in list_cluster_json_files(cluster_json_dir):
        cluster_data = load_cluster_json(json_file)
        top_by_method = cluster_data.get("top_features_by_method", {})
        row = {"cluster_id": cluster_data["cluster_id"]}
        for method in METHODS:
            if method in top_by_method:
                row[f"{method}_value"] = top_by_method[method]["value"]
                row[f"{method}_feature"] = top_by_method[method]["feature"]
        rows.append(row)
    return pd.DataFrame(rows)


def generate_top1_scatter_grid(cluster_json_dir: Path, output_path: Path) -> pd.DataFrame:
    """Generate scatter plots comparing top-1 scores between method pairs."""
    df = _load_top1_values(cluster_json_dir)
    if df.empty:
        raise FileNotFoundError(f"No usable cluster JSON files found in {cluster_json_dir}")

    pairs = [
        ("entropy_reduction_ratio", "mutual_information"),
        ("entropy_reduction_ratio", "chi2_stat"),
        ("entropy_reduction_ratio", "f_stat"),
        ("entropy_reduction_ratio", "pearson_r"),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (method_x, method_y) in zip(axes, pairs):
        x_col = f"{method_x}_value"
        y_col = f"{method_y}_value"
        sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, s=45, alpha=0.75)
        correlation = spearmanr(df[x_col], df[y_col]).correlation if len(df) > 1 else float("nan")
        ax.set_title(f"{METHOD_LABELS[method_x]} vs {METHOD_LABELS[method_y]}\nSpearman={correlation:.2f}")
        ax.set_xlabel(METHOD_LABELS[method_x])
        ax.set_ylabel(METHOD_LABELS[method_y])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote scatter grid to %s", output_path)
    return df
