"""Bar charts for method agreement summaries."""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .common import METHODS, METHOD_LABELS, list_cluster_json_files, load_cluster_json, top_k_feature_sets

logger = logging.getLogger(__name__)


def generate_agreement_bars(cluster_json_dir: Path, output_path: Path, top_k: int = 5) -> pd.DataFrame:
    """Generate a bar chart with mean pairwise top-k overlap between methods."""
    json_files = list_cluster_json_files(cluster_json_dir)
    if not json_files:
        raise FileNotFoundError(f"No cluster_*.json files found in {cluster_json_dir}")

    pair_to_values: dict[str, list[float]] = {}
    for json_file in json_files:
        cluster_data = load_cluster_json(json_file)
        feature_sets = top_k_feature_sets(cluster_data, top_k=top_k)
        for method_a, method_b in combinations(METHODS, 2):
            set_a = feature_sets.get(method_a, set())
            set_b = feature_sets.get(method_b, set())
            union = set_a | set_b
            jaccard = len(set_a & set_b) / len(union) if union else 0.0
            label = f"{METHOD_LABELS[method_a]} vs {METHOD_LABELS[method_b]}"
            pair_to_values.setdefault(label, []).append(jaccard)

    summary_df = pd.DataFrame(
        [
            {"pair": pair, "mean_jaccard": sum(values) / len(values) if values else 0.0}
            for pair, values in pair_to_values.items()
        ]
    ).sort_values("mean_jaccard", ascending=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=summary_df,
        x="mean_jaccard",
        y="pair",
        hue="pair",
        palette="viridis",
        dodge=False,
        legend=False,
    )
    plt.xlabel(f"Mean top-{top_k} Jaccard overlap")
    plt.ylabel("Method pair")
    plt.title(f"Top-{top_k} overlap between filter methods\n{cluster_json_dir.name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Wrote agreement bars to %s", output_path)
    return summary_df
