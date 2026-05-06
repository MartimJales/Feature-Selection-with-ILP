"""Top-k Venn diagrams for comparing filter methods."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib_venn import venn2

from .common import METHOD_LABELS, METHODS, load_cluster_json, top_k_feature_sets

logger = logging.getLogger(__name__)


def generate_venn_grid(cluster_json_path: Path, output_path: Path, top_k: int = 5) -> None:
    """Generate a grid of pairwise Venn diagrams for one cluster JSON."""
    cluster_data = load_cluster_json(cluster_json_path)
    feature_sets = top_k_feature_sets(cluster_data, top_k=top_k)

    base_method = "entropy_reduction_ratio"
    compare_methods = ["mutual_information", "chi2_stat", "f_stat"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(compare_methods), figsize=(16, 5))
    if len(compare_methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, compare_methods):
        set_a = feature_sets.get(base_method, set())
        set_b = feature_sets.get(method, set())
        venn2(
            [set_a, set_b],
            set_labels=(METHOD_LABELS[base_method], METHOD_LABELS[method]),
            ax=ax,
        )
        ax.set_title(f"Cluster {cluster_data['cluster_id']}\nTop-{top_k}")

    fig.suptitle(f"Top-{top_k} feature overlap for cluster {cluster_data['cluster_id']}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote venn diagram to %s", output_path)


def generate_venn_for_directory(cluster_json_dir: Path, output_dir: Path, top_k: int = 5) -> list[Path]:
    """Generate one Venn diagram per cluster JSON in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for cluster_json_path in sorted(cluster_json_dir.glob("cluster_*.json")):
        output_path = output_dir / f"{cluster_json_path.stem}_venn_top{top_k}.png"
        generate_venn_grid(cluster_json_path, output_path, top_k=top_k)
        written_paths.append(output_path)
    return written_paths
