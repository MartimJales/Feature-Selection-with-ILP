"""Venn diagrams: intersection of top-K features by method.

For a chosen cluster, generate Venn diagrams showing pairwise
overlap of top-K features selected by each filter method.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib_venn import venn2

workspace_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(workspace_root))

from src.entropy_knn.visualizations.common import METHODS, METHOD_LABELS


def _load_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def generate_venn_by_methods(scores_df: pd.DataFrame, output_path: Path, cluster_id: int | None = None, top_k: int = 5) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    methods_present = [m for m in METHODS if m in scores_df.columns]
    if not methods_present:
        numeric_cols = [c for c in scores_df.columns if scores_df[c].dtype.kind in "fi" and c not in ("cluster_id", "feature_rank")]
        methods_present = numeric_cols

    # default to first cluster if not specified
    if cluster_id is None:
        cluster_id = scores_df["cluster_id"].iloc[0]

    cluster_df = scores_df[scores_df["cluster_id"] == cluster_id].copy()
    if cluster_df.empty:
        print(f"[venn_by_methods] No data for cluster {cluster_id}")
        return

    # normalize columns
    for method in methods_present:
        cluster_df[method] = pd.to_numeric(cluster_df[method], errors="coerce").fillna(0.0)

    # get top-K per method
    feature_sets = {}
    for method in methods_present:
        top_features = cluster_df.nlargest(top_k, method)["feature"].astype(str).tolist()
        feature_sets[method] = set(top_features)

    if len(methods_present) < 2:
        print("[venn_by_methods] Need at least 2 methods to generate a Venn diagram.")
        return

    # pairwise Venn diagrams
    n_methods = len(methods_present)
    n_cols = min(3, n_methods)
    pair_count = n_methods * (n_methods - 1) // 2
    n_rows = max(1, math.ceil(pair_count / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    idx = 0
    for i, m1 in enumerate(methods_present):
        for m2 in methods_present[i + 1 :]:
            ax = axes[idx] if idx < len(axes) else None
            if ax is None:
                break

            venn2([feature_sets[m1], feature_sets[m2]], set_labels=(METHOD_LABELS.get(m1, m1), METHOD_LABELS.get(m2, m2)), ax=ax)
            ax.set_title(f"Cluster {cluster_id}: {METHOD_LABELS.get(m1, m1)} vs {METHOD_LABELS.get(m2, m2)}")
            idx += 1

    # hide unused subplots
    for j in range(idx, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Top-{top_k} feature overlap by method (cluster {cluster_id})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[venn_by_methods] Wrote to: {output_path}")


def _parse_args():
    workspace_root = Path(__file__).resolve().parents[3]
    default_scores = workspace_root / "reports" / "entropy_knn" / "score_only" / "cluster_500" / "seed_42" / "cluster_feature_scores.parquet"
    parser = argparse.ArgumentParser(description="Generate Venn diagrams for method agreement")
    parser.add_argument("--scores", type=Path, default=default_scores, help="Path to cluster_feature_scores.parquet or CSV")
    parser.add_argument("--output-path", type=Path, default=workspace_root / "reports" / "entropy_knn" / "analysis" / "visualizations" / "venn_by_methods.png")
    parser.add_argument("--cluster-id", type=int, default=None, help="Cluster ID to visualize (default: first)")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main():
    args = _parse_args()
    scores_df = _load_scores(args.scores)
    generate_venn_by_methods(scores_df, args.output_path, cluster_id=args.cluster_id, top_k=args.top_k)


if __name__ == "__main__":
    main()
