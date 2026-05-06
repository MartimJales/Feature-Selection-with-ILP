"""Heatmap: feature-feature co-occurrence in top-K.

Compute Jaccard agreement between pairs of features: how often they
appear together in top-K across clusters and methods.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

workspace_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(workspace_root))

from src.entropy_knn.visualizations.common import METHODS
from src.analysis.entropy_knn_visualizations.data_sources import load_scores_for_analysis


def generate_feature_pair_heatmap(scores_df: pd.DataFrame, output_path: Path, top_k: int = 5, top_m: int = 30) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    methods_present = [m for m in METHODS if m in scores_df.columns]
    if not methods_present:
        numeric_cols = [c for c in scores_df.columns if scores_df[c].dtype.kind in "fi" and c not in ("cluster_id", "feature_rank")]
        methods_present = numeric_cols

    # collect top-K features from all clusters + methods
    all_top_features = set()
    feature_pairs: dict[tuple[str, str], int] = {}

    for cid in scores_df["cluster_id"].unique():
        cluster_df = scores_df[scores_df["cluster_id"] == cid].copy()
        if cluster_df.empty:
            continue

        for method in methods_present:
            cluster_df[method] = pd.to_numeric(cluster_df[method], errors="coerce").fillna(0.0)
            top_features = cluster_df.nlargest(top_k, method)["feature"].astype(str).tolist()
            all_top_features.update(top_features)

            # pairwise co-occurrence
            for i, f1 in enumerate(top_features):
                for f2 in top_features[i + 1 :]:
                    key = tuple(sorted([f1, f2]))
                    feature_pairs[key] = feature_pairs.get(key, 0) + 1

    # select top M features by frequency
    top_features_list = sorted(all_top_features, key=lambda f: sum(1 for (f1, f2) in feature_pairs if f in (f1, f2)), reverse=True)[:top_m]
    top_features_list = sorted(top_features_list)

    if len(top_features_list) < 2:
        print("[feature_pair_heatmap] Not enough features for heatmap.")
        return

    # build co-occurrence matrix
    matrix = pd.DataFrame(0, index=top_features_list, columns=top_features_list)
    for (f1, f2), count in feature_pairs.items():
        if f1 in top_features_list and f2 in top_features_list:
            matrix.loc[f1, f2] = count
            matrix.loc[f2, f1] = count

    # diagonal = self-occurrence (count per feature)
    for feat in top_features_list:
        count = sum(1 for (f1, f2), c in feature_pairs.items() if feat in (f1, f2))
        matrix.loc[feat, feat] = count if count > 0 else 1

    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=False, cmap="YlOrRd", square=True, cbar_kws={"label": "Co-occurrence count"})
    plt.title(f"Feature-pair co-occurrence in top-{top_k} (top {top_m} features)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[feature_pair_heatmap] Wrote to: {output_path}")


def _parse_args():
    workspace_root = Path(__file__).resolve().parents[3]
    default_json_dir = workspace_root / "reports" / "entropy_knn" / "score_only" / "cluster_500" / "seed_42"
    default_scores = default_json_dir / "cluster_feature_scores.parquet"
    parser = argparse.ArgumentParser(description="Generate feature-pair co-occurrence heatmap")
    parser.add_argument("--cluster-json-dir", type=Path, default=default_json_dir, help="Directory with cluster_*.json (preferred)")
    parser.add_argument("--scores", type=Path, default=default_scores, help="Fallback path to cluster_feature_scores.parquet or CSV")
    parser.add_argument("--output-path", type=Path, default=workspace_root / "reports" / "entropy_knn" / "analysis" / "visualizations" / "feature_pair_heatmap.png")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-m", type=int, default=30, help="Number of top features to include in heatmap")
    return parser.parse_args()


def main():
    args = _parse_args()
    scores_df = load_scores_for_analysis(cluster_json_dir=args.cluster_json_dir, scores_path=args.scores)
    generate_feature_pair_heatmap(scores_df, args.output_path, top_k=args.top_k, top_m=args.top_m)


if __name__ == "__main__":
    main()
