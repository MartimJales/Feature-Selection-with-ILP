"""Bar chart: agreement of features across methods.

For each feature, count how many methods include it in their top-K
across all clusters, and show mean aggregated score.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

workspace_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(workspace_root))

from src.entropy_knn.visualizations.common import METHODS
from src.analysis.entropy_knn_visualizations.data_sources import load_scores_for_analysis


def generate_feature_agreement_bars(scores_df: pd.DataFrame, output_path: Path, top_k: int = 5) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    methods_present = [m for m in METHODS if m in scores_df.columns]
    if not methods_present:
        numeric_cols = [c for c in scores_df.columns if scores_df[c].dtype.kind in "fi" and c not in ("cluster_id", "feature_rank")]
        methods_present = numeric_cols

    feature_stats: dict[str, dict] = {}

    for cid in scores_df["cluster_id"].unique():
        cluster_df = scores_df[scores_df["cluster_id"] == cid].copy()
        if cluster_df.empty:
            continue

        for method in methods_present:
            cluster_df[method] = pd.to_numeric(cluster_df[method], errors="coerce").fillna(0.0)
            top_features = cluster_df.nlargest(top_k, method)["feature"].astype(str).tolist()
            for feat in top_features:
                if feat not in feature_stats:
                    feature_stats[feat] = {"method_count": 0, "total_score": 0.0, "clusters_seen": 0}
                feature_stats[feat]["method_count"] += 1
                feature_stats[feat]["total_score"] += float(cluster_df[cluster_df["feature"] == feat][method].values[0]) if feat in cluster_df["feature"].values else 0.0
                feature_stats[feat]["clusters_seen"] += 1

    # aggregate
    rows = []
    for feat, stats in feature_stats.items():
        rows.append({
            "feature": feat,
            "method_agreement_count": stats["method_count"],
            "mean_score": stats["total_score"] / max(1, stats["clusters_seen"]),
        })

    if not rows:
        print("[feature_agreement_by_feature] No features found.")
        return

    df = pd.DataFrame(rows).sort_values(["method_agreement_count", "mean_score"], ascending=[False, False]).head(50)

    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=df,
        x="mean_score",
        y="feature",
        hue="feature",
        palette="viridis",
        dodge=False,
        legend=False,
    )
    plt.xlabel("Mean aggregated score across clusters")
    plt.ylabel("Feature")
    plt.title(f"Top features by method agreement (top-K={top_k})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[feature_agreement_by_feature] Wrote to: {output_path}")


def _parse_args():
    workspace_root = Path(__file__).resolve().parents[3]
    default_json_dir = workspace_root / "reports" / "entropy_knn" / "score_only" / "cluster_500" / "seed_42"
    default_scores = default_json_dir / "cluster_feature_scores.parquet"
    parser = argparse.ArgumentParser(description="Generate feature agreement bar chart")
    parser.add_argument("--cluster-json-dir", type=Path, default=default_json_dir, help="Directory with cluster_*.json (preferred)")
    parser.add_argument("--scores", type=Path, default=default_scores, help="Fallback path to cluster_feature_scores.parquet or CSV")
    parser.add_argument("--output-path", type=Path, default=workspace_root / "reports" / "entropy_knn" / "analysis" / "visualizations" / "feature_agreement_bars.png")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main():
    args = _parse_args()
    scores_df = load_scores_for_analysis(cluster_json_dir=args.cluster_json_dir, scores_path=args.scores)
    generate_feature_agreement_bars(scores_df, args.output_path, top_k=args.top_k)


if __name__ == "__main__":
    main()
