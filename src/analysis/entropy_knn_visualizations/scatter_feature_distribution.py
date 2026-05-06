"""Scatter: distribution of feature scores across clusters.

For a given feature, plot its aggregated score (or rank) across
all clusters to identify stable discriminators.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

workspace_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(workspace_root))

from src.entropy_knn.visualizations.common import METHODS


def _load_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _minmax(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    mn = series.min()
    mx = series.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.0, index=series.index)
    return (series - mn) / (mx - mn)


def generate_scatter_feature_distribution(scores_df: pd.DataFrame, output_path: Path, top_n_features: int = 10) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    methods_present = [m for m in METHODS if m in scores_df.columns]
    if not methods_present:
        numeric_cols = [c for c in scores_df.columns if scores_df[c].dtype.kind in "fi" and c not in ("cluster_id", "feature_rank")]
        methods_present = numeric_cols

    # compute aggregated score per (cluster, feature)
    for method in methods_present:
        scores_df[method] = pd.to_numeric(scores_df[method], errors="coerce").fillna(0.0)

    norm_cols = [f"{m}_norm" for m in methods_present]
    for method in methods_present:
        scores_df[f"{method}_norm"] = _minmax(scores_df[method])

    scores_df["aggregated_score"] = scores_df[norm_cols].sum(axis=1) / max(1, len(norm_cols))

    # find top N features by mean aggregated score
    top_features = scores_df.groupby("feature")["aggregated_score"].mean().nlargest(top_n_features).index.tolist()

    # scatter: cluster_id vs aggregated_score per feature
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.tab20(range(len(top_features)))
    for i, feat in enumerate(top_features):
        feat_data = scores_df[scores_df["feature"] == feat]
        ax.scatter(
            feat_data["cluster_id"],
            feat_data["aggregated_score"],
            label=feat,
            alpha=0.6,
            s=50,
            color=colors[i % len(colors)],
        )

    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Aggregated score")
    ax.set_title(f"Feature score distribution across clusters (top {top_n_features} features)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[scatter_feature_distribution] Wrote to: {output_path}")


def _parse_args():
    workspace_root = Path(__file__).resolve().parents[3]
    default_scores = workspace_root / "reports" / "entropy_knn" / "score_only" / "cluster_500" / "seed_42" / "cluster_feature_scores.parquet"
    parser = argparse.ArgumentParser(description="Generate feature distribution scatter plot")
    parser.add_argument("--scores", type=Path, default=default_scores, help="Path to cluster_feature_scores.parquet or CSV")
    parser.add_argument("--output-path", type=Path, default=workspace_root / "reports" / "entropy_knn" / "analysis" / "visualizations" / "scatter_feature_distribution.png")
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def main():
    args = _parse_args()
    scores_df = _load_scores(args.scores)
    generate_scatter_feature_distribution(scores_df, args.output_path, top_n_features=args.top_n)


if __name__ == "__main__":
    main()
