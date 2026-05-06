"""Runner for top-feature analysis: orchestrate all modules.

Runs in sequence:
1. top_feature_consensus.py — generate per-cluster candidates for ILP
2. feature_agreement_by_feature.py — bar chart of feature agreement
3. feature_pair_heatmap.py — feature co-occurrence heatmap
4. scatter_feature_distribution.py — feature distribution across clusters
5. venn_by_methods.py — method overlap Venn diagrams
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ensure repo root on sys.path
workspace_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(workspace_root))

from src.analysis.entropy_knn_visualizations.top_feature_consensus import compute_consensus, _load_scores
from src.analysis.entropy_knn_visualizations.feature_agreement_by_feature import generate_feature_agreement_bars
from src.analysis.entropy_knn_visualizations.feature_pair_heatmap import generate_feature_pair_heatmap
from src.analysis.entropy_knn_visualizations.scatter_feature_distribution import generate_scatter_feature_distribution
from src.analysis.entropy_knn_visualizations.venn_by_methods import generate_venn_by_methods


def _parse_args():
    workspace_root = Path(__file__).resolve().parents[3]
    default_scores = workspace_root / "reports" / "entropy_knn" / "score_only" / "cluster_500" / "seed_42" / "cluster_feature_scores.parquet"
    parser = argparse.ArgumentParser(description="Run full top-feature analysis pipeline for ILP")
    parser.add_argument("--scores", type=Path, default=default_scores, help="Path to cluster_feature_scores.parquet or CSV")
    parser.add_argument("--output-dir", type=Path, default=workspace_root / "reports" / "entropy_knn" / "analysis", help="Base output dir")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K per method for consensus")
    parser.add_argument("--top-n", type=int, default=50, help="Number of candidates to export per cluster")
    parser.add_argument("--top-m", type=int, default=30, help="Number of features to include in pair heatmap")
    parser.add_argument("--top-features-scatter", type=int, default=10, help="Number of top features for scatter plot")
    parser.add_argument("--cluster-id-venn", type=int, default=None, help="Cluster ID for Venn (default: first)")
    parser.add_argument("--normalize", choices=["minmax", "z", "none"], default="minmax", help="Normalization for consensus")
    parser.add_argument("--skip-visualizations", action="store_true", help="Skip downstream visualizations, only compute consensus")
    return parser.parse_args()


def main():
    args = _parse_args()
    vis_dir = args.output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run_top_feature_analysis] Starting analysis pipeline...")
    print(f"[run_top_feature_analysis] Scores: {args.scores}")
    print(f"[run_top_feature_analysis] Output dir: {args.output_dir}")

    # Step 1: Consensus
    print(f"[run_top_feature_analysis] Step 1/5: Computing consensus candidates...")
    scores_df = _load_scores(args.scores)
    compute_consensus(scores_df=scores_df, output_dir=args.output_dir, top_k=args.top_k, top_n=args.top_n, normalize=args.normalize)

    if args.skip_visualizations:
        print(f"[run_top_feature_analysis] Done (skipped visualizations).")
        return

    # Step 2: Feature agreement bars
    print(f"[run_top_feature_analysis] Step 2/5: Generating feature agreement bars...")
    generate_feature_agreement_bars(scores_df, output_path=vis_dir / "feature_agreement_bars.png", top_k=args.top_k)

    # Step 3: Feature pair heatmap
    print(f"[run_top_feature_analysis] Step 3/5: Generating feature-pair heatmap...")
    generate_feature_pair_heatmap(scores_df, output_path=vis_dir / "feature_pair_heatmap.png", top_k=args.top_k, top_m=args.top_m)

    # Step 4: Scatter distribution
    print(f"[run_top_feature_analysis] Step 4/5: Generating scatter distribution...")
    generate_scatter_feature_distribution(scores_df, output_path=vis_dir / "scatter_feature_distribution.png", top_n_features=args.top_features_scatter)

    # Step 5: Venn diagrams
    print(f"[run_top_feature_analysis] Step 5/5: Generating Venn diagrams...")
    generate_venn_by_methods(scores_df, output_path=vis_dir / "venn_by_methods.png", cluster_id=args.cluster_id_venn, top_k=args.top_k)

    print(f"[run_top_feature_analysis] Done. All outputs in: {args.output_dir}")
    print(f"[run_top_feature_analysis] Candidates CSV: {args.output_dir}/top_feature_candidates.csv")
    print(f"[run_top_feature_analysis] Visualizations: {vis_dir}/")


if __name__ == "__main__":
    main()
