"""Run all Entropy-KNN visualizations from existing cluster JSON scores."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.generate_entropy_knn_visualizations import generate_visualizations


def _parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Generate all visualizations from precomputed cluster JSON files")
    parser.add_argument(
        "--cluster-json-dir",
        type=Path,
        default=workspace_root / "reports" / "entropy_knn" / "score_only" / "cluster_50" / "seed_42",
        help="Directory with cluster_*.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace_root / "reports" / "entropy_knn" / "comparison" / "visualizations",
        help="Base output directory",
    )
    parser.add_argument(
        "--venn-cluster-json",
        type=Path,
        default=None,
        help="Optional specific cluster_*.json for venn",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate_visualizations(
        cluster_json_dir=args.cluster_json_dir,
        output_dir=args.output_dir,
        venn_cluster_json=args.venn_cluster_json,
    )


if __name__ == "__main__":
    main()
