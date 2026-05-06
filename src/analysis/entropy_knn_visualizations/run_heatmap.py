"""Run only the Spearman heatmap visualization for Entropy-KNN method comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.entropy_knn.visualizations import generate_spearman_heatmap


def _parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Generate only heatmap visualization")
    parser.add_argument(
        "--cluster-json-dir",
        type=Path,
        default=workspace_root / "reports" / "entropy_knn" / "score_only" / "cluster_50" / "seed_42",
        help="Directory with cluster_*.json",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=workspace_root / "reports" / "entropy_knn" / "comparison" / "visualizations" / "heatmaps" / "cluster_50_seed_42_spearman_heatmap.png",
        help="PNG output path",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate_spearman_heatmap(args.cluster_json_dir, args.output_path)


if __name__ == "__main__":
    main()
