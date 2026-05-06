"""Run only the Venn visualization for one selected cluster JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.entropy_knn.visualizations import generate_venn_grid


def _parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[3]
    default_cluster_dir = workspace_root / "reports" / "entropy_knn" / "score_only" / "cluster_50" / "seed_42"
    default_cluster_json = default_cluster_dir / "cluster_0.json"

    parser = argparse.ArgumentParser(description="Generate only Venn visualization for one cluster JSON")
    parser.add_argument(
        "--cluster-json",
        type=Path,
        default=default_cluster_json,
        help="Path to a cluster_*.json file",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=workspace_root / "reports" / "entropy_knn" / "comparison" / "visualizations" / "venn" / "cluster_0_top5_venn.png",
        help="PNG output path",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k features used in overlap")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate_venn_grid(args.cluster_json, args.output_path, top_k=args.top_k)


if __name__ == "__main__":
    main()
