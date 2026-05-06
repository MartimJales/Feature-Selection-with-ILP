"""Run only the pairwise agreement bar chart for Entropy-KNN method comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.entropy_knn.visualizations import generate_agreement_bars


def _parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Generate only agreement bars visualization")
    parser.add_argument(
        "--cluster-json-dir",
        type=Path,
        default=workspace_root / "reports" / "entropy_knn" / "score_only" / "cluster_50" / "seed_42",
        help="Directory with cluster_*.json",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=workspace_root / "reports" / "entropy_knn" / "comparison" / "visualizations" / "bars" / "cluster_50_seed_42_agreement_bars.png",
        help="PNG output path",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k features for overlap computation")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate_agreement_bars(args.cluster_json_dir, args.output_path, top_k=args.top_k)


if __name__ == "__main__":
    main()
