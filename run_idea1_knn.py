#!/usr/bin/env python3
"""Quick experiment runner for Idea1 (KNN local IG analysis)."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.idea1.knn_experiment import Idea1KNNExperiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Idea1 KNN local-feature experiment")
    parser.add_argument("--features-path", default="./reports/extracted_features.parquet")
    parser.add_argument("--labels-path", default="./data/training_set.csv")
    parser.add_argument("--rankings-path", default="./reports/feature_analysis/feature_rankings_all.parquet")
    parser.add_argument("--output-dir", default="./reports/idea1")

    parser.add_argument("--top-features", type=int, default=2000)
    parser.add_argument("--n-clusters", type=int, default=100)
    parser.add_argument("--cluster-size", type=int, default=500)
    parser.add_argument("--top-local-features", type=int, default=30)
    parser.add_argument("--min-cluster-rows", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", action="store_true", help="Apply StandardScaler before KNN")

    args = parser.parse_args()

    log_dir = Path(__file__).parent / "logs" / "idea1"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "idea1_knn.log"),
            logging.StreamHandler(),
        ],
    )

    experiment = Idea1KNNExperiment(
        features_path=args.features_path,
        labels_path=args.labels_path,
        rankings_path=args.rankings_path,
        output_dir=args.output_dir,
    )

    summary = experiment.run(
        top_features=args.top_features,
        n_clusters=args.n_clusters,
        cluster_size=args.cluster_size,
        top_local_features=args.top_local_features,
        min_cluster_rows=args.min_cluster_rows,
        random_seed=args.seed,
        scale_features=args.scale,
    )

    logging.getLogger(__name__).info("Done. Summary: %s", summary)


if __name__ == "__main__":
    main()
