#!/usr/bin/env python3
"""Sweep runner for the Entropy KNN pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.entropy_knn.pipeline import EntropyKNNPipeline


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Entropy KNN sweep runner")
    parser.add_argument(
        "--mode",
        choices=["score-only", "threshold-sweep"],
        default="score-only",
        help="score-only: save per-feature entropy scores per cluster; threshold-sweep: run selection sweep",
    )
    parser.add_argument("--features-path", default="./reports/extracted_features.parquet")
    parser.add_argument("--labels-path", default="./data/training_set.csv")
    parser.add_argument("--rankings-path", default="./reports/feature_analysis/feature_rankings_all.parquet")
    parser.add_argument("--output-dir", default="./reports/entropy_knn")
    parser.add_argument("--cluster-sizes", default="500,250,150,100,50,30")
    parser.add_argument("--thresholds", default="0.5,0.6")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--top-features-global", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--base-n-clusters", type=int, default=100)
    parser.add_argument("--cluster-schedule", choices=["inverse-size", "fixed"], default="inverse-size")
    parser.add_argument("--scale", action="store_true")
    args = parser.parse_args()

    log_dir = Path(__file__).parent / "logs" / "entropy_knn"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "entropy_knn.log"),
            logging.StreamHandler(),
        ],
    )

    pipeline = EntropyKNNPipeline(
        features_path=args.features_path,
        labels_path=args.labels_path,
        rankings_path=args.rankings_path,
        output_dir=args.output_dir,
    )

    cluster_sizes = _parse_csv_ints(args.cluster_sizes)
    seeds = _parse_csv_ints(args.seeds)

    if args.mode == "score-only":
        score_df = pipeline.run_score_sweep(
            cluster_sizes=cluster_sizes,
            top_features_global=args.top_features_global,
            seeds=seeds,
            scale_features=args.scale,
            base_n_clusters=args.base_n_clusters,
            cluster_schedule=args.cluster_schedule,
        )
        logging.getLogger(__name__).info("Score-only sweep completed: %d runs", len(score_df))
    else:
        run_df, cluster_df = pipeline.run_sweep(
            cluster_sizes=cluster_sizes,
            thresholds=_parse_csv_floats(args.thresholds),
            top_features_global=args.top_features_global,
            top_k=args.top_k,
            seeds=seeds,
            scale_features=args.scale,
            base_n_clusters=args.base_n_clusters,
            cluster_schedule=args.cluster_schedule,
        )

        logging.getLogger(__name__).info(
            "Sweep completed: %d runs, %d clusters", len(run_df), len(cluster_df)
        )


if __name__ == "__main__":
    main()
