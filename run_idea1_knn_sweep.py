#!/usr/bin/env python3
"""Sweep cluster sizes for Idea1 KNN experiment and export quality-vs-size CSVs."""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.idea1.knn_experiment import Idea1KNNExperiment


def parse_int_list(csv_values: str) -> list[int]:
    values = [v.strip() for v in csv_values.split(",") if v.strip()]
    return [int(v) for v in values]


def main() -> None:
    parser = argparse.ArgumentParser(description="Idea1 KNN sweep: quality vs cluster size")

    parser.add_argument("--features-path", default="./reports/extracted_features.parquet")
    parser.add_argument("--labels-path", default="./data/training_set.csv")
    parser.add_argument("--rankings-path", default="./reports/feature_analysis/feature_rankings_all.parquet")
    parser.add_argument("--output-dir", default="./reports/idea1/sweep")

    parser.add_argument("--top-features", type=int, default=1000)
    parser.add_argument("--top-local-features", type=int, default=30)
    parser.add_argument("--min-cluster-rows", type=int, default=30)
    parser.add_argument("--scale", action="store_true", help="Apply StandardScaler before KNN")

    # Requested order: 500, 250, 150, 100, 50, 30
    parser.add_argument("--cluster-sizes", default="500,250,150,100,50,30")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds, e.g. 42,43,44")

    parser.add_argument("--base-n-clusters", type=int, default=100)
    parser.add_argument(
        "--cluster-schedule",
        choices=["inverse-size", "fixed"],
        default="inverse-size",
        help=(
            "inverse-size: increase n_clusters when cluster_size decreases "
            "(keeps total sampled rows roughly comparable); fixed: same n_clusters for all sizes"
        ),
    )

    args = parser.parse_args()

    cluster_sizes = parse_int_list(args.cluster_sizes)
    seeds = parse_int_list(args.seeds)

    if not cluster_sizes:
        raise ValueError("--cluster-sizes is empty")
    if not seeds:
        raise ValueError("--seeds is empty")

    base_size = cluster_sizes[0]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(__file__).parent / "logs" / "idea1"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "idea1_knn_sweep.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 90)
    logger.info("IDEA1 KNN SWEEP: QUALITY VS CLUSTER SIZE")
    logger.info("Cluster sizes order: %s", cluster_sizes)
    logger.info("Seeds: %s", seeds)
    logger.info("Cluster schedule: %s", args.cluster_schedule)
    logger.info("Base n_clusters: %d", args.base_n_clusters)
    logger.info("=" * 90)

    rows = []

    for size in cluster_sizes:
        if args.cluster_schedule == "inverse-size":
            n_clusters = max(1, int(round(args.base_n_clusters * (base_size / size))))
        else:
            n_clusters = args.base_n_clusters

        logger.info("\n--- cluster_size=%d | n_clusters=%d ---", size, n_clusters)

        for seed in seeds:
            run_output_dir = output_dir / f"size_{size}" / f"seed_{seed}"
            run_output_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Running size=%d seed=%d", size, seed)

            experiment = Idea1KNNExperiment(
                features_path=args.features_path,
                labels_path=args.labels_path,
                rankings_path=args.rankings_path,
                output_dir=str(run_output_dir),
            )

            t0 = time.time()
            try:
                summary = experiment.run(
                    top_features=args.top_features,
                    n_clusters=n_clusters,
                    cluster_size=size,
                    top_local_features=args.top_local_features,
                    min_cluster_rows=args.min_cluster_rows,
                    random_seed=seed,
                    scale_features=args.scale,
                )
                status = "ok"
                error_message = None
            except Exception as e:
                summary = {}
                status = "error"
                error_message = str(e)
                logger.exception("Run failed for size=%d seed=%d", size, seed)

            elapsed = time.time() - t0

            rows.append(
                {
                    "cluster_size": size,
                    "n_clusters": n_clusters,
                    "seed": seed,
                    "status": status,
                    "error_message": error_message,
                    "runtime_seconds": elapsed,
                    "n_clusters_valid": summary.get("n_clusters_valid"),
                    "mi_ratio_mean": summary.get("mi_ratio_mean"),
                    "mi_ratio_median": summary.get("mi_ratio_median"),
                    "mi_delta_mean": summary.get("mi_delta_mean"),
                    "mi_delta_median": summary.get("mi_delta_median"),
                    "mi_ratio_ge_1_5": summary.get("mi_ratio_ge_1_5"),
                    "acc_top30_mean": summary.get("acc_top30_mean"),
                    "f1_top30_mean": summary.get("f1_top30_mean"),
                    "result_csv_path": summary.get("output_csv"),
                }
            )

    df = pd.DataFrame(rows)
    out_detailed = output_dir / "knn_sweep_results.csv"
    df.to_csv(out_detailed, index=False)

    df_ok = df[df["status"] == "ok"].copy()
    if not df_ok.empty:
        agg = (
            df_ok.groupby(["cluster_size", "n_clusters"], as_index=False)
            .agg(
                runs=("seed", "count"),
                runtime_seconds_mean=("runtime_seconds", "mean"),
                n_clusters_valid_mean=("n_clusters_valid", "mean"),
                mi_ratio_mean=("mi_ratio_mean", "mean"),
                mi_delta_mean=("mi_delta_mean", "mean"),
                acc_top30_mean=("acc_top30_mean", "mean"),
                f1_top30_mean=("f1_top30_mean", "mean"),
            )
            .sort_values(by="cluster_size", ascending=False)
        )
    else:
        agg = pd.DataFrame()

    out_agg = output_dir / "knn_sweep_summary_by_size.csv"
    agg.to_csv(out_agg, index=False)

    logger.info("\n✓ Sweep detailed results: %s", out_detailed)
    logger.info("✓ Sweep summary by size: %s", out_agg)


if __name__ == "__main__":
    main()
