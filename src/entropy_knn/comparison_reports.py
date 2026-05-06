"""Generate comparison reports from per-cluster JSON files."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


class ComparisonReportGenerator:
    """Generate comparison summaries and visualizations from filter method scores."""

    @staticmethod
    def generate_comparison_summary(
        cluster_json_dir: Path,
        output_dir: Path,
    ) -> pd.DataFrame:
        """
        Generate comparison_summary.csv from per-cluster JSON files.

        One row per cluster with top-1 feature by each method.

        Args:
            cluster_json_dir: Directory containing cluster_*.json files
            output_dir: Output directory for CSVs

        Returns:
            DataFrame with comparison summary
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = []

        json_files = sorted(cluster_json_dir.glob("cluster_*.json"))
        logger.info(f"Processing {len(json_files)} cluster JSON files from {cluster_json_dir}")

        for json_file in json_files:
            try:
                with open(json_file, encoding="utf-8") as f:
                    cluster_data = json.load(f)

                cluster_id = cluster_data["cluster_id"]
                base_entropy = cluster_data["base_entropy"]
                top_by_method = cluster_data["top_features_by_method"]

                row = {
                    "cluster_id": cluster_id,
                    "base_entropy": base_entropy,
                    "n_samples": cluster_data["n_samples"],
                    "class_0": cluster_data["class_0"],
                    "class_1": cluster_data["class_1"],
                }

                # Add top-1 by each method
                for method, top_feature_info in top_by_method.items():
                    row[f"top_1_feature_{method}"] = top_feature_info["feature"]
                    row[f"top_1_value_{method}"] = top_feature_info["value"]

                rows.append(row)
            except Exception as e:
                logger.warning(f"Failed to process {json_file}: {e}")

        summary_df = pd.DataFrame(rows)
        output_path = output_dir / "comparison_summary.csv"
        summary_df.to_csv(output_path, index=False)
        logger.info(f"Wrote comparison summary to {output_path}")

        return summary_df

    @staticmethod
    def generate_method_agreement_summary(
        comparison_summary_df: pd.DataFrame,
        output_dir: Path,
    ) -> pd.DataFrame:
        """
        Generate method_agreement_summary.csv with concordance metrics.

        One row with global statistics about agreement between methods.

        Args:
            comparison_summary_df: Output from generate_comparison_summary
            output_dir: Output directory

        Returns:
            DataFrame with agreement metrics
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract ranks for each method
        methods = ["entropy_reduction_ratio", "mutual_information", "chi2_stat", "f_stat", "pearson_r"]

        # Build rank columns from top-1 values
        # This is a simple measure: how many clusters agree on top-1
        agreement_stats = {}

        for method in methods:
            top_feature_col = f"top_1_feature_{method}"
            if top_feature_col in comparison_summary_df.columns:
                agreement_stats[f"n_clusters_{method}"] = len(comparison_summary_df)

        # Compare top-1 agreement between pairs of methods
        for i, method1 in enumerate(methods):
            for method2 in methods[i + 1 :]:
                col1 = f"top_1_feature_{method1}"
                col2 = f"top_1_feature_{method2}"
                if col1 in comparison_summary_df.columns and col2 in comparison_summary_df.columns:
                    agreement = (comparison_summary_df[col1] == comparison_summary_df[col2]).sum()
                    n_clusters = len(comparison_summary_df)
                    pct = 100.0 * agreement / n_clusters if n_clusters > 0 else 0.0
                    agreement_stats[f"agreement_{method1}_vs_{method2}_pct"] = pct

        summary_row = agreement_stats
        summary_df = pd.DataFrame([summary_row])

        output_path = output_dir / "method_agreement_summary.csv"
        summary_df.to_csv(output_path, index=False)
        logger.info(f"Wrote method agreement summary to {output_path}")

        return summary_df

    @staticmethod
    def generate_per_cluster_size_report(
        comparison_dir: Path,
        cluster_size: int,
    ) -> None:
        """
        Generate consolidated reports for a given cluster size.

        Args:
            comparison_dir: Base comparison directory
            cluster_size: The cluster size (e.g., 50)
        """
        cluster_size_dir = comparison_dir / f"cluster_size_{cluster_size}"
        if not cluster_size_dir.exists():
            logger.warning(f"Cluster size dir not found: {cluster_size_dir}")
            return

        # For each seed, generate summary
        for seed_dir in sorted(cluster_size_dir.glob("seed_*")):
            json_dir = seed_dir
            summary_df = ComparisonReportGenerator.generate_comparison_summary(json_dir, seed_dir)
            agreement_df = ComparisonReportGenerator.generate_method_agreement_summary(summary_df, seed_dir)
            logger.info(f"Generated reports for {seed_dir.name}: {len(summary_df)} clusters")
