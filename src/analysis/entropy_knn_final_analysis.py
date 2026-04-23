"""Final aggregation and high-level analysis for Entropy KNN sweep outputs."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RunFiles:
    """File references for one run."""

    cluster_size: int
    threshold: float
    seed: int
    cluster_summary_path: Path
    selected_features_path: Path


def find_run_files(reports_dir: Path) -> list[RunFiles]:
    """Discover all available run folders and files."""
    run_files: list[RunFiles] = []
    pattern = "cluster_*/threshold_*/seed_*/cluster_summary.csv"

    for summary_path in sorted(reports_dir.glob(pattern)):
        run_dir = summary_path.parent
        selected_path = run_dir / "selected_features_by_cluster.csv"
        if not selected_path.exists():
            logger.warning("Missing selected_features file for %s", summary_path)
            continue

        cluster_size = _parse_cluster_size(summary_path)
        threshold = _parse_threshold(summary_path)
        seed = _parse_seed(summary_path)

        run_files.append(
            RunFiles(
                cluster_size=cluster_size,
                threshold=threshold,
                seed=seed,
                cluster_summary_path=summary_path,
                selected_features_path=selected_path,
            )
        )

    return run_files


def _parse_cluster_size(path: Path) -> int:
    token = next(part for part in path.parts if part.startswith("cluster_"))
    return int(token.split("_", maxsplit=1)[1])


def _parse_threshold(path: Path) -> float:
    token = next(part for part in path.parts if part.startswith("threshold_"))
    return float(token.split("_", maxsplit=1)[1])


def _parse_seed(path: Path) -> int:
    token = next(part for part in path.parts if part.startswith("seed_"))
    return int(token.split("_", maxsplit=1)[1])


def load_cluster_summary(path: Path) -> pd.DataFrame:
    """Load and normalize cluster summary columns."""
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]
    return df


def load_selected_features(path: Path) -> pd.DataFrame:
    """Load selected features per cluster."""
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]
    return df


def build_run_level_dataframe(run_files: list[RunFiles], top_k: int = 30) -> pd.DataFrame:
    """Create one row per run with compact KPIs."""
    rows: list[dict] = []

    for run in run_files:
        cluster_df = load_cluster_summary(run.cluster_summary_path)
        selected_df = load_selected_features(run.selected_features_path)

        if cluster_df.empty:
            continue

        selected_count_col = _resolve_selected_count_column(cluster_df)

        n_clusters = len(cluster_df)
        n_selected_total = int(cluster_df[selected_count_col].sum())
        n_selected_mean = float(cluster_df[selected_count_col].mean())
        n_selected_median = float(cluster_df[selected_count_col].median())

        selected_for_analysis = selected_df
        if "used_fallback" in selected_df.columns:
            selected_for_analysis = selected_df[~selected_df["used_fallback"].fillna(False)]
        selected_unique = int(selected_for_analysis["feature"].nunique()) if not selected_for_analysis.empty else 0

        threshold_pass_mean = float(cluster_df["threshold_pass_count"].mean()) if "threshold_pass_count" in cluster_df.columns else float("nan")
        fallback_rate = float(cluster_df["used_fallback"].mean()) if "used_fallback" in cluster_df.columns else float("nan")
        threshold_mode_rate = float((cluster_df["selection_mode"] == "threshold_top_k").mean()) if "selection_mode" in cluster_df.columns else float("nan")

        mean_reduction_ratio = float(cluster_df["mean_reduction_ratio"].mean())
        weighted_reduction_ratio = float(
            np.average(
                cluster_df["mean_reduction_ratio"],
                weights=cluster_df["n_samples"].clip(lower=1),
            )
        )
        mean_conditional_entropy = float(cluster_df["mean_conditional_entropy"].mean())

        pct_clusters_top_k = float((cluster_df[selected_count_col] >= top_k).mean())
        pct_clusters_under_5 = float((cluster_df[selected_count_col] < 5).mean())
        pct_clusters_zero_entropy = float((cluster_df["base_entropy"] <= 1e-12).mean())

        rows.append(
            {
                "cluster_size": run.cluster_size,
                "threshold": run.threshold,
                "seed": run.seed,
                "n_clusters": n_clusters,
                "n_selected_total": n_selected_total,
                "n_selected_mean": n_selected_mean,
                "n_selected_median": n_selected_median,
                "selected_unique": selected_unique,
                "mean_reduction_ratio": mean_reduction_ratio,
                "weighted_reduction_ratio": weighted_reduction_ratio,
                "mean_conditional_entropy": mean_conditional_entropy,
                "threshold_pass_mean": threshold_pass_mean,
                "fallback_rate": fallback_rate,
                "threshold_mode_rate": threshold_mode_rate,
                "pct_clusters_top_k": pct_clusters_top_k,
                "pct_clusters_under_5": pct_clusters_under_5,
                "pct_clusters_zero_entropy": pct_clusters_zero_entropy,
                "cluster_summary_path": str(run.cluster_summary_path),
                "selected_features_path": str(run.selected_features_path),
                "selected_count_column": selected_count_col,
            }
        )

    return pd.DataFrame(rows)


def _resolve_selected_count_column(cluster_df: pd.DataFrame) -> str:
    if "n_selected_features_analysis" in cluster_df.columns:
        return "n_selected_features_analysis"
    if "n_selected_features" in cluster_df.columns:
        return "n_selected_features"
    raise KeyError("Expected selected-feature count column not found in cluster summary")


def build_cluster_level_dataframe(run_files: list[RunFiles]) -> pd.DataFrame:
    """Create a unified cluster-level dataframe across all runs."""
    rows: list[pd.DataFrame] = []
    for run in run_files:
        cluster_df = load_cluster_summary(run.cluster_summary_path)
        if cluster_df.empty:
            continue

        cluster_df = cluster_df.copy()
        cluster_df["cluster_size"] = run.cluster_size
        cluster_df["threshold"] = run.threshold
        cluster_df["seed"] = run.seed
        rows.append(cluster_df)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def aggregate_by_configuration(run_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate run-level KPIs by (cluster_size, threshold)."""
    if run_df.empty:
        return pd.DataFrame()

    grouped = (
        run_df.groupby(["cluster_size", "threshold"], as_index=False)
        .agg(
            runs=("seed", "count"),
            n_clusters_mean=("n_clusters", "mean"),
            n_selected_mean=("n_selected_mean", "mean"),
            n_selected_median=("n_selected_median", "mean"),
            selected_unique_mean=("selected_unique", "mean"),
            mean_reduction_ratio=("mean_reduction_ratio", "mean"),
            weighted_reduction_ratio=("weighted_reduction_ratio", "mean"),
            mean_conditional_entropy=("mean_conditional_entropy", "mean"),
            threshold_pass_mean=("threshold_pass_mean", "mean"),
            fallback_rate=("fallback_rate", "mean"),
            threshold_mode_rate=("threshold_mode_rate", "mean"),
            pct_clusters_top_k=("pct_clusters_top_k", "mean"),
            pct_clusters_under_5=("pct_clusters_under_5", "mean"),
            pct_clusters_zero_entropy=("pct_clusters_zero_entropy", "mean"),
        )
        .sort_values(["cluster_size", "threshold"], ascending=[False, True])
    )

    grouped["compactness_score"] = 1.0 - (grouped["n_selected_mean"] / 30.0)
    grouped["compactness_score"] = grouped["compactness_score"].clip(lower=0.0, upper=1.0)
    grouped["quality_score"] = grouped["weighted_reduction_ratio"]
    grouped["tradeoff_score"] = 0.6 * grouped["quality_score"] + 0.4 * grouped["compactness_score"]
    grouped = grouped.sort_values("tradeoff_score", ascending=False)

    return grouped


def build_feature_stability(run_files: list[RunFiles]) -> pd.DataFrame:
    """Estimate how stable feature selection is across clusters and configs."""
    rows: list[dict] = []

    for run in run_files:
        selected_df = load_selected_features(run.selected_features_path)
        cluster_df = load_cluster_summary(run.cluster_summary_path)
        if selected_df.empty or cluster_df.empty:
            continue

        n_clusters = int(cluster_df["cluster_id"].nunique())
        freq = selected_df.groupby("feature", as_index=False)["cluster_id"].nunique()
        freq = freq.rename(columns={"cluster_id": "clusters_with_feature"})
        freq["cluster_coverage"] = freq["clusters_with_feature"] / max(n_clusters, 1)
        freq["cluster_size"] = run.cluster_size
        freq["threshold"] = run.threshold
        freq["seed"] = run.seed
        rows.extend(freq.to_dict(orient="records"))

    if not rows:
        return pd.DataFrame()

    stability = pd.DataFrame(rows)
    stability_summary = (
        stability.groupby("feature", as_index=False)
        .agg(
            mean_cluster_coverage=("cluster_coverage", "mean"),
            max_cluster_coverage=("cluster_coverage", "max"),
            appearances=("cluster_size", "count"),
        )
        .sort_values(["mean_cluster_coverage", "appearances"], ascending=[False, False])
    )
    return stability_summary


def summarize_coverage(run_df: pd.DataFrame) -> pd.DataFrame:
    """Report which (cluster_size, threshold, seed) combinations exist."""
    if run_df.empty:
        return pd.DataFrame()

    available = run_df[["cluster_size", "threshold", "seed"]].drop_duplicates().copy()
    cluster_sizes = sorted(available["cluster_size"].unique(), reverse=True)
    thresholds = sorted(available["threshold"].unique())
    seeds = sorted(available["seed"].unique())

    full_index = pd.MultiIndex.from_product(
        [cluster_sizes, thresholds, seeds],
        names=["cluster_size", "threshold", "seed"],
    )
    expected = full_index.to_frame(index=False)
    expected["present"] = expected.merge(
        available.assign(present=True),
        on=["cluster_size", "threshold", "seed"],
        how="left",
    )["present"].fillna(False)

    return expected


def write_markdown_summary(
    output_file: Path,
    run_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    stability_df: pd.DataFrame,
) -> None:
    """Write a concise executive report in Markdown."""
    lines: list[str] = []
    lines.append("# Entropy KNN — Final Aggregated Analysis")
    lines.append("")

    if run_df.empty:
        lines.append("No runs were found.")
        output_file.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("## 1. Execution Coverage")
    lines.append("")
    lines.append(f"- Runs discovered: **{len(run_df)}**")
    lines.append(f"- Cluster sizes discovered: **{sorted(run_df['cluster_size'].unique(), reverse=True)}**")
    lines.append(f"- Thresholds discovered: **{sorted(run_df['threshold'].unique())}**")
    lines.append(f"- Seeds discovered: **{sorted(run_df['seed'].unique())}**")

    if not coverage_df.empty:
        missing = coverage_df[~coverage_df["present"]]
        lines.append(f"- Missing configurations: **{len(missing)}**")

    lines.append("")
    lines.append("## 2. Key Observations")
    lines.append("")
    lines.append(
        "- `weighted_reduction_ratio` captures the average local entropy-reduction quality, weighted by cluster size."
    )
    lines.append(
        "- `n_selected_mean` captures feature compactness; lower values indicate more aggressive local filtering."
    )
    lines.append(
        "- `tradeoff_score` combines quality and compactness for a practical decision criterion."
    )

    if not stability_df.empty:
        lines.append("")
        lines.append("## 3. Most Stable Features")
        lines.append("")
        top_stable = stability_df.head(10)
        for row in top_stable.itertuples(index=False):
            lines.append(
                f"- {row.feature}: mean_coverage={row.mean_cluster_coverage:.4f}, "
                f"max_coverage={row.max_cluster_coverage:.4f}, appearances={int(row.appearances)}"
            )

    output_file.write_text("\n".join(lines), encoding="utf-8")


def save_outputs(
    output_dir: Path,
    run_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
) -> None:
    """Persist all aggregated artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    run_df.to_csv(output_dir / "run_level_summary.csv", index=False)
    stability_df.to_csv(output_dir / "feature_stability_summary.csv", index=False)
    coverage_df.to_csv(output_dir / "coverage_matrix.csv", index=False)

    write_markdown_summary(
        output_file=output_dir / "executive_summary.md",
        run_df=run_df,
        coverage_df=coverage_df,
        stability_df=stability_df,
    )


def run_final_analysis(
    reports_dir: str = "./reports/entropy_knn",
    output_dir: str = "./reports/entropy_knn/final_analysis",
) -> dict[str, int]:
    """Run full final aggregation analysis."""
    reports_path = Path(reports_dir)
    output_path = Path(output_dir)

    if not reports_path.exists():
        raise FileNotFoundError(f"Reports directory not found: {reports_path}")

    run_files = find_run_files(reports_path)
    logger.info("Discovered %d entropy_knn run folders", len(run_files))

    run_df = build_run_level_dataframe(run_files)
    stability_df = build_feature_stability(run_files)
    coverage_df = summarize_coverage(run_df)

    save_outputs(
        output_dir=output_path,
        run_df=run_df,
        stability_df=stability_df,
        coverage_df=coverage_df,
    )

    return {
        "runs": int(len(run_df)),
        "stable_features": int(len(stability_df)),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Final aggregated analysis for entropy_knn reports")
    parser.add_argument("--reports-dir", default="./reports/entropy_knn")
    parser.add_argument("--output-dir", default="./reports/entropy_knn/final_analysis")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    args = _build_parser().parse_args()
    stats = run_final_analysis(reports_dir=args.reports_dir, output_dir=args.output_dir)

    print("\n" + "=" * 80)
    print("ENTROPY KNN — FINAL ANALYSIS")
    print("=" * 80)
    print(f"Runs aggregated: {stats['runs']}")
    print(f"Stable features scored: {stats['stable_features']}")
    print("=" * 80)
