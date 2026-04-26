"""Main orchestration for the Entropy KNN pipeline."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .clustering import EntropyCluster, EntropyKNNClusterer
from .data_loader import EntropyKNNDataBundle, EntropyKNNDataLoader
from .report_io import write_tabular_report
from .selection import ClusterSelectionSummary, EntropyFeatureSelector

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EntropyKNNRunResult:
    """Result of one pipeline configuration."""

    cluster_size: int
    threshold: float
    top_features_global: int
    top_k: int
    seed: int
    n_clusters_requested: int
    n_clusters_valid: int
    n_selected_features_total: int
    n_selected_features_total_raw: int
    selected_features_unique: int
    selected_features_unique_raw: int
    mean_cluster_size: float
    mean_selected_per_cluster: float
    mean_selected_per_cluster_raw: float
    mean_entropy_reduction_ratio: float
    mean_conditional_entropy: float
    runtime_seconds: float
    status: str
    error_message: str | None
    summary_csv: str | None
    selected_features_csv: str | None


class EntropyKNNPipeline:
    """Entropy-driven KNN feature selection pipeline."""

    def __init__(
        self,
        features_path: str = "./reports/extracted_features.parquet",
        labels_path: str = "./data/training_set.csv",
        rankings_path: str = "./reports/feature_analysis/feature_rankings_all.parquet",
        output_dir: str = "./reports/entropy_knn",
    ) -> None:
        self.loader = EntropyKNNDataLoader(
            features_path=features_path,
            labels_path=labels_path,
            rankings_path=rankings_path,
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bundle: EntropyKNNDataBundle | None = None

    def run_sweep(
        self,
        cluster_sizes: list[int],
        thresholds: list[float],
        top_features_global: int = 1000,
        top_k: int = 30,
        seeds: list[int] | None = None,
        scale_features: bool = False,
        base_n_clusters: int = 100,
        cluster_schedule: str = "inverse-size",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        seeds = seeds or [42]
        run_rows: list[dict] = []
        cluster_rows: list[dict] = []
        total_runs = len(cluster_sizes) * len(thresholds) * len(seeds)
        run_index = 0

        logger.info("Starting Entropy KNN sweep")
        logger.info(
            "Runs=%d | cluster_sizes=%s | thresholds=%s | seeds=%s",
            total_runs,
            cluster_sizes,
            thresholds,
            seeds,
        )

        for cluster_size in cluster_sizes:
            n_clusters = self._resolve_n_clusters(cluster_size, cluster_sizes[0], base_n_clusters, cluster_schedule)
            for threshold in thresholds:
                for seed in seeds:
                    run_index += 1
                    logger.info(
                        "[%d/%d] Running config: cluster_size=%d | n_clusters=%d | threshold=%.3f | seed=%d",
                        run_index,
                        total_runs,
                        cluster_size,
                        n_clusters,
                        threshold,
                        seed,
                    )
                    result, cluster_details = self._run_single_configuration(
                        cluster_size=cluster_size,
                        threshold=threshold,
                        top_features_global=top_features_global,
                        top_k=top_k,
                        seed=seed,
                        n_clusters=n_clusters,
                        scale_features=scale_features,
                    )
                    run_rows.append(asdict(result))
                    cluster_rows.extend(cluster_details)
                    logger.info(
                        "[%d/%d] Finished: status=%s | runtime=%.1fs | valid_clusters=%d | selected_unique=%d",
                        run_index,
                        total_runs,
                        result.status,
                        result.runtime_seconds,
                        result.n_clusters_valid,
                        result.selected_features_unique,
                    )

        run_df = pd.DataFrame(run_rows)
        cluster_df = pd.DataFrame(cluster_rows)

        self._save_sweep_outputs(run_df, cluster_df)
        return run_df, cluster_df

    def _run_single_configuration(
        self,
        cluster_size: int,
        threshold: float,
        top_features_global: int,
        top_k: int,
        seed: int,
        n_clusters: int,
        scale_features: bool,
    ) -> tuple[EntropyKNNRunResult, list[dict]]:
        start_time = time.time()
        run_dir = self.output_dir / f"cluster_{cluster_size}" / f"threshold_{self._format_threshold(threshold)}" / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            bundle = self._load_bundle()
            X_global = self._select_global_features(bundle, top_features_global)
            clusterer = EntropyKNNClusterer(
                cluster_size=cluster_size,
                n_clusters=n_clusters,
                scale_features=scale_features,
                random_seed=seed,
            )
            clusters = clusterer.cluster(X_global)
            selector = EntropyFeatureSelector()

            cluster_summaries = self._score_clusters(
                X_global=X_global,
                y=bundle.y,
                clusters=clusters,
                selector=selector,
                top_k=top_k,
                threshold=threshold,
                progress_every=10,
            )

            summary_csv, selected_csv = self._save_run_outputs(run_dir, cluster_summaries)
            result = self._build_run_result(
                cluster_size=cluster_size,
                threshold=threshold,
                top_features_global=top_features_global,
                top_k=top_k,
                seed=seed,
                n_clusters=n_clusters,
                cluster_summaries=cluster_summaries,
                runtime_seconds=time.time() - start_time,
                status="ok",
                error_message=None,
                summary_csv=summary_csv,
                selected_features_csv=selected_csv,
            )
            cluster_details = self._build_cluster_rows(cluster_summaries, result)
            return result, cluster_details
        except Exception as exc:  # pragma: no cover - surfaced in output CSV
            logger.exception("Entropy KNN configuration failed")
            result = EntropyKNNRunResult(
                cluster_size=cluster_size,
                threshold=threshold,
                top_features_global=top_features_global,
                top_k=top_k,
                seed=seed,
                n_clusters_requested=n_clusters,
                n_clusters_valid=0,
                n_selected_features_total=0,
                n_selected_features_total_raw=0,
                selected_features_unique=0,
                selected_features_unique_raw=0,
                mean_cluster_size=0.0,
                mean_selected_per_cluster=0.0,
                mean_selected_per_cluster_raw=0.0,
                mean_entropy_reduction_ratio=0.0,
                mean_conditional_entropy=0.0,
                runtime_seconds=time.time() - start_time,
                status="error",
                error_message=str(exc),
                summary_csv=None,
                selected_features_csv=None,
            )
            return result, []

    def run_score_sweep(
        self,
        cluster_sizes: list[int],
        top_features_global: int = 1000,
        seeds: list[int] | None = None,
        scale_features: bool = False,
        base_n_clusters: int = 100,
        cluster_schedule: str = "inverse-size",
    ) -> pd.DataFrame:
        """Run scoring-only sweep (no threshold, no feature filtering)."""
        seeds = seeds or [42]
        total_runs = len(cluster_sizes) * len(seeds)
        run_index = 0
        rows: list[dict] = []

        logger.info("Starting Entropy KNN score-only sweep")
        logger.info("Runs=%d | cluster_sizes=%s | seeds=%s", total_runs, cluster_sizes, seeds)

        for cluster_size in cluster_sizes:
            n_clusters = self._resolve_n_clusters(cluster_size, cluster_sizes[0], base_n_clusters, cluster_schedule)
            for seed in seeds:
                run_index += 1
                logger.info(
                    "[%d/%d] Score-only config: cluster_size=%d | n_clusters=%d | seed=%d",
                    run_index,
                    total_runs,
                    cluster_size,
                    n_clusters,
                    seed,
                )
                row = self._run_score_only_configuration(
                    cluster_size=cluster_size,
                    top_features_global=top_features_global,
                    seed=seed,
                    n_clusters=n_clusters,
                    scale_features=scale_features,
                )
                rows.append(row)

        result_df = pd.DataFrame(rows)
        result_df.to_csv(self.output_dir / "score_sweep_results.csv", index=False)
        return result_df

    def _run_score_only_configuration(
        self,
        cluster_size: int,
        top_features_global: int,
        seed: int,
        n_clusters: int,
        scale_features: bool,
    ) -> dict:
        start_time = time.time()
        run_dir = self.output_dir / "score_only" / f"cluster_{cluster_size}" / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            bundle = self._load_bundle()
            X_global = self._select_global_features(bundle, top_features_global)
            clusterer = EntropyKNNClusterer(
                cluster_size=cluster_size,
                n_clusters=n_clusters,
                scale_features=scale_features,
                random_seed=seed,
            )
            clusters = clusterer.cluster(X_global)
            selector = EntropyFeatureSelector()

            feature_rows: list[pd.DataFrame] = []
            cluster_rows: list[dict] = []
            total_clusters = len(clusters)

            for index, cluster in enumerate(clusters, start=1):
                X_cluster = X_global.iloc[cluster.row_indices].reset_index(drop=True)
                y_cluster = bundle.y.iloc[cluster.row_indices].reset_index(drop=True)
                if X_cluster.empty or y_cluster.empty:
                    continue

                scores = selector.score_cluster(X_cluster, y_cluster)
                if scores.empty:
                    continue

                class_counts = y_cluster.value_counts().to_dict()
                scores = scores.copy()
                scores.insert(0, "feature_rank", range(1, len(scores) + 1))
                scores.insert(0, "cluster_id", cluster.cluster_id)
                scores.insert(1, "anchor_index", cluster.anchor_index)
                scores.insert(2, "n_samples", cluster.n_samples)
                scores.insert(3, "class_0", int(class_counts.get(0, 0)))
                scores.insert(4, "class_1", int(class_counts.get(1, 0)))
                feature_rows.append(scores)

                cluster_rows.append(
                    {
                        "cluster_id": cluster.cluster_id,
                        "anchor_index": cluster.anchor_index,
                        "n_samples": cluster.n_samples,
                        "class_0": int(class_counts.get(0, 0)),
                        "class_1": int(class_counts.get(1, 0)),
                        "base_entropy": float(scores["base_entropy"].iloc[0]),
                        "n_features_scored": int(len(scores)),
                        "max_reduction_ratio": float(scores["entropy_reduction_ratio"].max()),
                        "mean_reduction_ratio": float(scores["entropy_reduction_ratio"].mean()),
                        "mean_conditional_entropy": float(scores["conditional_entropy"].mean()),
                    }
                )

                if index == 1 or index % 10 == 0 or index == total_clusters:
                    logger.info(
                        "  Cluster progress: %d/%d | cluster_id=%d | rows=%d | features_scored=%d",
                        index,
                        total_clusters,
                        cluster.cluster_id,
                        cluster.n_samples,
                        len(scores),
                    )

            feature_scores_df = pd.concat(feature_rows, ignore_index=True) if feature_rows else pd.DataFrame()
            cluster_summary_df = pd.DataFrame(cluster_rows)

            feature_scores_path = run_dir / "cluster_feature_scores.parquet"
            cluster_summary_path = run_dir / "cluster_entropy_summary.parquet"
            write_tabular_report(feature_scores_df, feature_scores_path)
            write_tabular_report(cluster_summary_df, cluster_summary_path)

            return {
                "cluster_size": cluster_size,
                "seed": seed,
                "top_features_global": top_features_global,
                "n_clusters_requested": n_clusters,
                "n_clusters_valid": int(len(cluster_summary_df)),
                "runtime_seconds": float(time.time() - start_time),
                "status": "ok",
                "error_message": None,
                "cluster_entropy_summary_parquet": str(cluster_summary_path),
                "cluster_feature_scores_parquet": str(feature_scores_path),
            }
        except Exception as exc:  # pragma: no cover
            logger.exception("Entropy KNN score-only configuration failed")
            return {
                "cluster_size": cluster_size,
                "seed": seed,
                "top_features_global": top_features_global,
                "n_clusters_requested": n_clusters,
                "n_clusters_valid": 0,
                "runtime_seconds": float(time.time() - start_time),
                "status": "error",
                "error_message": str(exc),
                "cluster_entropy_summary_parquet": None,
                "cluster_feature_scores_parquet": None,
            }

    def _load_bundle(self) -> EntropyKNNDataBundle:
        if self.bundle is None:
            self.bundle = self.loader.load()
        return self.bundle

    @staticmethod
    def _select_global_features(bundle: EntropyKNNDataBundle, top_features_global: int) -> pd.DataFrame:
        selected = bundle.ranked_features[:top_features_global]
        if not selected:
            raise ValueError("No ranked features available in the extracted feature matrix")
        return bundle.X[selected].copy()

    @staticmethod
    def _score_clusters(
        X_global: pd.DataFrame,
        y: pd.Series,
        clusters: list[EntropyCluster],
        selector: EntropyFeatureSelector,
        top_k: int,
        threshold: float,
        progress_every: int = 10,
    ) -> list[ClusterSelectionSummary]:
        cluster_summaries: list[ClusterSelectionSummary] = []
        total_clusters = len(clusters)
        for index, cluster in enumerate(clusters, start=1):
            X_cluster = X_global.iloc[cluster.row_indices].reset_index(drop=True)
            y_cluster = y.iloc[cluster.row_indices].reset_index(drop=True)
            if X_cluster.empty or y_cluster.empty:
                continue

            scores = selector.score_cluster(X_cluster, y_cluster)
            selection = selector.select_features(scores, top_k=top_k, threshold=threshold)
            class_counts = y_cluster.value_counts().to_dict()

            cluster_summaries.append(
                ClusterSelectionSummary(
                    cluster_id=cluster.cluster_id,
                    anchor_index=cluster.anchor_index,
                    n_samples=cluster.n_samples,
                    class_0=int(class_counts.get(0, 0)),
                    class_1=int(class_counts.get(1, 0)),
                    base_entropy=float(scores["base_entropy"].iloc[0]) if not scores.empty else 0.0,
                    selected_features=selection.selected_features,
                    threshold_pass_count=selection.threshold_pass_count,
                    used_fallback=selection.used_fallback,
                    selection_mode=selection.selection_mode,
                    mean_reduction_ratio=float(scores["entropy_reduction_ratio"].mean()) if not scores.empty else 0.0,
                    mean_conditional_entropy=float(scores["conditional_entropy"].mean()) if not scores.empty else 0.0,
                    scores=scores,
                )
            )

            if index == 1 or index % progress_every == 0 or index == total_clusters:
                logger.info(
                    "  Cluster progress: %d/%d | cluster_id=%d | rows=%d | selected=%d",
                    index,
                    total_clusters,
                    cluster.cluster_id,
                    cluster.n_samples,
                    len(selection.selected_features),
                )

        return cluster_summaries

    def _build_run_result(
        self,
        cluster_size: int,
        threshold: float,
        top_features_global: int,
        top_k: int,
        seed: int,
        n_clusters: int,
        cluster_summaries: list[ClusterSelectionSummary],
        runtime_seconds: float,
        status: str,
        error_message: str | None,
        summary_csv: str | None,
        selected_features_csv: str | None,
    ) -> EntropyKNNRunResult:
        selected_counts_raw = [len(summary.selected_features) for summary in cluster_summaries]
        selected_counts_analysis = [self._analysis_selected_count(summary) for summary in cluster_summaries]
        unique_features_raw = sorted({feature for summary in cluster_summaries for feature in summary.selected_features})
        unique_features_analysis = sorted(
            {
                feature
                for summary in cluster_summaries
                if not summary.used_fallback
                for feature in summary.selected_features
            }
        )
        cluster_sizes = [summary.n_samples for summary in cluster_summaries]

        return EntropyKNNRunResult(
            cluster_size=cluster_size,
            threshold=threshold,
            top_features_global=top_features_global,
            top_k=top_k,
            seed=seed,
            n_clusters_requested=n_clusters,
            n_clusters_valid=len(cluster_summaries),
            n_selected_features_total=int(sum(selected_counts_analysis)),
            n_selected_features_total_raw=int(sum(selected_counts_raw)),
            selected_features_unique=len(unique_features_analysis),
            selected_features_unique_raw=len(unique_features_raw),
            mean_cluster_size=float(pd.Series(cluster_sizes).mean()) if cluster_sizes else 0.0,
            mean_selected_per_cluster=float(pd.Series(selected_counts_analysis).mean()) if selected_counts_analysis else 0.0,
            mean_selected_per_cluster_raw=float(pd.Series(selected_counts_raw).mean()) if selected_counts_raw else 0.0,
            mean_entropy_reduction_ratio=float(pd.Series([summary.mean_reduction_ratio for summary in cluster_summaries]).mean()) if cluster_summaries else 0.0,
            mean_conditional_entropy=float(pd.Series([summary.mean_conditional_entropy for summary in cluster_summaries]).mean()) if cluster_summaries else 0.0,
            runtime_seconds=runtime_seconds,
            status=status,
            error_message=error_message,
            summary_csv=summary_csv,
            selected_features_csv=selected_features_csv,
        )

    def _save_run_outputs(self, run_dir: Path, cluster_summaries: list[ClusterSelectionSummary]) -> tuple[str, str]:
        summary_df = self._cluster_summary_frame(cluster_summaries)
        selected_df = self._selected_features_frame(cluster_summaries)

        summary_csv = run_dir / "cluster_summary.csv"
        selected_csv = run_dir / "selected_features_by_cluster.csv"
        summary_df.to_csv(summary_csv, index=False)
        selected_df.to_csv(selected_csv, index=False)
        return str(summary_csv), str(selected_csv)

    @staticmethod
    def _cluster_summary_frame(cluster_summaries: list[ClusterSelectionSummary]) -> pd.DataFrame:
        rows = []
        for summary in cluster_summaries:
            n_selected_raw = len(summary.selected_features)
            n_selected_analysis = EntropyKNNPipeline._analysis_selected_count(summary)
            rows.append(
                {
                    "cluster_id": summary.cluster_id,
                    "anchor_index": summary.anchor_index,
                    "n_samples": summary.n_samples,
                    "class_0": summary.class_0,
                    "class_1": summary.class_1,
                    "base_entropy": summary.base_entropy,
                    "n_selected_features": n_selected_analysis,
                    "n_selected_features_raw": n_selected_raw,
                    "n_selected_features_analysis": n_selected_analysis,
                    "threshold_pass_count": summary.threshold_pass_count,
                    "used_fallback": summary.used_fallback,
                    "selection_mode": summary.selection_mode,
                    "mean_reduction_ratio": summary.mean_reduction_ratio,
                    "mean_conditional_entropy": summary.mean_conditional_entropy,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _selected_features_frame(cluster_summaries: list[ClusterSelectionSummary]) -> pd.DataFrame:
        rows = []
        for summary in cluster_summaries:
            for rank, feature in enumerate(summary.selected_features, start=1):
                rows.append(
                    {
                        "cluster_id": summary.cluster_id,
                        "anchor_index": summary.anchor_index,
                        "used_fallback": summary.used_fallback,
                        "selection_mode": summary.selection_mode,
                        "threshold_pass_count": summary.threshold_pass_count,
                        "rank": rank,
                        "feature": feature,
                    }
                )
        return pd.DataFrame(rows)

    def _build_cluster_rows(self, cluster_summaries: list[ClusterSelectionSummary], result: EntropyKNNRunResult) -> list[dict]:
        rows = []
        for summary in cluster_summaries:
            n_selected_raw = len(summary.selected_features)
            n_selected_analysis = self._analysis_selected_count(summary)
            rows.append(
                {
                    "cluster_size": result.cluster_size,
                    "threshold": result.threshold,
                    "top_features_global": result.top_features_global,
                    "top_k": result.top_k,
                    "seed": result.seed,
                    "cluster_id": summary.cluster_id,
                    "anchor_index": summary.anchor_index,
                    "n_samples": summary.n_samples,
                    "class_0": summary.class_0,
                    "class_1": summary.class_1,
                    "base_entropy": summary.base_entropy,
                    "n_selected_features": n_selected_analysis,
                    "n_selected_features_raw": n_selected_raw,
                    "n_selected_features_analysis": n_selected_analysis,
                    "threshold_pass_count": summary.threshold_pass_count,
                    "used_fallback": summary.used_fallback,
                    "selection_mode": summary.selection_mode,
                    "mean_reduction_ratio": summary.mean_reduction_ratio,
                    "mean_conditional_entropy": summary.mean_conditional_entropy,
                    "selected_features": summary.selected_features,
                }
            )
        return rows

    @staticmethod
    def _analysis_selected_count(summary: ClusterSelectionSummary) -> int:
        return 0 if summary.used_fallback else len(summary.selected_features)

    def _save_sweep_outputs(self, run_df: pd.DataFrame, cluster_df: pd.DataFrame) -> None:
        detailed_path = self.output_dir / "sweep_results.csv"
        cluster_path = self.output_dir / "cluster_results.csv"
        summary_path = self.output_dir / "sweep_summary.csv"

        run_df.to_csv(detailed_path, index=False)
        cluster_df.to_csv(cluster_path, index=False)

        if run_df.empty:
            summary_df = pd.DataFrame()
        else:
            summary_df = (
                run_df.groupby(["cluster_size", "threshold", "top_features_global", "top_k", "n_clusters_requested"], as_index=False)
                .agg(
                    runs=("seed", "count"),
                    runtime_seconds_mean=("runtime_seconds", "mean"),
                    n_clusters_valid_mean=("n_clusters_valid", "mean"),
                    n_selected_features_total_mean=("n_selected_features_total", "mean"),
                    selected_features_unique_mean=("selected_features_unique", "mean"),
                    mean_cluster_size_mean=("mean_cluster_size", "mean"),
                    mean_selected_per_cluster_mean=("mean_selected_per_cluster", "mean"),
                    mean_entropy_reduction_ratio_mean=("mean_entropy_reduction_ratio", "mean"),
                    mean_conditional_entropy_mean=("mean_conditional_entropy", "mean"),
                )
                .sort_values(["cluster_size", "threshold"], ascending=[False, True])
            )

        summary_df.to_csv(summary_path, index=False)

        config_path = self.output_dir / "sweep_config.json"
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump({"runs": int(len(run_df)), "clusters": int(len(cluster_df))}, handle, indent=2)

    @staticmethod
    def _resolve_n_clusters(cluster_size: int, base_size: int, base_n_clusters: int, cluster_schedule: str) -> int:
        if cluster_schedule == "inverse-size":
            return max(1, int(round(base_n_clusters * (base_size / cluster_size))))
        return int(base_n_clusters)

    @staticmethod
    def _format_threshold(threshold: float) -> str:
        return f"{threshold:.3f}".rstrip("0").rstrip(".")
