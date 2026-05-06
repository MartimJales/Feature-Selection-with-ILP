"""Filter-based feature selection methods (no model training)."""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway, pointbiserialr
from sklearn.feature_selection import f_classif, mutual_info_classif

logger = logging.getLogger(__name__)


class FilterMethodScorer:
    """Compute filter-based feature selection scores without model training."""

    @staticmethod
    def compute_all_scores(
        X_cluster: pd.DataFrame,
        y_cluster: pd.Series,
        base_entropy: float,
    ) -> dict[str, dict[str, float]]:
        """
        Compute all filter-based scores for every feature in the cluster.

        Args:
            X_cluster: Feature matrix (samples x features), all numeric
            y_cluster: Class labels (0/1 or similar), same length as X_cluster
            base_entropy: H(Y) for the cluster (pre-computed)

        Returns:
            Dictionary where keys are feature names and values are dicts of scores:
            {
                "feature_name": {
                    "entropy_reduction_ratio": float,
                    "conditional_entropy": float,
                    "mutual_information": float,
                    "normalized_ig": float,
                    "chi2_stat": float,
                    "chi2_pvalue": float,
                    "f_stat": float,
                    "f_pvalue": float,
                    "pearson_r": float,
                    "pearson_pvalue": float,
                }
            }
        """
        scores: dict[str, dict[str, float]] = {}

        # Vectorized: compute MI for all features at once
        mi_scores = FilterMethodScorer._mutual_information_all(X_cluster, y_cluster)

        # Vectorized: compute chi2 for all features
        chi2_results = FilterMethodScorer._chi2_all(X_cluster, y_cluster)

        # Vectorized: compute F-statistic for all features
        f_results = FilterMethodScorer._f_stat_all(X_cluster, y_cluster)

        # Pearson / point-biserial per feature (vectorized where possible)
        pearson_results = FilterMethodScorer._pearson_all(X_cluster, y_cluster)

        # Entropy metrics per feature
        entropy_results = FilterMethodScorer._entropy_all(X_cluster, y_cluster, base_entropy)

        # Consolidate
        for feature_name in X_cluster.columns:
            scores[feature_name] = {
                "entropy_reduction_ratio": float(entropy_results[feature_name]["entropy_reduction_ratio"]),
                "conditional_entropy": float(entropy_results[feature_name]["conditional_entropy"]),
                "mutual_information": float(mi_scores.get(feature_name, 0.0)),
                "normalized_ig": float(entropy_results[feature_name]["entropy_reduction_ratio"]),  # normalized by H(Y)
                "chi2_stat": float(chi2_results[feature_name]["stat"]),
                "chi2_pvalue": float(chi2_results[feature_name]["pvalue"]),
                "f_stat": float(f_results[feature_name]["stat"]),
                "f_pvalue": float(f_results[feature_name]["pvalue"]),
                "pearson_r": float(pearson_results[feature_name]["r"]),
                "pearson_pvalue": float(pearson_results[feature_name]["pvalue"]),
            }

        return scores

    @staticmethod
    def _mutual_information_all(X_cluster: pd.DataFrame, y_cluster: pd.Series) -> dict[str, float]:
        """Compute mutual information between each feature and target using sklearn."""
        try:
            # Convert to numpy; sklearn can handle continuous features
            X_np = X_cluster.values
            y_np = y_cluster.values

            # mutual_info_classif expects discrete or continuous features
            # For binary/sparse features, it should work well
            mi_array = mutual_info_classif(X_np, y_np, random_state=42)

            return {feature: float(mi) for feature, mi in zip(X_cluster.columns, mi_array)}
        except Exception as e:
            logger.warning(f"MI computation failed: {e}. Returning zeros.")
            return {feature: 0.0 for feature in X_cluster.columns}

    @staticmethod
    def _chi2_all(X_cluster: pd.DataFrame, y_cluster: pd.Series) -> dict[str, dict[str, float]]:
        """Compute chi-squared statistic for each feature vs target."""
        results = {}

        for feature_name in X_cluster.columns:
            try:
                feature_vals = X_cluster[feature_name].values
                target_vals = y_cluster.values

                # Create contingency table
                # For binary features (0/1), this is straightforward
                contingency = pd.crosstab(feature_vals, target_vals)

                # Chi-squared test
                chi2, pval, _, _ = chi2_contingency(contingency.values)

                results[feature_name] = {"stat": float(chi2), "pvalue": float(pval)}
            except Exception as e:
                logger.debug(f"Chi2 failed for {feature_name}: {e}")
                results[feature_name] = {"stat": 0.0, "pvalue": 1.0}

        return results

    @staticmethod
    def _f_stat_all(X_cluster: pd.DataFrame, y_cluster: pd.Series) -> dict[str, dict[str, float]]:
        """Compute F-statistic (ANOVA) for each feature vs target."""
        try:
            X_np = X_cluster.values
            y_np = y_cluster.values

            # sklearn's f_classif handles both continuous and discrete features
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                f_array, pval_array = f_classif(X_np, y_np)

            f_array = np.asarray(f_array, dtype=float)
            pval_array = np.asarray(pval_array, dtype=float)

            # Replace invalid numbers (nan/inf) with neutral values
            bad_mask = ~np.isfinite(f_array)
            if bad_mask.any():
                f_array[bad_mask] = 0.0
                pval_array[bad_mask] = 1.0

            return {
                feature: {"stat": float(f_stat), "pvalue": float(pval)}
                for feature, f_stat, pval in zip(X_cluster.columns, f_array, pval_array)
            }
        except Exception as e:
            logger.debug(f"F-stat computation failed: {e}")
            return {feature: {"stat": 0.0, "pvalue": 1.0} for feature in X_cluster.columns}

    @staticmethod
    def _pearson_all(X_cluster: pd.DataFrame, y_cluster: pd.Series) -> dict[str, dict[str, float]]:
        """Compute Pearson/point-biserial correlation for each feature vs target."""
        results = {}
        y_np = y_cluster.values

        for feature_name in X_cluster.columns:
            try:
                feature_vals = X_cluster[feature_name].values

                # If the feature is constant in this cluster, correlation is undefined.
                # Return neutral values and avoid calling scipy which emits warnings.
                try:
                    unique_count = pd.Series(feature_vals).nunique(dropna=False)
                except Exception:
                    unique_count = len(np.unique(feature_vals))

                if unique_count <= 1 or np.nanstd(feature_vals) == 0:
                    results[feature_name] = {"r": 0.0, "pvalue": 1.0}
                    continue

                # Point-biserial correlation (same as Pearson for binary target)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    r, pval = pointbiserialr(y_np, feature_vals)

                # If scipy returns non-finite values, map to neutral values
                if not np.isfinite(r) or not np.isfinite(pval):
                    results[feature_name] = {"r": 0.0, "pvalue": 1.0}
                else:
                    results[feature_name] = {"r": float(r), "pvalue": float(pval)}
            except Exception as e:
                logger.debug(f"Pearson failed for {feature_name}: {e}")
                results[feature_name] = {"r": 0.0, "pvalue": 1.0}

        return results

    @staticmethod
    def _entropy_all(
        X_cluster: pd.DataFrame,
        y_cluster: pd.Series,
        base_entropy: float,
    ) -> dict[str, dict[str, float]]:
        """Compute entropy reduction ratio and conditional entropy for each feature."""
        from .entropy import conditional_entropy, shannon_entropy

        results = {}

        for feature_name in X_cluster.columns:
            try:
                feature_vals = X_cluster[feature_name]
                h_y_x = conditional_entropy(y_cluster, feature_vals)

                # Entropy reduction ratio: (H(Y) - H(Y|X)) / H(Y)
                if base_entropy > 0.0:
                    err = (base_entropy - h_y_x) / base_entropy
                else:
                    err = 0.0

                results[feature_name] = {
                    "conditional_entropy": float(h_y_x),
                    "entropy_reduction_ratio": float(err),
                }
            except Exception as e:
                logger.debug(f"Entropy failed for {feature_name}: {e}")
                results[feature_name] = {
                    "conditional_entropy": float(base_entropy),
                    "entropy_reduction_ratio": 0.0,
                }

        return results

    @staticmethod
    def rank_features_by_method(scores: dict[str, dict[str, float]]) -> dict[str, dict[str, int]]:
        """
        Rank features by each scoring method (descending).

        Args:
            scores: Output from compute_all_scores

        Returns:
            Dict where keys are feature names, values are dicts of ranks per method
            {
                "feature_name": {
                    "rank_entropy_reduction_ratio": 1,
                    "rank_mutual_information": 2,
                    ...
                }
            }
        """
        methods = [
            "entropy_reduction_ratio",
            "mutual_information",
            "normalized_ig",
            "chi2_stat",
            "f_stat",
            "pearson_r",
        ]

        ranks: dict[str, dict[str, int]] = {feature: {} for feature in scores.keys()}

        for method in methods:
            # Extract scores for this method, sort descending
            method_scores = [(feature, scores[feature][method]) for feature in scores.keys()]
            method_scores.sort(key=lambda x: x[1], reverse=True)

            # Assign ranks (1-indexed)
            for rank, (feature, _) in enumerate(method_scores, start=1):
                ranks[feature][f"rank_{method}"] = rank

        return ranks
