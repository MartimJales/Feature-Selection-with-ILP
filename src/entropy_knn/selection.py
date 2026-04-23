"""Local scoring and feature selection helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .entropy import conditional_entropy, entropy_reduction_ratio, shannon_entropy


@dataclass(slots=True)
class ClusterSelectionSummary:
    """Summary of the selection performed inside one cluster."""

    cluster_id: int
    anchor_index: int
    n_samples: int
    class_0: int
    class_1: int
    base_entropy: float
    selected_features: list[str]
    threshold_pass_count: int
    used_fallback: bool
    selection_mode: str
    mean_reduction_ratio: float
    mean_conditional_entropy: float
    scores: pd.DataFrame


@dataclass(slots=True)
class SelectionOutcome:
    """Selection result together with decision metadata."""

    selected_features: list[str]
    threshold_pass_count: int
    used_fallback: bool
    selection_mode: str


class EntropyFeatureSelector:
    """Score local features by entropy and keep the best ones."""

    def score_cluster(self, X_cluster: pd.DataFrame, y_cluster: pd.Series) -> pd.DataFrame:
        rows = []
        base_entropy = shannon_entropy(y_cluster)

        for feature_name in X_cluster.columns:
            conditional = conditional_entropy(y_cluster, X_cluster[feature_name])
            reduction_ratio = entropy_reduction_ratio(y_cluster, X_cluster[feature_name])
            rows.append(
                {
                    "feature": feature_name,
                    "conditional_entropy": conditional,
                    "entropy_reduction_ratio": reduction_ratio,
                    "base_entropy": base_entropy,
                }
            )

        scores = pd.DataFrame(rows)
        if not scores.empty:
            scores = scores.sort_values(
                ["entropy_reduction_ratio", "conditional_entropy", "feature"],
                ascending=[False, True, True],
            ).reset_index(drop=True)
        return scores

    def select_features(
        self,
        scores: pd.DataFrame,
        top_k: int,
        threshold: float,
        fallback_to_top_k: bool = True,
    ) -> SelectionOutcome:
        if scores.empty:
            return SelectionOutcome([], 0, False, "empty")

        filtered = scores[scores["entropy_reduction_ratio"] >= threshold]
        threshold_pass_count = int(len(filtered))

        if filtered.empty and fallback_to_top_k:
            filtered = scores.head(top_k)
            selection_mode = "fallback_top_k"
            used_fallback = True
        else:
            filtered = filtered.head(top_k)
            selection_mode = "threshold_top_k" if threshold_pass_count > 0 else "threshold_empty"
            used_fallback = False

        return SelectionOutcome(
            selected_features=filtered["feature"].astype(str).tolist(),
            threshold_pass_count=threshold_pass_count,
            used_fallback=used_fallback,
            selection_mode=selection_mode,
        )
