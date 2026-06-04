"""Data loading helpers for the Entropy KNN pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.idea2.data_loader import Idea2DataLoader


@dataclass(slots=True)
class EntropyKNNDataBundle:
    """Container for aligned data used by the pipeline."""

    X: pd.DataFrame
    y: pd.Series
    rankings: pd.DataFrame
    ranked_features: list[str]
    original_indices: list[int] | None = None


class EntropyKNNDataLoader:
    """Thin wrapper around the existing Idea2 loader."""

    def __init__(
        self,
        features_path: str = "./reports/extracted_features.parquet",
        labels_path: str = "./data/training_set.csv",
        rankings_path: str = "./reports/feature_analysis/feature_rankings_all.parquet",
    ) -> None:
        self.loader = Idea2DataLoader(
            features_path=features_path,
            labels_path=labels_path,
            rankings_path=rankings_path,
        )

    def load(self) -> EntropyKNNDataBundle:
        X, y, rankings = self.loader.load()
        ranked_features = self._get_ranked_features(X, rankings)
        return EntropyKNNDataBundle(X=X, y=y, rankings=rankings, ranked_features=ranked_features)

    @staticmethod
    def _get_ranked_features(X: pd.DataFrame, rankings: pd.DataFrame) -> list[str]:
        available = set(X.columns)
        return [str(feature) for feature in rankings["feature"].astype(str).tolist() if feature in available]
