"""Balanced (1:1 malware:goodware) orchestration for Entropy KNN pipeline."""

from __future__ import annotations

import logging
from dataclasses import replace

import pandas as pd

from src.entropy_knn.pipeline import EntropyKNNPipeline

logger = logging.getLogger(__name__)


class BalancedEntropyKNNPipeline(EntropyKNNPipeline):
    """Entropy KNN pipeline that balances data to a 1:1 class ratio before clustering."""

    def __init__(
        self,
        features_path: str = "./reports/extracted_features.parquet",
        labels_path: str = "./data/training_set.csv",
        rankings_path: str = "./reports/feature_analysis/feature_rankings_all.parquet",
        output_dir: str = "./reports/entropy_knn_balanced",
        balance_seed: int = 42,
    ) -> None:
        super().__init__(
            features_path=features_path,
            labels_path=labels_path,
            rankings_path=rankings_path,
            output_dir=output_dir,
        )
        self.balance_seed = int(balance_seed)

    def _load_bundle(self):
        """Load and cache a balanced bundle with 1:1 class proportion."""
        if self.bundle is None:
            raw_bundle = self.loader.load()
            self.bundle = self._balance_bundle(raw_bundle, self.balance_seed)
        return self.bundle

    @staticmethod
    def _balance_bundle(bundle, random_seed: int):
        y = bundle.y.astype(int)

        malware_mask = y == 1
        goodware_mask = y == 0

        malware_count = int(malware_mask.sum())
        goodware_count = int(goodware_mask.sum())

        if malware_count == 0 or goodware_count == 0:
            raise ValueError(
                "Cannot build 1:1 dataset because one class is missing "
                f"(malware={malware_count}, goodware={goodware_count})"
            )

        target_per_class = min(malware_count, goodware_count)

        malware_indices = y[malware_mask].sample(n=target_per_class, random_state=random_seed).index
        goodware_indices = y[goodware_mask].sample(n=target_per_class, random_state=random_seed).index

        selected_indices = pd.Index(malware_indices.tolist() + goodware_indices.tolist())

        # Shuffle final selection to avoid class blocks.
        selected_indices = pd.Series(selected_indices).sample(frac=1.0, random_state=random_seed).tolist()

        X_balanced = bundle.X.iloc[selected_indices].reset_index(drop=True)
        y_balanced = y.iloc[selected_indices].reset_index(drop=True)

        logger.info(
            "Balanced dataset ready: malware=%d | goodware=%d | total=%d",
            int((y_balanced == 1).sum()),
            int((y_balanced == 0).sum()),
            len(y_balanced),
        )

        ranked_features = [feature for feature in bundle.ranked_features if feature in X_balanced.columns]

        return replace(
            bundle,
            X=X_balanced,
            y=y_balanced,
            ranked_features=ranked_features,
        )
