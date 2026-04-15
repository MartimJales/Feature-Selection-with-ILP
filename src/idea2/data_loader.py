"""
Data loader for Idea2: loads training set and feature rankings.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class Idea2DataLoader:
    """Load and validate training set and feature rankings."""

    def __init__(
        self,
        training_set_path: str = "./data/training_set.csv",
        rankings_path: str = "./reports/feature_analysis/feature_rankings_all.parquet",
    ):
        """
        Initialize data loader.

        Args:
            training_set_path: Path to training set CSV
            rankings_path: Path to feature rankings Parquet
        """
        self.training_set_path = Path(training_set_path)
        self.rankings_path = Path(rankings_path)

        self.df_train = None
        self.df_rankings = None
        self.feature_names = None
        self.label_column = None

    def load(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Load training set and rankings.

        Returns:
            Tuple of (X, y, rankings_df)
            - X: features dataframe
            - y: labels series
            - rankings_df: feature rankings sorted by IG
        """
        logger.info(f"Loading training set from {self.training_set_path}...")
        self.df_train = pd.read_csv(self.training_set_path)
        logger.info(f"✓ Loaded {len(self.df_train)} samples with {len(self.df_train.columns)} columns")

        # Identify label column
        self.label_column = self._detect_label_column()
        if self.label_column is None:
            raise ValueError("Could not detect label column (expected: 'Label', 'label', 'target', 'y')")

        logger.info(f"✓ Detected label column: '{self.label_column}'")

        # Split X and y
        y = self.df_train[self.label_column]
        X = self.df_train.drop(columns=[self.label_column])

        logger.info(f"  Class distribution: {y.value_counts().to_dict()}")

        # Load rankings
        logger.info(f"Loading rankings from {self.rankings_path}...")
        self.df_rankings = pd.read_parquet(self.rankings_path)
        logger.info(f"✓ Loaded {len(self.df_rankings)} ranked features")

        # Verify top features
        top_5 = self.df_rankings.head(5)
        logger.info(f"Top-5 features by IG:")
        for idx, row in top_5.iterrows():
            logger.info(f"  {row['feature']}: IG={row['information_gain']:.6f}")

        self.feature_names = self.df_rankings['feature'].tolist()

        return X, y, self.df_rankings

    def _detect_label_column(self) -> str:
        """Auto-detect label column name."""
        candidates = ['Label', 'label', 'target', 'y', 'class', 'Class']
        for col in candidates:
            if col in self.df_train.columns:
                return col
        return None

    def validate_features(self, features: List[str], data: pd.DataFrame) -> bool:
        """
        Validate that all requested features exist in data.

        Args:
            features: List of feature names
            data: DataFrame to check

        Returns:
            True if all features present, False otherwise
        """
        missing = set(features) - set(data.columns)
        if missing:
            logger.warning(f"Missing features: {missing}")
            return False
        return True

    def get_label_column(self) -> str:
        """Return detected label column name."""
        return self.label_column
