"""
Data loader for Idea2: loads extracted features, labels and feature rankings.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class Idea2DataLoader:
    """Load and validate extracted features, labels and feature rankings."""

    def __init__(
        self,
        features_path: str = "./reports/extracted_features.parquet",
        labels_path: str = "./data/training_set.csv",
        rankings_path: str = "./reports/feature_analysis/feature_rankings_all.parquet",
    ):
        """
        Initialize data loader.

        Args:
            features_path: Path to extracted features file (.parquet/.csv)
            labels_path: Path to labels CSV
            rankings_path: Path to feature rankings Parquet
        """
        self.features_path = Path(features_path)
        self.labels_path = Path(labels_path)
        self.rankings_path = Path(rankings_path)

        self.df_features = None
        self.df_labels = None
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
        logger.info(f"Loading extracted features from {self.features_path}...")
        self.df_features = self._load_features_file(self.features_path)
        logger.info(
            f"✓ Loaded {len(self.df_features)} samples with {len(self.df_features.columns)} columns"
        )

        logger.info(f"Loading labels from {self.labels_path}...")
        self.df_labels = pd.read_csv(self.labels_path)
        logger.info(f"✓ Loaded {len(self.df_labels)} labels"
        )

        # Identify label / hash columns
        label_column, hash_column = self._detect_label_and_hash_columns(self.df_labels)
        self.label_column = label_column
        if self.label_column is None:
            raise ValueError("Could not detect label column (expected: 'Label', 'label', 'target', 'y')")
        if hash_column is None:
            raise ValueError("Could not detect hash column (expected: 'sha256', 'hash', 'file_hash' or 'file')")

        logger.info(f"✓ Detected label column: '{self.label_column}'")
        logger.info(f"✓ Detected hash column: '{hash_column}'")

        # Normalize join keys
        features_df = self.df_features.copy()
        labels_df = self.df_labels.copy()

        if "file_hash" not in features_df.columns:
            raise ValueError("Features file must contain a 'file_hash' column")

        features_df["file_hash"] = features_df["file_hash"].astype(str).str.lower().str.strip()
        labels_df[hash_column] = labels_df[hash_column].astype(str).str.lower().str.strip()

        merged = features_df.merge(
            labels_df[[hash_column, self.label_column]],
            left_on="file_hash",
            right_on=hash_column,
            how="inner",
        )
        merged = merged.drop(columns=["file_hash", hash_column], errors="ignore")
        merged = merged.dropna(subset=[self.label_column])

        y = merged[self.label_column].astype(int)
        X = merged.drop(columns=[self.label_column])

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

        available_features = set(X.columns)
        self.feature_names = [feat for feat in self.df_rankings['feature'].tolist() if feat in available_features]
        missing_ranked = len(self.df_rankings) - len(self.feature_names)
        if missing_ranked:
            logger.warning(f"⚠ {missing_ranked} ranked features are not present in the extracted feature matrix")
        logger.info(f"✓ Using {len(self.feature_names)} ranked features present in the extracted matrix")

        return X, y, self.df_rankings

    def _load_features_file(self, features_path: Path) -> pd.DataFrame:
        if features_path.suffix == ".parquet":
            return pd.read_parquet(features_path)
        if features_path.suffix == ".csv":
            return pd.read_csv(features_path)
        raise ValueError(f"Unsupported features file format: {features_path.suffix}")

    def _detect_label_and_hash_columns(self, labels_df: pd.DataFrame) -> tuple[str | None, str | None]:
        """Auto-detect label and hash column names."""
        label_candidates = ['label', 'Label', 'target', 'y', 'class', 'Class']
        hash_candidates = ['sha256', 'SHA256', 'hash', 'file_hash', 'file']

        label_column = next((col for col in label_candidates if col in labels_df.columns), None)
        hash_column = next((col for col in hash_candidates if col in labels_df.columns), None)

        if hash_column == 'file':
            # The legacy CSV may store hashes in a file column ending in .json
            labels_df['file'] = labels_df['file'].astype(str).str.replace('.json', '', regex=False)

        return label_column, hash_column

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
