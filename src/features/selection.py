import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class FeatureSelector:
    """Feature selection using Information Gain and Mutual Information."""

    def __init__(self):
        self.feature_scores = None

    def calculate_entropy(self, y: np.ndarray) -> float:
        """Calculate Shannon entropy of target variable."""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def calculate_information_gain(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_name: str
    ) -> float:
        """
        Calculate Information Gain for a single feature.
        IG(Y|X) = H(Y) - H(Y|X)
        """
        H_Y = self.calculate_entropy(y)

        feature_values = X[feature_name].values
        unique_values = np.unique(feature_values)

        H_Y_given_X = 0
        for value in unique_values:
            mask = feature_values == value
            p_value = np.sum(mask) / len(y)
            H_Y_given_value = self.calculate_entropy(y[mask])
            H_Y_given_X += p_value * H_Y_given_value

        return H_Y - H_Y_given_X

    def rank_features_by_information_gain(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        top_k: int = 50
    ) -> pd.DataFrame:
        """Rank all features by Information Gain."""
        logger.info("Calculating Information Gain for all features...")

        scores = []
        for feature in X.columns:
            if feature == 'file_hash':
                continue
            ig = self.calculate_information_gain(X, y, feature)
            scores.append({'feature': feature, 'information_gain': ig})

        df_scores = pd.DataFrame(scores)
        df_scores = df_scores.sort_values('information_gain', ascending=False)

        logger.info(f"Top feature (IG): {df_scores.iloc[0]['feature']} = {df_scores.iloc[0]['information_gain']:.4f}")
        return df_scores.head(top_k)

    def rank_features_by_mutual_information(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        top_k: int = 50,
        random_state: int = 42
    ) -> pd.DataFrame:
        """Rank features using sklearn's Mutual Information."""
        logger.info("Calculating Mutual Information for all features...")

        X_filtered = X.drop(columns=['file_hash'], errors='ignore')

        mi_scores = mutual_info_classif(
            X_filtered,
            y,
            discrete_features=True,
            random_state=random_state
        )

        df_scores = pd.DataFrame({
            'feature': X_filtered.columns,
            'mutual_information': mi_scores
        })
        df_scores = df_scores.sort_values('mutual_information', ascending=False)

        logger.info(f"Top feature (MI): {df_scores.iloc[0]['feature']} = {df_scores.iloc[0]['mutual_information']:.4f}")
        return df_scores.head(top_k)

    def compare_methods(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        top_k: int = 30
    ) -> pd.DataFrame:
        """Compare IG and MI rankings side by side."""
        ig_df = self.rank_features_by_information_gain(X, y, top_k)
        mi_df = self.rank_features_by_mutual_information(X, y, top_k)

        comparison = ig_df.merge(
            mi_df,
            on='feature',
            how='outer',
            suffixes=('_ig', '_mi')
        )

        # Calculate combined score
        comparison['combined_score'] = (
            comparison['information_gain'].fillna(0) +
            comparison['mutual_information'].fillna(0)
        ) / 2

        comparison = comparison.sort_values('combined_score', ascending=False)

        return comparison
