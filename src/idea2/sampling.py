"""
Sampling strategy: stratified sampling from training set.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class SamplingStrategy:
    """Stratified sampling for reproducible train/val splits."""

    def __init__(self, random_seed: int = 42):
        """
        Initialize sampling strategy.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def stratified_sample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_size: int,
        label_column: str = None,
    ) -> Tuple[pd.DataFrame, pd.Series, List[int]]:
        """
        Stratified sample from dataset preserving class distribution.

        Args:
            X: Features dataframe
            y: Labels series
            sample_size: Number of samples to draw
            label_column: Name of label column (if None, assume y is aligned with X)

        Returns:
            Tuple of (X_sample, y_sample, indices)
            - indices: original indices for reproducibility
        """
        n_total = len(X)
        if sample_size > n_total:
            logger.warning(
                f"Requested sample size ({sample_size}) > total ({n_total}), "
                f"using full dataset"
            )
            sample_size = n_total

        # Create temporary dataframe for stratified split
        df_temp = pd.DataFrame({"features_idx": range(len(X))})
        df_temp["label"] = y.values

        # Stratified groupby sampling
        samples_per_class = {}
        for label in y.unique():
            n_class = (y == label).sum()
            ratio = n_class / n_total
            n_sample = int(sample_size * ratio)
            samples_per_class[label] = n_sample

        sampled_indices = []
        for label, n_sample in samples_per_class.items():
            class_indices = df_temp[df_temp["label"] == label]["features_idx"].values
            if len(class_indices) < n_sample:
                logger.warning(
                    f"Class {label}: requested {n_sample}, only {len(class_indices)} available"
                )
                n_sample = len(class_indices)

            selected = np.random.choice(class_indices, size=n_sample, replace=False)
            sampled_indices.extend(selected)

        sampled_indices = np.array(sampled_indices)

        X_sample = X.iloc[sampled_indices].reset_index(drop=True)
        y_sample = y.iloc[sampled_indices].reset_index(drop=True)

        logger.info(
            f"✓ Stratified sample: {len(X_sample)} samples "
            f"(ratio: {len(X_sample)/len(X):.2%})"
        )
        logger.info(f"  Class distribution: {y_sample.value_counts().to_dict()}")

        return X_sample, y_sample, sampled_indices.tolist()

    def get_sample_sizes(self, total_size: int, num_levels: int = 3) -> List[int]:
        """
        Generate sample size levels: small, medium, large.

        Args:
            total_size: Total number of samples available
            num_levels: Number of levels (default: 3)

        Returns:
            List of sample sizes
        """
        sizes = []
        if num_levels >= 1:
            sizes.append(min(int(total_size * 0.1), total_size))  # 10%
        if num_levels >= 2:
            sizes.append(min(int(total_size * 0.5), total_size))  # 50%
        if num_levels >= 3:
            sizes.append(total_size)  # 100%

        # Remove duplicates and sort
        sizes = sorted(set(sizes))

        logger.info(f"Sample size levels: {sizes}")
        return sizes
