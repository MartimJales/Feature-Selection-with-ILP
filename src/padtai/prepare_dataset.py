#!/usr/bin/env python3
"""
Prepare top-200 IG features for PADTAI.

1. Reads feature_rankings_all.parquet
2. Extracts top-200 by Information Gain
3. Loads raw features + labels
4. Creates CSV with top-200 + label in last column
5. Saves to data/ilp/top200_ig_malware.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def prepare_top200_dataset(
    rankings_file: str = "./reports/feature_analysis/feature_rankings_all.parquet",
    features_file: str = "./reports/extracted_features.parquet",
    labels_file: str = "./data/training_set.csv",
    output_file: str = "./data/ilp/top200_ig_malware.csv",
    top_k: int = 200
) -> str:
    """
    Prepare top-K features for PADTAI.

    Args:
        rankings_file: Path to feature_rankings_all.parquet
        features_file: Path to extracted_features.parquet
        labels_file: Path to training_set.csv with labels
        output_file: Output CSV path
        top_k: Number of top features to select (default: 200)

    Returns:
        Path to output file
    """

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load feature rankings
    logger.info(f"Loading feature rankings from {rankings_file}...")
    rankings_df = pd.read_parquet(rankings_file)

    # 2. Get top-K by Information Gain
    logger.info(f"Selecting top-{top_k} features by Information Gain...")
    top_features = rankings_df.nlargest(top_k, 'information_gain')['feature'].tolist()
    logger.info(f"✓ Selected {len(top_features)} features")
    print(f"Top-{top_k} features (by IG):")
    for i, feat in enumerate(top_features[:10], 1):
        print(f"  {i}. {feat}")
    if len(top_features) > 10:
        print(f"  ... and {len(top_features) - 10} more")

    # 3. Load features
    logger.info(f"\nLoading features from {features_file}...")
    if features_file.endswith('.parquet'):
        features_df = pd.read_parquet(features_file)
    else:
        features_df = pd.read_csv(features_file)
    logger.info(f"✓ Loaded {features_df.shape[0]} samples × {features_df.shape[1]} features")

    # 4. Load labels
    logger.info(f"Loading labels from {labels_file}...")
    labels_df = pd.read_csv(labels_file)
    logger.info(f"✓ Loaded {len(labels_df)} labels")

    # 5. Merge features with labels
    logger.info("\nMerging features with labels...")

    # Normalize file_hash
    if 'file_hash' in features_df.columns:
        features_df['file_hash'] = features_df['file_hash'].str.lower().str.strip()
    else:
        logger.warning("⚠ 'file_hash' column not found in features. Trying to match by row index...")

    labels_df['sha256'] = labels_df['sha256'].str.lower().str.strip()

    # Merge
    merged_df = features_df.merge(
        labels_df[['sha256', 'label']],
        left_on='file_hash' if 'file_hash' in features_df.columns else features_df.index,
        right_on='sha256',
        how='left'
    )

    # Remove rows without labels
    merged_df = merged_df.dropna(subset=['label'])
    logger.info(f"✓ Matched {len(merged_df)} samples with labels")

    # 6. Select only top-200 features + label
    logger.info(f"\nSelecting top-{top_k} features + label...")

    # Filter to top features
    cols_to_keep = [col for col in top_features if col in merged_df.columns]
    missing_features = set(top_features) - set(cols_to_keep)

    if missing_features:
        logger.warning(f"⚠ {len(missing_features)} features not found in data: {list(missing_features)[:5]}")

    # Build final dataframe
    final_df = merged_df[cols_to_keep + ['label']].copy()

    # Reorder: top features + label in last column
    final_df = final_df[cols_to_keep + ['label']]

    logger.info(f"✓ Final dataset: {final_df.shape[0]} samples × {final_df.shape[1]} features (including label)")

    # 7. Check data types (should be binary for top features)
    logger.info("\nData type summary:")
    print(f"  Shape: {final_df.shape}")
    print(f"  Label distribution: {final_df['label'].value_counts().to_dict()}")
    print(f"  Features with all 0/1 values: {sum(1 for col in cols_to_keep if set(final_df[col].unique()).issubset({0, 1, 0.0, 1.0}))}/{len(cols_to_keep)}")

    # 8. Save to CSV
    logger.info(f"\nSaving to {output_file}...")
    final_df.to_csv(output_file, index=False)
    logger.info(f"✓ Saved successfully!")

    return output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare top-K features for PADTAI")
    parser.add_argument('--rankings', default='./reports/feature_analysis/feature_rankings_all.parquet',
                        help='Path to feature_rankings_all.parquet')
    parser.add_argument('--features', default='./reports/extracted_features.parquet',
                        help='Path to extracted_features.parquet or .csv')
    parser.add_argument('--labels', default='./data/training_set.csv',
                        help='Path to training_set.csv')
    parser.add_argument('--output', default='./data/ilp/top200_ig_malware.csv',
                        help='Output CSV file path')
    parser.add_argument('--top-k', type=int, default=200,
                        help='Number of top features (default: 200)')

    args = parser.parse_args()

    prepare_top200_dataset(
        rankings_file=args.rankings,
        features_file=args.features,
        labels_file=args.labels,
        output_file=args.output,
        top_k=args.top_k
    )
