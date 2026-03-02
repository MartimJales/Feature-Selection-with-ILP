import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import argparse

from src.features.extractor import JSONFeatureExtractor
from src.features.selection import FeatureSelector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_features(
    json_dir: str = "./data/destino",
    labels_path: str = "./data/training_set.csv",
    output_dir: str = "./reports/feature_analysis",
    top_k: int = 30
):
    """Complete feature analysis pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract features and load labels
    extractor = JSONFeatureExtractor(json_dir)
    X, y = extractor.load_with_labels(labels_path)

    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Feature selection
    selector = FeatureSelector()
    comparison_df = selector.compare_methods(X, y, top_k=top_k)

    # Save & plot
    comparison_df.to_csv(output_path / f"feature_rankings_top{top_k}.csv", index=False)
    plot_feature_importance(comparison_df, output_path, top_k)

def plot_feature_importance(df: pd.DataFrame, output_dir: Path, top_k: int):
    """Plot top features by IG and MI."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Information Gain
    top_ig = df.nlargest(min(15, top_k), 'information_gain')
    axes[0].barh(top_ig['feature'], top_ig['information_gain'])
    axes[0].set_xlabel('Information Gain')
    axes[0].set_title(f'Top 15 Features by Information Gain (top-k={top_k})')
    axes[0].invert_yaxis()

    # Mutual Information
    top_mi = df.nlargest(min(15, top_k), 'mutual_information')
    axes[1].barh(top_mi['feature'], top_mi['mutual_information'])
    axes[1].set_xlabel('Mutual Information')
    axes[1].set_title(f'Top 15 Features by Mutual Information (top-k={top_k})')
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_dir / f"feature_importance_top{top_k}.png", dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to {output_dir / f'feature_importance_top{top_k}.png'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature analysis with configurable top-k')
    parser.add_argument('--json-dir', default='./data/destino', help='Directory with JSON files')
    parser.add_argument('--labels-path', default='./data/training_set.csv', help='Path to labels CSV')
    parser.add_argument('--output-dir', default='./reports/feature_analysis', help='Output directory')
    parser.add_argument('--top-k', type=int, default=30, help='Number of top features to select')

    args = parser.parse_args()
    analyze_features(args.json_dir, args.labels_path, args.output_dir, args.top_k)
