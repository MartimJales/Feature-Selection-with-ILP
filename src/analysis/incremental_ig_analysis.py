import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt

from src.features.extractor import JSONFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_incremental_ig_from_parquet(parquet_file, labels_path, output_dir="./reports/feature_analysis"):
    """
    Load rankings from Parquet and analyze incremental IG gain.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load all feature rankings from Parquet
    logger.info("Loading feature rankings from Parquet...")
    df_rankings = pd.read_parquet(parquet_file)
    logger.info(f"Loaded {len(df_rankings)} features with IG and MI scores")

    # Calculate cumulative IG scores in steps of 100
    step_size = 100
    max_k = len(df_rankings)

    incremental_results = []

    for top_k in range(step_size, max_k + 1, step_size):
        # Get top-K features
        top_k_df = df_rankings.iloc[:top_k]

        # Calculate cumulative metrics
        cumulative_ig = top_k_df['information_gain'].sum()
        cumulative_mi = top_k_df['mutual_information'].sum()
        mean_ig = top_k_df['information_gain'].mean()
        mean_mi = top_k_df['mutual_information'].mean()

        # Get the worst feature in this batch (the one added at this step)
        worst_ig = top_k_df.iloc[-1]['information_gain']
        worst_mi = top_k_df.iloc[-1]['mutual_information']

        incremental_results.append({
            'top_k': top_k,
            'cumulative_ig': cumulative_ig,
            'cumulative_mi': cumulative_mi,
            'mean_ig': mean_ig,
            'mean_mi': mean_mi,
            'worst_ig_in_batch': worst_ig,
            'worst_mi_in_batch': worst_mi
        })

        logger.info(f"Top-{top_k}: Cumulative IG={cumulative_ig:.4f}, Mean IG={mean_ig:.4f}")

    # Create results dataframe
    incremental_df = pd.DataFrame(incremental_results)

    # Calculate incremental gain (difference from previous)
    incremental_df['ig_gain'] = incremental_df['cumulative_ig'].diff()
    incremental_df['mi_gain'] = incremental_df['cumulative_mi'].diff()
    incremental_df['mean_ig_gain'] = incremental_df['mean_ig'].diff()

    # Save to CSV
    output_file = output_path / "incremental_ig_analysis.csv"
    incremental_df.to_csv(output_file, index=False)
    logger.info(f"✓ Incremental analysis saved to {output_file}")

    return incremental_df

def plot_incremental_analysis(df, output_dir="./reports/feature_analysis"):
    """Plot incremental IG growth."""
    output_path = Path(output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Cumulative IG
    axes[0, 0].plot(df['top_k'], df['cumulative_ig'], 'o-', linewidth=2, markersize=6, color='#3357FF')
    axes[0, 0].set_xlabel('Number of Features (Top-K)', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Cumulative Information Gain', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Cumulative IG vs Feature Count', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)

    # Plot 2: Cumulative MI
    axes[0, 1].plot(df['top_k'], df['cumulative_mi'], 'o-', linewidth=2, markersize=6, color='#FF5733')
    axes[0, 1].set_xlabel('Number of Features (Top-K)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Cumulative Mutual Information', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Cumulative MI vs Feature Count', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    # Plot 3: Mean IG per Feature
    axes[1, 0].plot(df['top_k'], df['mean_ig'], 'o-', linewidth=2, markersize=6, color='#3357FF')
    axes[1, 0].set_xlabel('Number of Features (Top-K)', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Mean IG per Feature', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Average IG per Feature', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    # Plot 4: Incremental Gain (IG added per 100 features)
    axes[1, 1].bar(df['top_k'], df['ig_gain'].fillna(0), width=80, alpha=0.7, color='#3357FF')
    axes[1, 1].set_xlabel('Number of Features (Top-K)', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('IG Gain per 100 Features', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Incremental IG Gain', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3, axis='y')

    plt.tight_layout()

    # Save plots
    png_file = output_path / "incremental_ig_analysis.png"
    pdf_file = output_path / "incremental_ig_analysis.pdf"

    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_file, bbox_inches='tight')
    logger.info(f"✓ Plots saved to {png_file} and {pdf_file}")

if __name__ == "__main__":
    # Paths
    parquet_file = "./reports/feature_analysis/feature_rankings_all.parquet"
    labels_path = "./data/training_set.csv"
    output_dir = "./reports/feature_analysis"

    # Check if parquet file exists
    if not Path(parquet_file).exists():
        logger.error(f"Parquet file not found: {parquet_file}")
        logger.error("Run the pipeline first: ./run_complete_pipeline.sh full")
        exit(1)

    # Analyze incremental IG
    incremental_df = analyze_incremental_ig_from_parquet(parquet_file, labels_path, output_dir)

    # Plot results
    plot_incremental_analysis(incremental_df, output_dir)

    # Print summary
    print("\n" + "="*80)
    print("INCREMENTAL IG ANALYSIS SUMMARY")
    print("="*80)
    print(incremental_df.to_string(index=False))
    print("="*80)
    print(f"\nKey Findings:")
    print(f"  - Total features analyzed: {incremental_df['top_k'].max()}")
    print(f"  - Total cumulative IG: {incremental_df['cumulative_ig'].iloc[-1]:.4f}")
    print(f"  - Average IG per feature: {incremental_df['mean_ig'].iloc[-1]:.6f}")
    print(f"  - Highest IG gain (per 100): {incremental_df['ig_gain'].max():.4f} at {incremental_df[incremental_df['ig_gain'].idxmax()]['top_k']:.0f} features")
    print(f"  - Lowest IG gain (per 100): {incremental_df['ig_gain'].min():.4f} at {incremental_df[incremental_df['ig_gain'].idxmin()]['top_k']:.0f} features")
    print("="*80)
