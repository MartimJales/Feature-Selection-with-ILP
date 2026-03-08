import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_incremental_ig_detailed(parquet_file, output_dir="./reports/feature_analysis"):
    """
    Detailed incremental IG analysis focused on growth phase (top-20k features).
    Uses smaller steps for better granularity.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load all feature rankings from Parquet
    logger.info("Loading feature rankings from Parquet...")
    df_rankings = pd.read_parquet(parquet_file)
    logger.info(f"Loaded {len(df_rankings)} features with IG and MI scores")

    # Detailed analysis: steps of 50 up to 20k features
    step_size = 50
    max_k = min(1500, len(df_rankings))

    logger.info(f"Analyzing incremental growth with step={step_size} up to top-{max_k}")

    incremental_results = []

    for top_k in range(step_size, max_k + 1, step_size):
        # Get top-K features
        top_k_df = df_rankings.iloc[:top_k]

        # Calculate cumulative metrics
        cumulative_ig = top_k_df['information_gain'].sum()
        cumulative_mi = top_k_df['mutual_information'].sum()
        mean_ig = top_k_df['information_gain'].mean()
        mean_mi = top_k_df['mutual_information'].mean()

        # Get the worst feature in this batch
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

        if top_k % 1000 == 0:
            logger.info(f"Top-{top_k}: Cumulative IG={cumulative_ig:.4f}, Mean IG={mean_ig:.6f}")

    # Create results dataframe
    incremental_df = pd.DataFrame(incremental_results)

    # Calculate incremental gain
    incremental_df['ig_gain'] = incremental_df['cumulative_ig'].diff()
    incremental_df['mi_gain'] = incremental_df['cumulative_mi'].diff()
    incremental_df['mean_ig_change'] = incremental_df['mean_ig'].diff()

    # Save to CSV
    output_file = output_path / "incremental_ig_analysis_detailed.csv"
    incremental_df.to_csv(output_file, index=False)
    logger.info(f"✓ Detailed incremental analysis saved to {output_file}")

    return incremental_df

def plot_incremental_analysis(df, output_dir="./reports/feature_analysis"):
    """Create static PNG plots with matplotlib."""
    output_path = Path(output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].plot(df['top_k'], df['cumulative_ig'], color='#3357FF', linewidth=2)
    axes[0, 0].set_title('Cumulative Information Gain', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Top-K Features', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Cumulative IG', fontsize=11, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(df['top_k'], df['cumulative_mi'], color='#FF5733', linewidth=2)
    axes[0, 1].set_title('Cumulative Mutual Information', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Top-K Features', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Cumulative MI', fontsize=11, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(df['top_k'], df['mean_ig'], color='#3357FF', linewidth=2)
    axes[1, 0].set_title('Mean IG per Feature', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Top-K Features', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Mean IG', fontsize=11, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].bar(df['top_k'], df['ig_gain'].fillna(0), width=40, color='#3357FF', alpha=0.7)
    axes[1, 1].set_title('Incremental IG Gain (per 50)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Top-K Features', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('IG Gain', fontsize=11, fontweight='bold')
    axes[1, 1].grid(alpha=0.3, axis='y')

    plt.tight_layout()

    # Save static PNG
    png_file = output_path / "incremental_ig_analysis_detailed.png"
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"✓ Plot saved to {png_file}")


if __name__ == "__main__":
    # Paths
    parquet_file = "./reports/feature_analysis/feature_rankings_all.parquet"
    output_dir = "./reports/feature_analysis"

    # Check if parquet file exists
    if not Path(parquet_file).exists():
        logger.error(f"Parquet file not found: {parquet_file}")
        logger.error("Run the pipeline first: ./run_complete_pipeline.sh full")
        exit(1)

    # Analyze incremental IG with detailed steps
    incremental_df = analyze_incremental_ig_detailed(parquet_file, output_dir)

    # Plot results (PNG only)
    plot_incremental_analysis(incremental_df, output_dir)

    # Print summary
    print("\n" + "="*80)
    print("DETAILED INCREMENTAL IG ANALYSIS (Growth Phase Focus)")
    print("="*80)
    print(f"Step size: 50 features")
    print(f"Max features analyzed: {incremental_df['top_k'].max()}")
    print(f"\nFirst 10 steps:")
    print(incremental_df.head(10).to_string(index=False))
    print(f"\n...")
    print(f"\nLast 10 steps:")
    print(incremental_df.tail(10).to_string(index=False))
    print("="*80)
    print(f"\nKey Findings:")
    print(f"  - Total cumulative IG (top-{incremental_df['top_k'].max()}): {incremental_df['cumulative_ig'].iloc[-1]:.4f}")
    print(f"  - Average IG per feature: {incremental_df['mean_ig'].iloc[-1]:.6f}")
    print(f"  - Highest gain (per 50): {incremental_df['ig_gain'].max():.4f} at top-{incremental_df.loc[incremental_df['ig_gain'].idxmax(), 'top_k']:.0f}")
    print("="*80)
