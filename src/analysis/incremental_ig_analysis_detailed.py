import pandas as pd
import numpy as np
from pathlib import Path
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    max_k = min(20000, len(df_rankings))

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

def plot_interactive_analysis(df, output_dir="./reports/feature_analysis"):
    """Create interactive plots with Plotly."""
    output_path = Path(output_dir)

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Cumulative Information Gain',
            'Cumulative Mutual Information',
            'Mean IG per Feature',
            'Incremental Gain (per 50 features)'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    # Plot 1: Cumulative IG
    fig.add_trace(
        go.Scatter(
            x=df['top_k'],
            y=df['cumulative_ig'],
            mode='lines+markers',
            name='Cumulative IG',
            line=dict(color='#3357FF', width=2),
            marker=dict(size=4),
            hovertemplate='<b>Top-%{x}</b><br>Cumulative IG: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Plot 2: Cumulative MI
    fig.add_trace(
        go.Scatter(
            x=df['top_k'],
            y=df['cumulative_mi'],
            mode='lines+markers',
            name='Cumulative MI',
            line=dict(color='#FF5733', width=2),
            marker=dict(size=4),
            hovertemplate='<b>Top-%{x}</b><br>Cumulative MI: %{y:.4f}<extra></extra>'
        ),
        row=1, col=2
    )

    # Plot 3: Mean IG
    fig.add_trace(
        go.Scatter(
            x=df['top_k'],
            y=df['mean_ig'],
            mode='lines+markers',
            name='Mean IG',
            line=dict(color='#3357FF', width=2),
            marker=dict(size=4),
            hovertemplate='<b>Top-%{x}</b><br>Mean IG: %{y:.6f}<extra></extra>'
        ),
        row=2, col=1
    )

    # Plot 4: Incremental Gain
    fig.add_trace(
        go.Bar(
            x=df['top_k'],
            y=df['ig_gain'].fillna(0),
            name='IG Gain',
            marker=dict(color='#3357FF', opacity=0.7),
            hovertemplate='<b>Top-%{x}</b><br>Gain: %{y:.4f}<extra></extra>'
        ),
        row=2, col=2
    )

    # Update axes labels
    fig.update_xaxes(title_text="Number of Features (Top-K)", row=1, col=1)
    fig.update_xaxes(title_text="Number of Features (Top-K)", row=1, col=2)
    fig.update_xaxes(title_text="Number of Features (Top-K)", row=2, col=1)
    fig.update_xaxes(title_text="Number of Features (Top-K)", row=2, col=2)

    fig.update_yaxes(title_text="Cumulative IG", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative MI", row=1, col=2)
    fig.update_yaxes(title_text="Mean IG", row=2, col=1)
    fig.update_yaxes(title_text="IG Gain", row=2, col=2)

    # Update layout
    fig.update_layout(
        title_text="Detailed Incremental Feature Analysis (Focus: Growth Phase)",
        title_font_size=16,
        showlegend=False,
        height=800,
        width=1400,
        hovermode='closest'
    )

    # Save interactive HTML
    html_file = output_path / "incremental_ig_analysis_detailed.html"
    fig.write_html(html_file)
    logger.info(f"✓ Interactive plot saved to {html_file}")

    # Also save static PNG
    png_file = output_path / "incremental_ig_analysis_detailed.png"
    fig.write_image(png_file, width=1400, height=800, scale=2)
    logger.info(f"✓ Static plot saved to {png_file}")

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

    # Plot results (interactive + static)
    try:
        plot_interactive_analysis(incremental_df, output_dir)
    except Exception as e:
        logger.warning(f"Could not create interactive plot: {e}")
        logger.info("Install kaleido for static image export: pip install kaleido")
        # Fallback to just HTML
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        output_path = Path(output_dir)
        fig = make_subplots(rows=2, cols=2)
        # ... (same plotting code)
        html_file = output_path / "incremental_ig_analysis_detailed.html"
        fig.write_html(html_file)
        logger.info(f"✓ Interactive HTML saved to {html_file}")

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
    print(f"  - Growth phase captures: ~{incremental_df['cumulative_ig'].iloc[-1]/df_all_ig*100:.1f}% of total IG" if 'df_all_ig' in dir() else "")
    print("="*80)
