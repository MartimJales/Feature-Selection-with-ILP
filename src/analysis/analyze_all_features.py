import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import logging
import matplotlib.pyplot as plt

from src.features.extractor import JSONFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_all_feature_scores(parquet_file, labels_path):
    """Calculate IG and MI for ALL features in the dataset."""

    logger.info("Loading features from Parquet...")
    df = pd.read_parquet(parquet_file)
    logger.info(f"Loaded {df.shape[0]} samples x {df.shape[1]} columns")

    # Load labels
    logger.info("Loading labels...")
    labels_df = pd.read_csv(labels_path)

    # Extract X and y
    X = df.drop(columns=['file_hash'], errors='ignore')
    feature_names = X.columns.tolist()

    # Merge with labels
    df_with_labels = df.merge(
        labels_df[['sha256', 'label']],
        left_on='file_hash',
        right_on='sha256',
        how='inner'
    )

    y = df_with_labels['label'].astype(int).values
    X = df_with_labels[feature_names].values

    logger.info(f"Dataset: {X.shape[0]} samples x {X.shape[1]} features")
    logger.info(f"Label distribution: {np.sum(y)} malicious, {len(y) - np.sum(y)} benign")

    # Calculate MI for all features
    logger.info("Calculating Mutual Information for all features...")
    mi_scores = mutual_info_classif(X, y, random_state=42)

    # Calculate IG manually (entropy-based)
    logger.info("Calculating Information Gain for all features...")
    from sklearn.tree import DecisionTreeClassifier

    ig_scores = []
    for i, feature_name in enumerate(feature_names):
        if (i + 1) % 5000 == 0:
            logger.info(f"  Processed {i+1}/{len(feature_names)}")

        # Simple IG: use feature importance from single-feature tree
        X_single = X[:, i].reshape(-1, 1)
        clf = DecisionTreeClassifier(max_depth=1, random_state=42)
        clf.fit(X_single, y)
        ig_scores.append(clf.feature_importances_[0])

    ig_scores = np.array(ig_scores)

    # Create results DataFrame
    results = pd.DataFrame({
        'feature': feature_names,
        'information_gain': ig_scores,
        'mutual_information': mi_scores
    })

    # Calculate combined score
    results['combined_score'] = (results['information_gain'] + results['mutual_information']) / 2

    # Sort by IG (descending)
    results = results.sort_values('information_gain', ascending=False).reset_index(drop=True)

    logger.info(f"Calculated scores for all {len(results)} features")

    return results, X, y, feature_names

def analyze_incremental_gain(results, X, y, feature_names):
    """Analyze IG gain when adding features in steps of 100."""

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    logger.info("\nAnalyzing incremental gain (step of 100)...")

    step_size = 100
    max_features = min(len(results), 1000)  # Limit to 1000 for computation time

    incremental_results = []

    for top_k in range(step_size, max_features + 1, step_size):
        logger.info(f"  Evaluating top-{top_k} features...")

        # Get top-K features
        top_features = results.iloc[:top_k]['feature'].tolist()
        feature_indices = [feature_names.index(f) for f in top_features]

        X_subset = X[:, feature_indices]

        # Train and evaluate
        clf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1, max_depth=10)
        scores = cross_val_score(clf, X_subset, y, cv=3, scoring='f1')

        mean_score = scores.mean()

        incremental_results.append({
            'top_k': top_k,
            'mean_f1_score': mean_score,
            'std_f1_score': scores.std()
        })

    incremental_df = pd.DataFrame(incremental_results)

    # Calculate gain
    incremental_df['f1_gain'] = incremental_df['mean_f1_score'].diff()

    return incremental_df

if __name__ == "__main__":
    # Paths
    parquet_file = "./reports/extracted_features.parquet"
    labels_path = "./data/training_set.csv"
    output_dir = Path("./reports/feature_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Calculate scores for all features
    results, X, y, feature_names = calculate_all_feature_scores(parquet_file, labels_path)

    # Save all rankings
    all_rankings_file = output_dir / "feature_rankings_all.csv"
    results.to_csv(all_rankings_file, index=False)
    logger.info(f"✓ All feature rankings saved to {all_rankings_file}")

    # Step 2: Analyze incremental gain
    incremental_df = analyze_incremental_gain(results, X, y, feature_names)

    # Save incremental analysis
    incremental_file = output_dir / "incremental_gain_analysis.csv"
    incremental_df.to_csv(incremental_file, index=False)
    logger.info(f"✓ Incremental gain analysis saved to {incremental_file}")

    # Step 3: Plot incremental gain
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: F1 Score vs Top-K
    ax1.plot(incremental_df['top_k'], incremental_df['mean_f1_score'], 'o-', linewidth=2, markersize=6)
    ax1.fill_between(
        incremental_df['top_k'],
        incremental_df['mean_f1_score'] - incremental_df['std_f1_score'],
        incremental_df['mean_f1_score'] + incremental_df['std_f1_score'],
        alpha=0.2
    )
    ax1.set_xlabel('Number of Features (Top-K)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('F1 Score (Cross-Validation)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance vs Feature Count', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)

    # Plot 2: Incremental Gain
    ax2.bar(incremental_df['top_k'], incremental_df['f1_gain'].fillna(0), width=80, alpha=0.7, color='#3357FF')
    ax2.set_xlabel('Number of Features (Top-K)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1 Score Gain', fontsize=12, fontweight='bold')
    ax2.set_title('Incremental Gain per 100 Features', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')

    plt.tight_layout()

    # Save plots
    plt.savefig(output_dir / "incremental_gain_analysis.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "incremental_gain_analysis.pdf", bbox_inches='tight')
    logger.info(f"✓ Plots saved to {output_dir}/incremental_gain_analysis.{{png,pdf}}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(incremental_df.to_string(index=False))
    print("="*60)
