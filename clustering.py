import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_features_file
from src.features.extractor import JSONFeatureExtractor


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_project_path(path_str: str) -> Path:
    """Resolve absolute paths directly and relative paths from the project root."""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_dataset(json_dir: str, labels_path: str, features_path: str | None = None):
    """Load feature matrix X and labels y, preferring a pre-extracted feature file when available."""
    labels_file = resolve_project_path(labels_path)

    if features_path:
        fp = resolve_project_path(features_path)
        if fp.exists():
            logger.info(f"Loading pre-extracted features from {fp}")
            return load_features_file(str(fp), str(labels_file))
        logger.warning(f"features_path not found: {fp}. Falling back to JSON extraction.")

    json_path = resolve_project_path(json_dir)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_path}")
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")

    extractor = JSONFeatureExtractor(str(json_path))
    return extractor.load_with_labels(str(labels_file))


def load_top_features(ranking_path: str, top_k: int) -> list[str]:
    """Load ranked features and return top-k feature names."""
    ranking_file = resolve_project_path(ranking_path)
    if not ranking_file.exists():
        raise FileNotFoundError(f"Ranking file not found: {ranking_file}")

    if ranking_file.suffix == ".parquet":
        ranking_df = pd.read_parquet(ranking_file)
    elif ranking_file.suffix == ".csv":
        ranking_df = pd.read_csv(ranking_file)
    else:
        raise ValueError(f"Unsupported ranking format: {ranking_file.suffix}")

    required_cols = {"feature"}
    if not required_cols.issubset(set(ranking_df.columns)):
        raise ValueError("Ranking file must contain column: 'feature'")

    if "combined_score" not in ranking_df.columns:
        if {"information_gain", "mutual_information"}.issubset(set(ranking_df.columns)):
            ranking_df["combined_score"] = (
                ranking_df["information_gain"].fillna(0) + ranking_df["mutual_information"].fillna(0)
            ) / 2
        else:
            raise ValueError(
                "Ranking file must contain 'combined_score' or both 'information_gain' and 'mutual_information'."
            )

    ranking_df = ranking_df.sort_values("combined_score", ascending=False)
    return ranking_df["feature"].head(top_k).tolist()


def save_cluster_summary(X_selected: pd.DataFrame, y, clusters, output_dir: Path):
    """Save cluster composition and discriminative feature deltas."""
    summary_df = pd.DataFrame({"label": y, "cluster": clusters})

    composition = (
        summary_df.groupby("cluster")["label"]
        .agg(total="count", malware="sum")
        .reset_index()
    )
    composition["goodware"] = composition["total"] - composition["malware"]
    composition["malware_ratio"] = composition["malware"] / composition["total"]
    composition.to_csv(output_dir / "cluster_composition.csv", index=False)

    cluster_means = X_selected.groupby(clusters).mean()
    if cluster_means.shape[0] == 2:
        deltas = (cluster_means.iloc[0] - cluster_means.iloc[1]).abs().sort_values(ascending=False)
        top_deltas = deltas.head(30).rename("mean_abs_diff").reset_index()
        top_deltas.columns = ["feature", "mean_abs_diff"]
        top_deltas.to_csv(output_dir / "top_discriminative_features.csv", index=False)



def run_clustering(
    json_dir: str,
    labels_path: str,
    ranking_path: str,
    output_dir: str,
    top_k: int = 200,
    n_clusters: int = 2,
    random_state: int = 42,
    features_path: str | None = None,
):
    out = resolve_project_path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Output directory: {out}")

    X, y = load_dataset(json_dir=json_dir, labels_path=labels_path, features_path=features_path)
    logger.info(f"Dataset loaded: {X.shape[0]} samples x {X.shape[1]} features")

    top_features = load_top_features(ranking_path, top_k=top_k)
    selected_features = [f for f in top_features if f in X.columns]

    if len(selected_features) == 0:
        raise ValueError("None of the top-ranked features were found in the dataset columns.")

    if len(selected_features) < top_k:
        logger.warning(f"Only {len(selected_features)}/{top_k} ranked features found in dataset.")

    pd.DataFrame({"feature": selected_features}).to_csv(out / f"selected_features_top{len(selected_features)}.csv", index=False)

    X_selected = X[selected_features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    clusters = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, clusters)
    ari = adjusted_rand_score(y, clusters)

    logger.info(f"Silhouette score: {sil:.4f}")
    logger.info(f"Adjusted Rand Index (vs labels): {ari:.4f}")

    assignment_df = pd.DataFrame({
        "sample_idx": range(len(clusters)),
        "true_label": y,
        "cluster": clusters,
    })
    assignment_df.to_csv(out / "cluster_assignments.csv", index=False)

    metrics_df = pd.DataFrame([
        {
            "top_k": len(selected_features),
            "n_clusters": n_clusters,
            "silhouette": sil,
            "adjusted_rand_index": ari,
        }
    ])
    metrics_df.to_csv(out / "clustering_metrics.csv", index=False)

    save_cluster_summary(X_selected, y, clusters, out)

    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, s=10, alpha=0.8)
    axes[0].set_title(f"K-Means clusters (top-{len(selected_features)})")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=y, s=10, alpha=0.8)
    axes[1].set_title("True labels (post-hoc)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    plt.tight_layout()
    plt.savefig(out / f"clustering_pca_top{len(selected_features)}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"✓ Clustering report saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsupervised clustering with top-k ranked features")
    parser.add_argument("--json-dir", default="data/destino", help="Directory with JSON files")
    parser.add_argument("--labels-path", default="data/training_set.csv", help="Path to labels CSV")
    parser.add_argument(
        "--ranking-path",
        default="reports/feature_analysis/feature_rankings_all.parquet",
        help="Path to feature ranking file (.parquet or .csv)",
    )
    parser.add_argument(
        "--features-path",
        default="reports/extracted_features.parquet",
        help="Optional pre-extracted features file (.parquet or .csv)",
    )
    parser.add_argument("--output-dir", default="reports/clustering_top200", help="Output directory")
    parser.add_argument("--top-k", type=int, default=200, help="Number of top-ranked features to use")
    parser.add_argument("--n-clusters", type=int, default=2, help="Number of clusters for K-Means")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run_clustering(
        json_dir=args.json_dir,
        labels_path=args.labels_path,
        ranking_path=args.ranking_path,
        output_dir=args.output_dir,
        top_k=args.top_k,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        features_path=args.features_path,
    )
