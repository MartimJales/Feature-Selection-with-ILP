"""Run top-feature analysis one cluster at a time.

This variant focuses on feature-vs-method comparison inside each cluster.
Each `cluster_*.json` is processed independently and writes its own outputs
under a dedicated cluster folder.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ensure repo root on sys.path
workspace_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(workspace_root))

from src.analysis.entropy_knn_visualizations.data_sources import load_scores_for_analysis
from src.entropy_knn.visualizations.common import METHODS, METHOD_LABELS


sns.set_theme(style="whitegrid", context="talk")


def _parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[3]
    default_json_dir = workspace_root / "reports" / "entropy_knn" / "score_only" / "cluster_500" / "seed_42"
    default_output_dir = workspace_root / "reports" / "entropy_knn" / "analysis" / "per_cluster_feature_vs_method"

    parser = argparse.ArgumentParser(description="Run per-cluster top-feature analysis")
    parser.add_argument("--cluster-json-dir", type=Path, default=default_json_dir, help="Directory with cluster_*.json")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir, help="Base output directory")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K threshold used to count feature agreement per method")
    parser.add_argument("--top-n", type=int, default=50, help="Number of candidate rows to keep per cluster")
    parser.add_argument("--normalize", choices=["minmax", "z", "none"], default="minmax", help="Normalization for per-method scores")
    parser.add_argument("--max-clusters", type=int, default=None, help="Optional cap on the number of clusters to process")
    return parser.parse_args()



def _copy_cluster_json(cluster_json_path: Path, cluster_input_dir: Path) -> Path:
    cluster_input_dir.mkdir(parents=True, exist_ok=True)
    copied_path = cluster_input_dir / cluster_json_path.name
    shutil.copy2(cluster_json_path, copied_path)
    return copied_path


def _normalize_scores(score_frame: pd.DataFrame, normalize: str) -> pd.DataFrame:
    normalized_frame = score_frame.copy()
    if normalized_frame.empty or normalize == "none":
        return normalized_frame

    for method in METHODS:
        if method not in normalized_frame.columns:
            continue

        values = normalized_frame[method].astype(float)
        if normalize == "minmax":
            minimum = float(values.min())
            maximum = float(values.max())
            span = maximum - minimum
            normalized_frame[method] = 0.0 if span == 0 else (values - minimum) / span
        elif normalize == "z":
            mean_value = float(values.mean())
            std_value = float(values.std(ddof=0))
            normalized_frame[method] = 0.0 if std_value == 0 else (values - mean_value) / std_value
        else:
            raise ValueError(f"Unsupported normalization: {normalize}")

    return normalized_frame


def _build_feature_summary(score_frame: pd.DataFrame, top_k: int, normalize: str) -> pd.DataFrame:
    if score_frame.empty:
        return pd.DataFrame()

    normalized_frame = _normalize_scores(score_frame, normalize)
    rank_frame = pd.DataFrame({"feature": normalized_frame["feature"].astype(str).tolist()})
    for method in METHODS:
        if method in normalized_frame.columns:
            rank_frame[method] = normalized_frame[method].rank(method="average", ascending=False)

    summary_rows: list[dict] = []
    for row_index, feature in enumerate(normalized_frame["feature"].astype(str).tolist()):
        row_values = normalized_frame.iloc[row_index]
        method_values = {method: float(row_values.get(method, 0.0)) for method in METHODS}
        ranked_methods = sorted(method_values.items(), key=lambda item: item[1], reverse=True)
        top_methods = [method for method, _ in ranked_methods[: min(3, len(ranked_methods))]]

        method_count = 0
        for method in METHODS:
            if method in rank_frame.columns and float(rank_frame.iloc[row_index][method]) <= top_k:
                method_count += 1

        values_series = pd.Series(method_values)
        ranks_series = pd.Series({method: float(rank_frame.iloc[row_index].get(method, 0.0)) for method in METHODS})
        summary_rows.append(
            {
                "feature": feature,
                "method_count": int(method_count),
                "aggregated_score": float(values_series.mean()),
                "score_std": float(values_series.std(ddof=0)),
                "rank_mean": float(ranks_series.mean()),
                "rank_std": float(ranks_series.std(ddof=0)),
                "top_methods": ", ".join(top_methods),
                **method_values,
            }
        )

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame = summary_frame.sort_values(
        by=["method_count", "aggregated_score", "rank_mean", "score_std"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    return summary_frame


def _select_top_features(summary_frame: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if summary_frame.empty:
        return summary_frame
    return summary_frame.head(top_n).copy()


def _feature_method_long_frame(top_features: pd.DataFrame) -> pd.DataFrame:
    if top_features.empty:
        return pd.DataFrame(columns=["feature", "method", "score", "method_label"])

    value_columns = [method for method in METHODS if method in top_features.columns]
    long_frame = top_features[["feature", *value_columns]].melt(
        id_vars="feature",
        value_vars=value_columns,
        var_name="method",
        value_name="score",
    )
    long_frame["method_label"] = long_frame["method"].map(lambda method: METHOD_LABELS.get(method, method))
    return long_frame


def _save_feature_method_heatmap(top_features: pd.DataFrame, output_path: Path, cluster_id: int, normalize: str) -> None:
    if top_features.empty:
        return

    heatmap_data = top_features.set_index("feature")[[method for method in METHODS if method in top_features.columns]]
    plt.figure(figsize=(max(10, len(heatmap_data.columns) * 1.6), max(6, len(heatmap_data) * 0.35)))
    ax = sns.heatmap(
        heatmap_data,
        cmap="viridis",
        annot=heatmap_data.shape[0] <= 20,
        fmt=".2f",
        cbar_kws={"label": f"{normalize} score" if normalize != "none" else "score"},
    )
    ax.set_title(f"Cluster {cluster_id} - Feature vs Method Heatmap")
    ax.set_xlabel("Method")
    ax.set_ylabel("Feature")
    ax.set_xticklabels([METHOD_LABELS.get(method, method) for method in heatmap_data.columns], rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def _save_feature_method_bars(top_features: pd.DataFrame, output_path: Path, cluster_id: int) -> None:
    long_frame = _feature_method_long_frame(top_features)
    if long_frame.empty:
        return

    feature_order = top_features["feature"].tolist()
    plt.figure(figsize=(max(12, len(feature_order) * 0.5), max(6, len(feature_order) * 0.35)))
    ax = sns.barplot(
        data=long_frame,
        x="score",
        y="feature",
        hue="method_label",
        order=feature_order,
        orient="h",
        dodge=True,
    )
    ax.set_title(f"Cluster {cluster_id} - Feature Scores Across Methods")
    ax.set_xlabel("Normalized score")
    ax.set_ylabel("Feature")
    ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def _save_feature_agreement(top_features: pd.DataFrame, output_path: Path, cluster_id: int, top_k: int) -> None:
    if top_features.empty:
        return

    plot_frame = top_features[["feature", "method_count", "aggregated_score"]].sort_values(
        by=["method_count", "aggregated_score"], ascending=[False, False]
    )
    plt.figure(figsize=(max(12, len(plot_frame) * 0.5), max(6, len(plot_frame) * 0.35)))
    ax = sns.barplot(data=plot_frame, x="method_count", y="feature", color="#4c72b0")
    ax.set_title(f"Cluster {cluster_id} - Feature Agreement Across Methods")
    ax.set_xlabel(f"Methods selecting the feature in their top-{top_k}")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def _save_score_spread(top_features: pd.DataFrame, output_path: Path, cluster_id: int) -> None:
    if top_features.empty:
        return

    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(
        data=top_features,
        x="score_std",
        y="aggregated_score",
        size="method_count",
        hue="method_count",
        palette="viridis",
        sizes=(40, 300),
        edgecolor="black",
    )
    for _, row in top_features.iterrows():
        ax.text(float(row["score_std"]) + 0.002, float(row["aggregated_score"]) + 0.002, str(row["feature"]), fontsize=8)
    ax.set_title(f"Cluster {cluster_id} - Feature Score Spread")
    ax.set_xlabel("Score standard deviation across methods")
    ax.set_ylabel("Mean normalized score")
    ax.legend(title="Method count", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def _summarize_candidates(candidates_path: Path, cluster_id: int, cluster_json_name: str, cluster_output_dir: Path) -> dict:
    if not candidates_path.exists():
        return {
            "cluster_id": cluster_id,
            "cluster_json": cluster_json_name,
            "candidate_count": 0,
            "top_feature": None,
            "top_method_count": None,
            "top_aggregated_score": None,
            "output_dir": str(cluster_output_dir),
        }

    candidates_df = pd.read_csv(candidates_path)
    if candidates_df.empty:
        return {
            "cluster_id": cluster_id,
            "cluster_json": cluster_json_name,
            "candidate_count": 0,
            "top_feature": None,
            "top_method_count": None,
            "top_aggregated_score": None,
            "output_dir": str(cluster_output_dir),
        }

    top_row = candidates_df.iloc[0]
    return {
        "cluster_id": cluster_id,
        "cluster_json": cluster_json_name,
        "candidate_count": int(len(candidates_df)),
        "top_feature": str(top_row["feature"]),
        "top_method_count": int(top_row["method_count"]),
        "top_aggregated_score": float(top_row["aggregated_score"]),
        "output_dir": str(cluster_output_dir),
    }


def generate_per_cluster_analysis(
    cluster_json_dir: Path,
    output_dir: Path,
    top_k: int = 5,
    top_n: int = 50,
    normalize: str = "minmax",
    max_clusters: int | None = None,
) -> pd.DataFrame:
    cluster_json_dir = Path(cluster_json_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_json_files = sorted(cluster_json_dir.glob("cluster_*.json"))
    if not cluster_json_files:
        raise FileNotFoundError(f"No cluster_*.json files found in {cluster_json_dir}")

    if max_clusters is not None:
        cluster_json_files = cluster_json_files[:max_clusters]

    summary_rows: list[dict] = []

    for index, cluster_json_path in enumerate(cluster_json_files, start=1):
        cluster_id = int(cluster_json_path.stem.split("_")[-1])
        cluster_output_dir = output_dir / f"cluster_{cluster_id}"
        cluster_input_dir = cluster_output_dir
        cluster_vis_dir = cluster_output_dir / "visualizations"

        print(f"[run_cluster_top_feature_analysis] ({index}/{len(cluster_json_files)}) Processing cluster {cluster_id}...", flush=True)
        _copy_cluster_json(cluster_json_path, cluster_input_dir)

        scores_df = load_scores_for_analysis(cluster_json_dir=cluster_input_dir)
        feature_summary = _build_feature_summary(scores_df, top_k=top_k, normalize=normalize)
        top_features = _select_top_features(feature_summary, top_n=top_n)

        cluster_output_dir.mkdir(parents=True, exist_ok=True)
        cluster_vis_dir.mkdir(parents=True, exist_ok=True)

        feature_summary_path = cluster_output_dir / "feature_method_summary.csv"
        candidates_path = cluster_output_dir / "top_feature_candidates.csv"
        feature_summary.to_csv(feature_summary_path, index=False)
        top_features.to_csv(candidates_path, index=False)

        _save_feature_method_heatmap(
            top_features,
            cluster_vis_dir / f"cluster_{cluster_id}_feature_method_heatmap.png",
            cluster_id=cluster_id,
            normalize=normalize,
        )
        _save_feature_method_bars(
            top_features,
            cluster_vis_dir / f"cluster_{cluster_id}_feature_method_bars.png",
            cluster_id=cluster_id,
        )
        _save_feature_agreement(
            top_features,
            cluster_vis_dir / f"cluster_{cluster_id}_feature_agreement.png",
            cluster_id=cluster_id,
            top_k=top_k,
        )
        _save_score_spread(
            top_features,
            cluster_vis_dir / f"cluster_{cluster_id}_feature_score_spread.png",
            cluster_id=cluster_id,
        )

        summary_rows.append(_summarize_candidates(candidates_path, cluster_id, cluster_json_path.name, cluster_output_dir))

    summary_df = pd.DataFrame(summary_rows).sort_values("cluster_id") if summary_rows else pd.DataFrame()
    summary_path = output_dir / "cluster_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"[run_cluster_top_feature_analysis] Wrote summary to: {summary_path}")
    print(f"[run_cluster_top_feature_analysis] Cluster outputs in: {output_dir}")
    return summary_df



def main() -> None:
    args = _parse_args()
    print("[run_cluster_top_feature_analysis] Starting per-cluster analysis...", flush=True)
    print(f"[run_cluster_top_feature_analysis] Input JSON dir: {args.cluster_json_dir}", flush=True)
    print(f"[run_cluster_top_feature_analysis] Output dir: {args.output_dir}", flush=True)
    generate_per_cluster_analysis(
        cluster_json_dir=args.cluster_json_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
        top_n=args.top_n,
        normalize=args.normalize,
        max_clusters=args.max_clusters,
    )
    print("[run_cluster_top_feature_analysis] Done.", flush=True)



if __name__ == "__main__":
    main()
