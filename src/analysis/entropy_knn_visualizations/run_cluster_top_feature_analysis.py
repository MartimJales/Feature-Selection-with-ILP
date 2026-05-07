"""Run top-feature analysis one cluster at a time.

This variant mirrors the score pipeline more closely: each `cluster_*.json`
is processed independently and writes its own outputs under a dedicated
cluster folder.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

# ensure repo root on sys.path
workspace_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(workspace_root))

from src.analysis.entropy_knn_visualizations.data_sources import load_scores_for_analysis
from src.analysis.entropy_knn_visualizations.top_feature_consensus import compute_consensus
from src.entropy_knn.visualizations import (
    generate_agreement_bars,
    generate_spearman_heatmap,
    generate_top1_scatter_grid,
    generate_venn_grid,
)


def _parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[3]
    default_json_dir = workspace_root / "reports" / "entropy_knn" / "score_only" / "cluster_500" / "seed_42"
    default_output_dir = workspace_root / "reports" / "entropy_knn" / "analysis" / "per_cluster"

    parser = argparse.ArgumentParser(description="Run per-cluster top-feature analysis")
    parser.add_argument("--cluster-json-dir", type=Path, default=default_json_dir, help="Directory with cluster_*.json")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir, help="Base output directory")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K per method used for consensus and overlap charts")
    parser.add_argument("--top-n", type=int, default=50, help="Number of candidate rows to keep per cluster")
    parser.add_argument("--normalize", choices=["minmax", "z", "none"], default="minmax", help="Normalization for consensus ranking")
    parser.add_argument("--max-clusters", type=int, default=None, help="Optional cap on the number of clusters to process")
    parser.add_argument("--include-scatter", action="store_true", help="Also generate the top-1 scatter grid for each cluster")
    return parser.parse_args()


def _copy_cluster_json(cluster_json_path: Path, cluster_input_dir: Path) -> Path:
    cluster_input_dir.mkdir(parents=True, exist_ok=True)
    copied_path = cluster_input_dir / cluster_json_path.name
    shutil.copy2(cluster_json_path, copied_path)
    return copied_path


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
        "output_dir": str(candidates_path.parent.parent),
    }


def generate_per_cluster_analysis(
    cluster_json_dir: Path,
    output_dir: Path,
    top_k: int = 5,
    top_n: int = 50,
    normalize: str = "minmax",
    max_clusters: int | None = None,
    include_scatter: bool = False,
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
        copied_json_path = _copy_cluster_json(cluster_json_path, cluster_input_dir)

        scores_df = load_scores_for_analysis(cluster_json_dir=cluster_input_dir)
        compute_consensus(
            scores_df=scores_df,
            output_dir=cluster_output_dir,
            top_k=top_k,
            top_n=top_n,
            normalize=normalize,
        )

        cluster_vis_dir.mkdir(parents=True, exist_ok=True)
        generate_agreement_bars(cluster_input_dir, cluster_vis_dir / f"cluster_{cluster_id}_agreement_bars.png", top_k=top_k)
        generate_spearman_heatmap(cluster_input_dir, cluster_vis_dir / f"cluster_{cluster_id}_spearman_heatmap.png")
        generate_venn_grid(copied_json_path, cluster_vis_dir / f"cluster_{cluster_id}_venn_top{top_k}.png", top_k=top_k)

        if include_scatter:
            generate_top1_scatter_grid(cluster_input_dir, cluster_vis_dir / f"cluster_{cluster_id}_top1_scatter_grid.png")

        candidates_path = cluster_output_dir / "top_feature_candidates.csv"
        summary_rows.append(_summarize_candidates(candidates_path, cluster_id, cluster_json_path.name, cluster_output_dir))

    summary_df = pd.DataFrame(summary_rows).sort_values("cluster_id")
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
        include_scatter=args.include_scatter,
    )
    print("[run_cluster_top_feature_analysis] Done.", flush=True)


if __name__ == "__main__":
    main()
