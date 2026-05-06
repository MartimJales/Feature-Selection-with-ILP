"""Generate all Entropy KNN comparison visualizations from cluster JSON files.

This script is independent from the score computation pipeline and can be run
any time after cluster JSON score artifacts exist.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.entropy_knn.visualizations import (
    generate_agreement_bars,
    generate_spearman_heatmap,
    generate_top1_scatter_grid,
    generate_venn_grid,
)


def generate_visualizations(cluster_json_dir: Path, output_dir: Path, venn_cluster_json: Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[visualizations] Generating heatmap...", flush=True)
    generate_spearman_heatmap(
        cluster_json_dir=cluster_json_dir,
        output_path=output_dir / "heatmaps" / f"{cluster_json_dir.name}_spearman_heatmap.png",
    )
    print("[visualizations] Heatmap done.", flush=True)

    print("[visualizations] Generating scatter grid...", flush=True)
    generate_top1_scatter_grid(
        cluster_json_dir=cluster_json_dir,
        output_path=output_dir / "scatter" / f"{cluster_json_dir.name}_top1_scatter_grid.png",
    )
    print("[visualizations] Scatter grid done.", flush=True)

    print("[visualizations] Generating agreement bars...", flush=True)
    generate_agreement_bars(
        cluster_json_dir=cluster_json_dir,
        output_path=output_dir / "bars" / f"{cluster_json_dir.name}_agreement_bars.png",
    )
    print("[visualizations] Agreement bars done.", flush=True)

    venn_source = venn_cluster_json
    if venn_source is None:
        cluster_json_files = sorted(cluster_json_dir.glob("cluster_*.json"))
        if cluster_json_files:
            venn_source = cluster_json_files[0]

    if venn_source is not None:
        print(f"[visualizations] Generating venn diagram from {venn_source.name}...", flush=True)
        generate_venn_grid(
            cluster_json_path=venn_source,
            output_path=output_dir / "venn" / f"{venn_source.stem}_top5_venn.png",
            top_k=5,
        )
        print("[visualizations] Venn diagram done.", flush=True)
    else:
        print("[visualizations] Skipping venn diagram (no cluster JSON found).", flush=True)


def _parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[2]
    default_cluster_json_dir = workspace_root / "reports" / "entropy_knn" / "score_only" / "cluster_50" / "seed_42"
    default_output_dir = workspace_root / "reports" / "entropy_knn" / "comparison" / "visualizations"

    parser = argparse.ArgumentParser(description="Generate all Entropy-KNN visualizations from existing cluster JSON scores")
    parser.add_argument("--cluster-json-dir", type=Path, default=default_cluster_json_dir, help="Directory with cluster_*.json")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir, help="Base output directory for generated visualizations")
    parser.add_argument(
        "--venn-cluster-json",
        type=Path,
        default=None,
        help="Optional path to a specific cluster_*.json for the venn chart (default: first available)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_visualizations(
        cluster_json_dir=args.cluster_json_dir,
        output_dir=args.output_dir,
        venn_cluster_json=args.venn_cluster_json,
    )
