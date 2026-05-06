"""Compute per-cluster top-feature consensus candidates for ILP.

Reads `cluster_feature_scores.parquet` (or CSV) produced by the scoring
pipeline and computes, per cluster, aggregated metrics used to select
candidate features to feed into the ILP.

Outputs per-cluster CSV files and a combined summary CSV.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo root on path so imports work when running script directly
workspace_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(workspace_root))

from src.entropy_knn.visualizations.common import METHODS


def _load_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _minmax(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    mn = series.min()
    mx = series.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.0, index=series.index)
    return (series - mn) / (mx - mn)


def compute_consensus(
    scores_df: pd.DataFrame,
    output_dir: Path,
    top_k: int = 5,
    top_n: int = 50,
    normalize: str = "minmax",
) -> None:
    output_dir = Path(output_dir)
    out_consensus_dir = output_dir / "consensus"
    out_consensus_dir.mkdir(parents=True, exist_ok=True)

    methods_present = [m for m in METHODS if m in scores_df.columns]
    if not methods_present:
        # fallback: try to detect any numeric score columns beyond known meta
        numeric_cols = [c for c in scores_df.columns if scores_df[c].dtype.kind in "fi" and c not in ("cluster_id", "feature_rank")]
        methods_present = numeric_cols

    combined_rows: list[dict] = []

    for cid in sorted(scores_df["cluster_id"].unique()):
        cluster_df = scores_df[scores_df["cluster_id"] == cid].copy()
        if cluster_df.empty:
            continue
        cluster_df = cluster_df.reset_index(drop=True)

        # compute ranks and normalized scores per method
        norm_cols = []
        rank_cols = []
        top_sets = {}
        for method in methods_present:
            # safe numeric coercion
            cluster_df[method] = pd.to_numeric(cluster_df[method], errors="coerce").fillna(0.0)
            rank_col = f"{method}_rank"
            norm_col = f"{method}_norm"
            cluster_df[rank_col] = cluster_df[method].rank(method="average", ascending=False)
            rank_cols.append(rank_col)

            if normalize == "minmax":
                cluster_df[norm_col] = _minmax(cluster_df[method])
            elif normalize == "z":
                cluster_df[norm_col] = (cluster_df[method] - cluster_df[method].mean()) / (cluster_df[method].std() if cluster_df[method].std() else 1.0)
            else:
                cluster_df[norm_col] = cluster_df[method]
            norm_cols.append(norm_col)

            # top-k per method
            top_features = cluster_df.sort_values(method, ascending=False)["feature"].head(top_k).astype(str).tolist()
            top_sets[method] = set(top_features)

        # how many methods include a feature in their top-k
        def count_methods_in_top(feature_val: str) -> int:
            return sum(1 for m in methods_present if feature_val in top_sets.get(m, set()))

        cluster_df["method_count"] = cluster_df["feature"].astype(str).apply(count_methods_in_top)

        # aggregated normalized score (mean of normalized method scores)
        if norm_cols:
            cluster_df["aggregated_score"] = cluster_df[norm_cols].sum(axis=1) / max(1, len(norm_cols))
        else:
            cluster_df["aggregated_score"] = 0.0

        # mean/std of ranks
        if rank_cols:
            cluster_df["rank_mean"] = cluster_df[rank_cols].mean(axis=1)
            cluster_df["rank_std"] = cluster_df[rank_cols].std(axis=1)
        else:
            cluster_df["rank_mean"] = cluster_df["feature_rank"].astype(float)
            cluster_df["rank_std"] = 0.0

        # list of methods that included the feature in top-k
        cluster_df["top_methods"] = cluster_df["feature"].astype(str).apply(lambda f: ",".join(sorted([m for m in methods_present if f in top_sets.get(m, set())])))

        # prepare output columns
        out_cols = [
            "cluster_id",
            "feature",
            "aggregated_score",
            "method_count",
            "rank_mean",
            "rank_std",
            "top_methods",
        ]
        # include raw method scores and normalized/rank columns
        for m in methods_present:
            out_cols.append(m)
            out_cols.append(f"{m}_norm")
            out_cols.append(f"{m}_rank")

        out_df = cluster_df[out_cols].sort_values(["method_count", "aggregated_score"], ascending=[False, False])

        # write per-cluster CSV
        out_path = out_consensus_dir / f"cluster_{cid}_candidates.csv"
        out_df.head(top_n).to_csv(out_path, index=False)

        # append to combined
        for _, row in out_df.head(top_n).iterrows():
            combined_rows.append({**{c: row[c] for c in out_cols}, "cluster_id": int(cid)})

    # write combined summary
    combined_df = pd.DataFrame(combined_rows)
    combined_out = output_dir / "top_feature_candidates.csv"
    combined_df.to_csv(combined_out, index=False)

    print(f"[top_feature_consensus] Wrote per-cluster candidates to: {out_consensus_dir}")
    print(f"[top_feature_consensus] Wrote combined summary to: {combined_out}")


def _parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[3]
    default_scores = workspace_root / "reports" / "entropy_knn" / "score_only" / "cluster_500" / "seed_42" / "cluster_feature_scores.parquet"
    parser = argparse.ArgumentParser(description="Compute top-feature consensus candidates for ILP")
    parser.add_argument("--scores", type=Path, default=default_scores, help="Path to cluster_feature_scores.parquet or .csv")
    parser.add_argument("--output-dir", type=Path, default=workspace_root / "reports" / "entropy_knn" / "analysis", help="Output base dir")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K per method to consider for consensus")
    parser.add_argument("--top-n", type=int, default=50, help="Number of candidates to export per cluster")
    parser.add_argument("--normalize", choices=["minmax", "z", "none"], default="minmax", help="Normalization for aggregating scores")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scores_df = _load_scores(args.scores)
    compute_consensus(scores_df=scores_df, output_dir=args.output_dir, top_k=args.top_k, top_n=args.top_n, normalize=args.normalize)


if __name__ == "__main__":
    main()
