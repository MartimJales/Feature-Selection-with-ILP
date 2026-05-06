"""Data loading helpers for top-feature visualizations.

Priority order for current pipeline artifacts:
1) cluster JSON files (`cluster_*.json`) from score-only runs
2) parquet report (`cluster_feature_scores.parquet`)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.entropy_knn.visualizations.common import METHODS, list_cluster_json_files, load_cluster_json


def load_scores_from_json_dir(cluster_json_dir: Path) -> pd.DataFrame:
    json_dir = Path(cluster_json_dir)
    json_files = list_cluster_json_files(json_dir)
    if not json_files:
        raise FileNotFoundError(f"No cluster_*.json files found in {json_dir}")

    rows: list[dict] = []
    for json_file in json_files:
        cluster_data = load_cluster_json(json_file)
        cluster_id = cluster_data.get("cluster_id")
        feature_scores = cluster_data.get("feature_scores", {})
        if not feature_scores:
            continue

        for feature, score_dict in feature_scores.items():
            row = {
                "cluster_id": int(cluster_id),
                "feature": str(feature),
            }
            for method in METHODS:
                row[method] = score_dict.get(method, 0.0)
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["cluster_id", "feature", *METHODS])

    return pd.DataFrame(rows)


def load_scores_from_parquet(scores_path: Path) -> pd.DataFrame:
    path = Path(scores_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported score file format: {path.suffix}")


def load_scores_for_analysis(
    cluster_json_dir: Path | None = None,
    scores_path: Path | None = None,
) -> pd.DataFrame:
    """Load feature scores prioritizing current pipeline JSON artifacts.

    If `cluster_json_dir` is provided and contains cluster JSONs, they are used.
    Otherwise, falls back to `scores_path` (parquet/csv).
    """
    if cluster_json_dir is not None:
        json_dir = Path(cluster_json_dir)
        if json_dir.exists():
            json_files = list_cluster_json_files(json_dir)
            if json_files:
                return load_scores_from_json_dir(json_dir)

    if scores_path is not None:
        return load_scores_from_parquet(Path(scores_path))

    raise FileNotFoundError("No valid input data source found. Provide --cluster-json-dir or --scores.")
