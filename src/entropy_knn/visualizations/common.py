"""Shared helpers for entropy_knn comparison visualizations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = [
    "entropy_reduction_ratio",
    "mutual_information",
    "chi2_stat",
    "f_stat",
    "pearson_r",
]

METHOD_LABELS = {
    "entropy_reduction_ratio": "IG / Entropy Reduction",
    "mutual_information": "Mutual Information",
    "chi2_stat": "Chi-square",
    "f_stat": "F-statistic",
    "pearson_r": "Pearson r",
}


def load_cluster_json(cluster_json_path: Path) -> dict:
    with open(cluster_json_path, encoding="utf-8") as handle:
        return json.load(handle)


def list_cluster_json_files(cluster_json_dir: Path) -> list[Path]:
    return sorted(cluster_json_dir.glob("cluster_*.json"))


def build_score_frame(cluster_data: dict) -> pd.DataFrame:
    feature_scores = cluster_data.get("feature_scores", {})
    if not feature_scores:
        return pd.DataFrame()

    frame = pd.DataFrame.from_dict(feature_scores, orient="index")
    frame.index.name = "feature"
    frame.reset_index(inplace=True)
    return frame


def build_rank_frame(score_frame: pd.DataFrame) -> pd.DataFrame:
    if score_frame.empty:
        return score_frame

    rank_frame = pd.DataFrame({"feature": score_frame["feature"].tolist()})
    for method in METHODS:
        if method not in score_frame.columns:
            continue
        rank_frame[method] = score_frame[method].rank(method="average", ascending=False)
    return rank_frame


def top_k_feature_sets(cluster_data: dict, top_k: int = 5) -> dict[str, set[str]]:
    score_frame = build_score_frame(cluster_data)
    if score_frame.empty:
        return {method: set() for method in METHODS}

    feature_sets: dict[str, set[str]] = {}
    for method in METHODS:
        if method not in score_frame.columns:
            feature_sets[method] = set()
            continue
        top_features = (
            score_frame.sort_values(method, ascending=False)["feature"].head(top_k).astype(str).tolist()
        )
        feature_sets[method] = set(top_features)
    return feature_sets


def average_pairwise(values: list[float]) -> float:
    clean_values = [value for value in values if value is not None and not np.isnan(value)]
    return float(np.mean(clean_values)) if clean_values else float("nan")
