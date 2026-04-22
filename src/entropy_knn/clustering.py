"""KNN clustering helpers for local feature analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class EntropyCluster:
    """One local KNN cluster."""

    cluster_id: int
    anchor_index: int
    row_indices: list[int]

    @property
    def n_samples(self) -> int:
        return len(self.row_indices)


class EntropyKNNClusterer:
    """Build local clusters from the selected global feature space."""

    def __init__(
        self,
        cluster_size: int,
        n_clusters: int,
        scale_features: bool = False,
        random_seed: int = 42,
    ) -> None:
        self.cluster_size = int(cluster_size)
        self.n_clusters = int(n_clusters)
        self.scale_features = bool(scale_features)
        self.random_seed = int(random_seed)

    def cluster(self, X: pd.DataFrame) -> list[EntropyCluster]:
        if X.empty:
            return []

        matrix = self._prepare_matrix(X)
        anchor_indices = self._sample_anchor_indices(len(X))
        knn = self._fit_knn(matrix)

        clusters: list[EntropyCluster] = []
        for cluster_id, anchor_index in enumerate(anchor_indices):
            row_indices = self._get_neighbour_indices(knn, matrix, anchor_index)
            if row_indices:
                clusters.append(
                    EntropyCluster(
                        cluster_id=cluster_id,
                        anchor_index=int(anchor_index),
                        row_indices=row_indices,
                    )
                )

        return clusters

    def _prepare_matrix(self, X: pd.DataFrame) -> np.ndarray:
        matrix = X.to_numpy(copy=True)
        if self.scale_features:
            matrix = StandardScaler().fit_transform(matrix)
        return matrix

    def _sample_anchor_indices(self, n_rows: int) -> np.ndarray:
        n_clusters = min(self.n_clusters, n_rows)
        rng = np.random.default_rng(self.random_seed)
        return rng.choice(n_rows, size=n_clusters, replace=False)

    def _fit_knn(self, matrix: np.ndarray) -> NearestNeighbors:
        n_neighbors = min(self.cluster_size, len(matrix))
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1)
        knn.fit(matrix)
        return knn

    @staticmethod
    def _get_neighbour_indices(knn: NearestNeighbors, matrix: np.ndarray, anchor_index: int) -> list[int]:
        _, indices = knn.kneighbors(matrix[anchor_index].reshape(1, -1), return_distance=True)
        return [int(index) for index in indices[0].tolist()]
