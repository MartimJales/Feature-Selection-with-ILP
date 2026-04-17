import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.idea2.data_loader import Idea2DataLoader

logger = logging.getLogger(__name__)


@dataclass
class KNNClusterResult:
    cluster_id: int
    anchor_index: int
    n_samples: int
    class_0: int
    class_1: int
    ig_sum: float
    ig_top30_sum: float
    ig_top30_ratio: float
    acc_top30: float | None
    f1_top30: float | None
    top30_features: List[str]


class Idea1KNNExperiment:
    def __init__(
        self,
        features_path: str,
        labels_path: str,
        rankings_path: str,
        output_dir: str = "./reports/idea1",
    ):
        self.loader = Idea2DataLoader(features_path, labels_path, rankings_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        top_features: int = 1000,
        n_clusters: int = 100,
        cluster_size: int = 500,
        top_local_features: int = 30,
        min_cluster_rows: int = 150,
        random_seed: int = 42,
        scale_features: bool = False,
    ) -> Dict:
        X, y, rankings = self.loader.load()

        ranked = rankings["feature"].astype(str).tolist()
        ranked_available = [f for f in ranked if f in X.columns]
        selected_features = ranked_available[:top_features]
        if not selected_features:
            raise ValueError("No ranked features available in extracted feature matrix")

        logger.info(
            f"Ranked available features in matrix: {len(ranked_available)} | "
            f"Using top-{len(selected_features)} by global IG"
        )

        X_sel = X[selected_features].copy()
        y = y.reset_index(drop=True)

        if scale_features:
            scaler = StandardScaler()
            X_knn = scaler.fit_transform(X_sel.values)
        else:
            X_knn = X_sel.values

        n_clusters = min(n_clusters, len(X_sel))
        if cluster_size < min_cluster_rows:
            logger.warning(
                f"cluster_size ({cluster_size}) < min_cluster_rows ({min_cluster_rows}); "
                f"using min_cluster_rows"
            )
            cluster_size = min_cluster_rows

        rng = np.random.default_rng(random_seed)
        anchor_indices = rng.choice(len(X_sel), size=n_clusters, replace=False)

        logger.info(f"KNN fit on {len(X_sel)} rows × {len(selected_features)} features")
        logger.info(f"Anchors: {n_clusters} | cluster_size: {cluster_size}")

        knn = NearestNeighbors(n_neighbors=min(cluster_size, len(X_sel)), metric="euclidean", n_jobs=-1)
        knn.fit(X_knn)

        rows: List[KNNClusterResult] = []

        for cluster_id, anchor_idx in enumerate(anchor_indices):
            _, idx = knn.kneighbors(X_knn[anchor_idx].reshape(1, -1), return_distance=True)
            idx = idx[0]

            Xc = X_sel.iloc[idx].reset_index(drop=True)
            yc = y.iloc[idx].reset_index(drop=True)

            if len(Xc) < min_cluster_rows:
                continue

            class_counts = yc.value_counts().to_dict()
            n0 = int(class_counts.get(0, 0))
            n1 = int(class_counts.get(1, 0))

            if yc.nunique() < 2:
                rows.append(
                    KNNClusterResult(
                        cluster_id=cluster_id,
                        anchor_index=int(anchor_idx),
                        n_samples=len(Xc),
                        class_0=n0,
                        class_1=n1,
                        ig_sum=0.0,
                        ig_top30_sum=0.0,
                        ig_top30_ratio=0.0,
                        acc_top30=None,
                        f1_top30=None,
                        top30_features=[],
                    )
                )
                continue

            ig = mutual_info_classif(Xc, yc, random_state=random_seed, discrete_features="auto")
            ig_series = pd.Series(ig, index=Xc.columns).sort_values(ascending=False)

            topk = ig_series.head(top_local_features)
            ig_sum = float(ig_series.sum())
            ig_top = float(topk.sum())
            ig_ratio = float(ig_top / ig_sum) if ig_sum > 0 else 0.0

            acc, f1 = self._evaluate_local_topk(Xc[topk.index], yc, random_seed)

            rows.append(
                KNNClusterResult(
                    cluster_id=cluster_id,
                    anchor_index=int(anchor_idx),
                    n_samples=len(Xc),
                    class_0=n0,
                    class_1=n1,
                    ig_sum=ig_sum,
                    ig_top30_sum=ig_top,
                    ig_top30_ratio=ig_ratio,
                    acc_top30=acc,
                    f1_top30=f1,
                    top30_features=list(topk.index),
                )
            )

            if (cluster_id + 1) % 10 == 0:
                logger.info(f"Processed {cluster_id + 1}/{n_clusters} anchors")

        df = pd.DataFrame([r.__dict__ for r in rows])
        if df.empty:
            raise RuntimeError("No valid clusters were produced")

        out_csv = self.output_dir / "knn_cluster_results.csv"
        df.to_csv(out_csv, index=False)

        summary = {
            "n_clusters_requested": int(n_clusters),
            "n_clusters_valid": int(len(df)),
            "top_features_global": int(len(selected_features)),
            "cluster_size": int(cluster_size),
            "min_cluster_rows": int(min_cluster_rows),
            "top_local_features": int(top_local_features),
            "ig_ratio_mean": float(df["ig_top30_ratio"].fillna(0).mean()),
            "ig_ratio_median": float(df["ig_top30_ratio"].fillna(0).median()),
            "ig_ratio_ge_0_90": int((df["ig_top30_ratio"].fillna(0) >= 0.90).sum()),
            "acc_top30_mean": float(pd.to_numeric(df["acc_top30"], errors="coerce").dropna().mean()),
            "f1_top30_mean": float(pd.to_numeric(df["f1_top30"], errors="coerce").dropna().mean()),
            "output_csv": str(out_csv),
        }

        out_json = self.output_dir / "knn_summary.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"✓ Saved cluster results: {out_csv}")
        logger.info(f"✓ Saved summary: {out_json}")
        logger.info(
            "IG top30 ratio (mean/median): "
            f"{summary['ig_ratio_mean']:.3f}/{summary['ig_ratio_median']:.3f}"
        )

        return summary

    @staticmethod
    def _evaluate_local_topk(X: pd.DataFrame, y: pd.Series, random_seed: int) -> tuple[float | None, float | None]:
        if y.nunique() < 2:
            return None, None

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.3,
                random_state=random_seed,
                stratify=y,
            )
        except ValueError:
            return None, None

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            return None, None

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=2000, random_state=random_seed)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)

        return float(accuracy_score(y_test, y_pred)), float(f1_score(y_test, y_pred))
