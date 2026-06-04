"""Balanced (1:1 malware:goodware) orchestration for Entropy KNN pipeline."""

from __future__ import annotations

import logging
import requests
import time

from dataclasses import replace

import pandas as pd

from src.entropy_knn.pipeline import EntropyKNNPipeline

logger = logging.getLogger(__name__)


class BalancedEntropyKNNPipeline(EntropyKNNPipeline):
    """Entropy KNN pipeline that balances data to a 1:1 class ratio before clustering."""

    def __init__(
        self,
        features_path: str = "./reports/extracted_features.parquet",
        labels_path: str = "./data/training_set.csv",
        rankings_path: str = "./reports/feature_analysis/feature_rankings_all.parquet",
        output_dir: str = "./reports/entropy_knn_balanced",
        balance_seed: int = 42,
        discord_webhook_url: str = "",
        discord_user_id: str = "",
    ) -> None:
        super().__init__(
            features_path=features_path,
            labels_path=labels_path,
            rankings_path=rankings_path,
            output_dir=output_dir,
        )
        self.balance_seed = int(balance_seed)
        self.discord_webhook_url = discord_webhook_url
        self.discord_user_id = discord_user_id

    def _load_bundle(self):
        """Load and cache a balanced bundle with 1:1 class proportion."""
        if self.bundle is None:
            raw_bundle = self.loader.load()
            self.bundle = self._balance_bundle(raw_bundle, self.balance_seed)

            balanced_malware = int((self.bundle.y == 1).sum())
            balanced_goodware = int((self.bundle.y == 0).sum())
            self._send_discord(
                (
                    "📦 Balanced dataset ready\n"
                    f"malware={balanced_malware} | goodware={balanced_goodware} | total={len(self.bundle.y)}\n"
                    f"balance_seed={self.balance_seed}"
                ),
                url=self.discord_webhook_url,
                user_id=self.discord_user_id or None,
            )
        return self.bundle

    @staticmethod
    def _balance_bundle(bundle, random_seed: int):
        y = bundle.y.astype(int)

        malware_mask = y == 1
        goodware_mask = y == 0

        malware_count = int(malware_mask.sum())
        goodware_count = int(goodware_mask.sum())

        if malware_count == 0 or goodware_count == 0:
            raise ValueError(
                "Cannot build 1:1 dataset because one class is missing "
                f"(malware={malware_count}, goodware={goodware_count})"
            )

        target_per_class = min(malware_count, goodware_count)

        malware_indices = y[malware_mask].sample(n=target_per_class, random_state=random_seed).index
        goodware_indices = y[goodware_mask].sample(n=target_per_class, random_state=random_seed).index

        selected_indices = pd.Index(malware_indices.tolist() + goodware_indices.tolist())

        # Shuffle final selection to avoid class blocks.
        selected_indices = pd.Series(selected_indices).sample(frac=1.0, random_state=random_seed).tolist()

        X_balanced = bundle.X.loc[selected_indices].reset_index(drop=True)
        y_balanced = y.loc[selected_indices].reset_index(drop=True)

        logger.info(
            "Balanced dataset ready: malware=%d | goodware=%d | total=%d",
            int((y_balanced == 1).sum()),
            int((y_balanced == 0).sum()),
            len(y_balanced),
        )

        ranked_features = [feature for feature in bundle.ranked_features if feature in X_balanced.columns]

        return replace(
            bundle,
            X=X_balanced,
            y=y_balanced,
            ranked_features=ranked_features,
            original_indices=[int(index) for index in selected_indices],
        )

    @staticmethod
    def _send_discord(msg: str, url: str, user_id: str | None = None) -> None:
        """Send a Discord notification message via webhook."""
        if not url:
            return
        mention = f"<@{user_id}> " if user_id else ""
        content = f"{mention}{msg}"

        max_retries = 3
        backoff = 2.0

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(url, json={"content": content}, timeout=15)
                if response.status_code in (200, 204):
                    logger.info("Discord notification sent (attempt %d)", attempt)
                    return
                logger.warning(
                    "Discord notification attempt %d failed: %s - %s",
                    attempt,
                    response.status_code,
                    response.text,
                )
            except Exception as exc:
                logger.warning("Discord notification attempt %d exception: %s", attempt, exc)

            if attempt < max_retries:
                try:
                    time.sleep(backoff * attempt)
                except Exception:
                    pass

        logger.error("Discord notification failed after %d attempts", max_retries)

    def _run_score_only_configuration(
        self,
        cluster_size: int,
        top_features_global: int,
        seed: int,
        n_clusters: int,
        scale_features: bool,
    ) -> dict:
        """Override to add per-cluster Discord notifications."""
        self._send_discord(
            (
                "🧪 Balanced score-only config started\n"
                f"cluster_size={cluster_size} | seed={seed} | n_clusters={n_clusters} | top_features_global={top_features_global}"
            ),
            url=self.discord_webhook_url,
            user_id=self.discord_user_id or None,
        )

        result = super()._run_score_only_configuration(
            cluster_size=cluster_size,
            top_features_global=top_features_global,
            seed=seed,
            n_clusters=n_clusters,
            scale_features=scale_features,
        )

        # Send Discord notification for this configuration
        if result["status"] == "ok":
            self._send_discord(
                f"✅ Balanced (1:1) score-only config completed: "
                f"cluster_size={cluster_size}, seed={seed}, "
                f"n_clusters={result.get('n_clusters_valid', 0)}, "
                f"runtime={result.get('runtime_seconds', 0):.1f}s",
                url=self.discord_webhook_url,
                user_id=self.discord_user_id or None,
            )
        elif result["status"] == "error":
            self._send_discord(
                f"❌ Balanced (1:1) score-only config FAILED: "
                f"cluster_size={cluster_size}, seed={seed}, "
                f"error={result.get('error_message', 'unknown')[:200]}",
                url=self.discord_webhook_url,
                user_id=self.discord_user_id or None,
            )

        return result
