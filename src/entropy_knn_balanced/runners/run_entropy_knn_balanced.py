#!/usr/bin/env python3
"""Balanced 1:1 sweep runner for the Entropy KNN pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.entropy_knn_balanced.pipeline import BalancedEntropyKNNPipeline


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def send_discord(msg: str, url: str, user_id: str | None = None) -> None:
    if not url:
        return
    mention = f"<@{user_id}> " if user_id else ""
    content = f"{mention}{msg}"

    logger = logging.getLogger(__name__)
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


def main() -> None:
    _load_env_file(Path(__file__).resolve().parents[3] / ".env")

    parser = argparse.ArgumentParser(description="Balanced Entropy KNN sweep runner (1:1 malware:goodware)")
    parser.add_argument("--mode", choices=["score-only", "threshold-sweep"], default="score-only")
    parser.add_argument("--features-path", default="./reports/extracted_features.parquet")
    parser.add_argument("--labels-path", default="./data/training_set.csv")
    parser.add_argument("--rankings-path", default="./reports/feature_analysis/feature_rankings_all.parquet")
    parser.add_argument("--output-dir", default="./reports/entropy_knn_balanced")
    parser.add_argument("--cluster-sizes", default="500,250,150,100,50,30")
    parser.add_argument("--thresholds", default="0.5,0.6")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--top-features-global", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--base-n-clusters", type=int, default=100)
    parser.add_argument("--cluster-schedule", choices=["inverse-size", "fixed"], default="inverse-size")
    parser.add_argument("--scale", action="store_true")
    parser.add_argument("--balance-seed", type=int, default=42)
    parser.add_argument("--discord-webhook-url", default=os.getenv("DISCORD_WEBHOOK_URL", ""))
    parser.add_argument("--discord-user-id", default=os.getenv("DISCORD_USER_ID", ""))
    args = parser.parse_args()

    log_dir = Path(__file__).resolve().parents[1] / "logs" / "entropy_knn_balanced"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline_debug.log"

    handlers: list[logging.Handler] = [logging.FileHandler(log_file)]
    if sys.stdout.isatty():
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )

    pipeline = BalancedEntropyKNNPipeline(
        features_path=args.features_path,
        labels_path=args.labels_path,
        rankings_path=args.rankings_path,
        output_dir=args.output_dir,
        balance_seed=args.balance_seed,
    )

    cluster_sizes = _parse_csv_ints(args.cluster_sizes)
    seeds = _parse_csv_ints(args.seeds)

    try:
        if args.mode == "score-only":
            score_df = pipeline.run_score_sweep(
                cluster_sizes=cluster_sizes,
                top_features_global=args.top_features_global,
                seeds=seeds,
                scale_features=args.scale,
                base_n_clusters=args.base_n_clusters,
                cluster_schedule=args.cluster_schedule,
            )
            logging.getLogger(__name__).info("Balanced score-only sweep completed: %d runs", len(score_df))
            send_discord(
                msg=f"Entropy KNN balanced ({args.mode}) terminou com sucesso. Runs: {len(score_df)}",
                url=args.discord_webhook_url,
                user_id=args.discord_user_id or None,
            )
        else:
            run_df, cluster_df = pipeline.run_sweep(
                cluster_sizes=cluster_sizes,
                thresholds=_parse_csv_floats(args.thresholds),
                top_features_global=args.top_features_global,
                top_k=args.top_k,
                seeds=seeds,
                scale_features=args.scale,
                base_n_clusters=args.base_n_clusters,
                cluster_schedule=args.cluster_schedule,
            )
            logging.getLogger(__name__).info(
                "Balanced sweep completed: %d runs, %d clusters", len(run_df), len(cluster_df)
            )
            send_discord(
                msg=(
                    f"Entropy KNN balanced ({args.mode}) terminou com sucesso. "
                    f"Runs: {len(run_df)} | Clusters: {len(cluster_df)}"
                ),
                url=args.discord_webhook_url,
                user_id=args.discord_user_id or None,
            )
    except Exception as exc:
        send_discord(
            msg=f"Entropy KNN balanced ({args.mode}) falhou: {exc}",
            url=args.discord_webhook_url,
            user_id=args.discord_user_id or None,
        )
        raise


if __name__ == "__main__":
    main()
