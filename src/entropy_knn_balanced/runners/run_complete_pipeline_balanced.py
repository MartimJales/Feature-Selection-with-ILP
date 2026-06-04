#!/usr/bin/env python3
"""
Complete balanced 1:1 pipeline runner.
Integrates:
1) global feature filtering + balanced 1:1 clustering,
2) per-cluster consensus feature analysis,
3) PADTAI rule discovery per cluster.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import subprocess
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.analysis.entropy_knn_visualizations.run_cluster_top_feature_analysis import (
    generate_per_cluster_analysis,
)
from src.entropy_knn_balanced.pipeline import BalancedEntropyKNNPipeline
from src.ilp_pipeline.runners.run_ilp_per_cluster_test import (
    get_cluster_dirs,
    run_ilp_cluster_from_data,
    send_discord,
)


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


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
        if key and (key not in os.environ or not os.environ.get(key, "").strip()):
            os.environ[key] = value


def _iter_run_dirs(output_dir: Path, cluster_sizes: list[int], seeds: list[int]) -> list[tuple[int, int, Path]]:
    runs: list[tuple[int, int, Path]] = []
    for cluster_size in cluster_sizes:
        for seed in seeds:
            run_dir = output_dir / "score_only" / f"cluster_{cluster_size}" / f"seed_{seed}"
            runs.append((cluster_size, seed, run_dir))
    return runs


def _analyze_clusters_with_malware(output_dir: Path, pipeline: str = "balanced") -> list[int]:
    """
    Analyze clusters and return list of cluster IDs that contain malware.
    Runs the analyzer script and reads the output file.

    Returns:
        List of cluster IDs with malware, or empty list if analysis fails
    """
    import subprocess

    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "analyze_clusters_with_malware.py"

    if not script_path.exists():
        logging.warning(f"Analyzer script not found: {script_path}")
        return []

    try:
        # Run the analyzer
        result = subprocess.run(
            ["python3", str(script_path), "--pipeline", pipeline],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            logging.warning(f"Analyzer script failed: {result.stderr}")
            return []

        # Read the output file
        output_file = repo_root / "scripts" / f"clusters_with_malware_{pipeline}.txt"
        if not output_file.exists():
            logging.warning(f"Analyzer output file not found: {output_file}")
            return []

        cluster_ids = [int(line.strip()) for line in output_file.read_text().splitlines() if line.strip()]
        return cluster_ids

    except Exception as e:
        logging.warning(f"Failed to analyze clusters: {e}")
        return []


def _analyze_clusters_in_dir(analysis_dir: Path) -> list[int]:
    """
    Analyze clusters in a specific analysis directory and return IDs with malware.
    Looks for padtai_input.csv in each cluster's ilp_results/ subdirectory.

    Returns:
        List of cluster IDs with malware
    """
    clusters_with_malware = []

    if not analysis_dir.exists():
        logging.warning(f"Analysis directory not found: {analysis_dir}")
        return []

    try:
        # Find all cluster directories
        cluster_dirs = sorted(
            [d for d in analysis_dir.glob("cluster_*") if d.is_dir()],
            key=lambda x: int(x.name.split("_")[1])
        )

        for cluster_dir in cluster_dirs:
            cluster_id = int(cluster_dir.name.split("_")[1])
            padtai_input = cluster_dir / "ilp_results" / "padtai_input.csv"

            if not padtai_input.exists():
                continue

            try:
                df = pd.read_csv(padtai_input)
                if 'label' in df.columns:
                    malware_count = int((df['label'] == 1).sum())
                    if malware_count > 0:
                        clusters_with_malware.append(cluster_id)
            except Exception as e:
                logging.debug(f"Error reading cluster {cluster_id}: {e}")
                continue

        return sorted(clusters_with_malware)

    except Exception as e:
        logging.warning(f"Failed to analyze clusters in dir: {e}")
        return []


def _notify(args: argparse.Namespace, msg: str) -> None:
    """Small helper for Discord progress notifications."""
    send_discord(
        msg,
        url=args.discord_webhook_url,
        user_id=args.discord_user_id or None,
    )


def main() -> None:
    _load_env_file(Path(__file__).resolve().parents[3] / ".env")

    parser = argparse.ArgumentParser(
        description="Complete balanced 1:1 pipeline: clustering + PADTAI rule discovery"
    )
    parser.add_argument("--features-path", default="./reports/extracted_features.parquet")
    parser.add_argument("--labels-path", default="./data/training_set.csv")
    parser.add_argument("--rankings-path", default="./reports/feature_analysis/feature_rankings_all.parquet")
    parser.add_argument("--output-dir", default="./reports/entropy_knn_balanced")
    parser.add_argument("--cluster-sizes", default="500")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--top-features-global", type=int, default=1000)
    parser.add_argument("--base-n-clusters", type=int, default=100)
    parser.add_argument("--cluster-schedule", choices=["inverse-size", "fixed"], default="inverse-size")
    parser.add_argument("--scale", action="store_true")
    parser.add_argument("--balance-seed", type=int, default=42)
    parser.add_argument("--consensus-top-k", type=int, default=25)
    parser.add_argument("--consensus-top-n", type=int, default=50)
    parser.add_argument("--consensus-normalize", choices=["minmax", "z", "none"], default="minmax")
    parser.add_argument("--max-clusters", type=int, default=None)
    parser.add_argument("--cluster-ids", type=int, nargs="+", default=None)
    parser.add_argument("--ilp-top-n", type=int, default=30, help="Top N features for PADTAI")
    parser.add_argument("--ilp-timeout", type=int, default=1200, help="PADTAI timeout in seconds")
    parser.add_argument("--ilp-dry-run", action="store_true", help="Prepare ILP inputs but do not execute PADTAI")
    parser.add_argument("--discord-webhook-url", default=os.getenv("DISCORD_WEBHOOK_URL", ""))
    parser.add_argument("--discord-user-id", default=os.getenv("DISCORD_USER_ID", ""))
    args = parser.parse_args()

    # Resolve Discord settings robustly (CLI if provided, else env/.env).
    args.discord_webhook_url = (args.discord_webhook_url or os.getenv("DISCORD_WEBHOOK_URL", "")).strip()
    args.discord_user_id = (args.discord_user_id or os.getenv("DISCORD_USER_ID", "")).strip()

    if args.discord_webhook_url:
        os.environ["DISCORD_WEBHOOK_URL"] = args.discord_webhook_url
    if args.discord_user_id:
        os.environ["DISCORD_USER_ID"] = args.discord_user_id

    log_dir = Path(__file__).resolve().parents[1] / "logs" / "entropy_knn_balanced_full"
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

    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("BALANCED 1:1 COMPLETE PIPELINE: Clustering + PADTAI")
    logger.info("=" * 70)

    cluster_sizes = _parse_csv_ints(args.cluster_sizes)
    seeds = _parse_csv_ints(args.seeds)
    run_dirs = _iter_run_dirs(output_dir=Path(args.output_dir), cluster_sizes=cluster_sizes, seeds=seeds)

    _notify(
        args,
        (
            "🚀 Balanced 1:1 pipeline started\n"
            f"runs={len(run_dirs)} | cluster_sizes={cluster_sizes} | seeds={seeds}\n"
            f"output={args.output_dir}"
        ),
    )

    # Phase 1: Balanced clustering + feature selection
    logger.info("\n[PHASE 1] Running balanced 1:1 entropy KNN clustering...")
    _notify(
        args,
        (
            "🧪 Phase 1/2 started: balanced clustering (score-only)\n"
            f"features_path={args.features_path}\n"
            f"labels_path={args.labels_path}\n"
            f"rankings_path={args.rankings_path}\n"
            f"output_dir={args.output_dir}\n"
            f"cluster_sizes={cluster_sizes} | seeds={seeds} | top_features_global={args.top_features_global}"
        ),
    )

    pipeline = BalancedEntropyKNNPipeline(
        features_path=args.features_path,
        labels_path=args.labels_path,
        rankings_path=args.rankings_path,
        output_dir=args.output_dir,
        balance_seed=args.balance_seed,
        discord_webhook_url=args.discord_webhook_url,
        discord_user_id=args.discord_user_id,
    )

    try:
        _notify(
            args,
            (
                "📌 Feature selection now running\n"
                f"balance_seed={args.balance_seed} | base_n_clusters={args.base_n_clusters} | schedule={args.cluster_schedule}\n"
                "Expect several minutes per configuration depending on cluster size and data volume"
            ),
        )

        score_df = pipeline.run_score_sweep(
            cluster_sizes=cluster_sizes,
            top_features_global=args.top_features_global,
            seeds=seeds,
            scale_features=args.scale,
            base_n_clusters=args.base_n_clusters,
            cluster_schedule=args.cluster_schedule,
        )
        logger.info("Balanced clustering completed: %d runs", len(score_df))
        _notify(
            args,
            (
                "✅ Phase 1/2 completed\n"
                f"balanced runs={len(score_df)}\n"
                f"score-only outputs written to: {Path(args.output_dir) / 'score_only'}"
            ),
        )
    except Exception as exc:
        logger.exception("Balanced clustering failed")
        _notify(args, f"❌ Phase 1/2 FAILED: {exc}")
        sys.exit(1)

    if pipeline.bundle is None:
        logger.error("Balanced bundle was not loaded after Phase 1")
        _notify(args, "❌ Phase 1/2 FAILED: balanced bundle was not loaded")
        sys.exit(1)

    balanced_X = pipeline.bundle.X
    balanced_y = pipeline.bundle.y
    logger.info(
        "Reusing in-memory balanced dataset for ILP: samples=%d | features=%d",
        len(balanced_X),
        len(balanced_X.columns),
    )

    # Phase 2: Consensus analysis + PADTAI rule discovery on balanced clusters
    logger.info("\n[PHASE 2] Running consensus analysis + PADTAI on balanced clusters...")
    _notify(args, "🔍 Phase 2/2 started: consensus + PADTAI")

    output_dir = Path(args.output_dir)

    ilp_results: list[dict] = []
    total_clusters_seen = 0

    for run_index, (cluster_size, seed, cluster_json_dir) in enumerate(run_dirs, start=1):
        logger.info(
            "[RUN %d/%d] cluster_size=%d seed=%d | json_dir=%s",
            run_index,
            len(run_dirs),
            cluster_size,
            seed,
            cluster_json_dir,
        )
        _notify(
            args,
            (
                f"▶️ Run {run_index}/{len(run_dirs)} started\n"
                f"cluster_size={cluster_size} | seed={seed}"
            ),
        )

        if not cluster_json_dir.exists():
            logger.warning("Skipping missing score-only directory: %s", cluster_json_dir)
            _notify(
                args,
                (
                    f"⚠️ Run {run_index}/{len(run_dirs)} skipped\n"
                    f"missing dir: {cluster_json_dir}"
                ),
            )
            continue

        analysis_out_dir = (
            output_dir
            / "analysis"
            / "per_cluster_feature_vs_method"
            / f"cluster_{cluster_size}"
            / f"seed_{seed}"
        )
        analysis_out_dir.mkdir(parents=True, exist_ok=True)

        try:
            summary_df = generate_per_cluster_analysis(
                cluster_json_dir=cluster_json_dir,
                output_dir=analysis_out_dir,
                top_k=args.consensus_top_k,
                top_n=args.consensus_top_n,
                normalize=args.consensus_normalize,
                max_clusters=args.max_clusters,
            )
            logger.info(
                "Consensus analysis complete: %d clusters in %s",
                len(summary_df) if isinstance(summary_df, pd.DataFrame) else 0,
                analysis_out_dir,
            )
            _notify(
                args,
                (
                    f"🧩 Consensus done (run {run_index}/{len(run_dirs)})\n"
                    f"cluster_size={cluster_size} | seed={seed} | clusters={len(summary_df)}"
                ),
            )
        except Exception as exc:
            logger.exception("Consensus analysis failed for cluster_size=%d seed=%d", cluster_size, seed)
            _notify(
                args,
                (
                    "❌ Consensus failed\n"
                    f"cluster_size={cluster_size} | seed={seed}\n"
                    f"error={exc}"
                ),
            )
            continue

        cluster_dirs = get_cluster_dirs(analysis_out_dir, args.cluster_ids)
        if not cluster_dirs:
            logger.warning("No cluster dirs found for ILP in %s", analysis_out_dir)
            _notify(
                args,
                (
                    f"⚠️ No clusters for ILP (run {run_index}/{len(run_dirs)})\n"
                    f"path={analysis_out_dir}"
                ),
            )
            continue

        # Analyze which clusters contain malware and filter for ILP
        logger.info("Analyzing clusters to filter those with malware...")
        clusters_with_malware = _analyze_clusters_in_dir(analysis_out_dir)

        if clusters_with_malware:
            logger.info("Found %d clusters with malware in this run", len(clusters_with_malware))
            _notify(
                args,
                (
                    f"📊 Malware filter complete (run {run_index}/{len(run_dirs)})\n"
                    f"clusters with malware: {len(clusters_with_malware)}"
                ),
            )
            cluster_dirs = [d for d in cluster_dirs if int(d.name.split("_")[1]) in clusters_with_malware]

            if not cluster_dirs:
                logger.warning("No clusters with malware found for ILP - skipping this run")
                _notify(
                    args,
                    (
                        f"⚠️ No clusters with malware (run {run_index}/{len(run_dirs)})\n"
                        f"skipping ILP for this run"
                    ),
                )
                continue
        else:
            logger.warning("Malware analysis failed - will process all clusters as fallback")

        total_clusters_seen += len(cluster_dirs)
        logger.info(
            "Starting ILP for %d clusters (cluster_size=%d seed=%d)",
            len(cluster_dirs),
            cluster_size,
            seed,
        )
        _notify(
            args,
            (
                f"🤖 ILP started (run {run_index}/{len(run_dirs)})\n"
                f"cluster_size={cluster_size} | seed={seed} | total_clusters={len(cluster_dirs)}"
            ),
        )

        for idx, cluster_dir in enumerate(cluster_dirs, start=1):
            logger.info(
                "  [ILP %d/%d] %s",
                idx,
                len(cluster_dirs),
                cluster_dir.name,
            )
            if idx == 1 or idx % 10 == 0 or idx == len(cluster_dirs):
                _notify(
                    args,
                    (
                        f"📍 ILP progress (run {run_index}/{len(run_dirs)})\n"
                        f"cluster {idx}/{len(cluster_dirs)}: {cluster_dir.name}"
                    ),
                )

            result = run_ilp_cluster_from_data(
                cluster_dir=cluster_dir,
                top_n=args.ilp_top_n,
                timeout=args.ilp_timeout,
                features_df=balanced_X,
                labels=balanced_y,
                dry_run=args.ilp_dry_run,
            )
            result["cluster_size"] = cluster_size
            result["seed"] = seed
            ilp_results.append(result)

            metadata_file = cluster_dir / "ilp_results" / "ilp_metadata.json"
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_file, "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)

        run_success = sum(
            1
            for r in ilp_results
            if r.get("cluster_size") == cluster_size and r.get("seed") == seed and r.get("status") == "success"
        )
        run_total = sum(
            1
            for r in ilp_results
            if r.get("cluster_size") == cluster_size and r.get("seed") == seed
        )
        _notify(
            args,
            (
                f"✅ Run {run_index}/{len(run_dirs)} finished\n"
                f"cluster_size={cluster_size} | seed={seed} | ilp_success={run_success}/{run_total}"
            ),
        )

    successful = sum(1 for r in ilp_results if r.get("status") == "success")
    failed = sum(1 for r in ilp_results if r.get("status") == "failed")
    dry_runs = sum(1 for r in ilp_results if r.get("status") == "dry_run")
    errored = sum(1 for r in ilp_results if r.get("status") == "error")

    summary_dir = output_dir / "analysis"
    summary_dir.mkdir(parents=True, exist_ok=True)
    ilp_summary_csv = summary_dir / "balanced_ilp_run_summary.csv"
    pd.DataFrame(ilp_results).to_csv(ilp_summary_csv, index=False)

    logger.info("\n[PHASE 2 SUMMARY]")
    logger.info("Cluster groups processed: %d", len(run_dirs))
    logger.info("Total clusters seen for ILP: %d", total_clusters_seen)
    logger.info(
        "ILP results: success=%d | failed=%d | error=%d | dry_run=%d | total=%d",
        successful,
        failed,
        errored,
        dry_runs,
        len(ilp_results),
    )
    logger.info("ILP summary CSV: %s", ilp_summary_csv)
    _notify(
        args,
        (
            "📊 Phase 2/2 summary\n"
            f"clusters_seen={total_clusters_seen} | total_ilp={len(ilp_results)}\n"
            f"success={successful} | failed={failed} | error={errored} | dry_run={dry_runs}"
        ),
    )

    # Final notification
    _notify(
        args,
        (
            "✅ Balanced 1:1 complete pipeline finished! "
            f"ILP success={successful}/{len(ilp_results)} "
            f"(failed={failed}, error={errored}, dry_run={dry_runs})"
        ),
    )

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("Output dir: %s", args.output_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
