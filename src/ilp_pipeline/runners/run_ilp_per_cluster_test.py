#!/usr/bin/env python3
"""
ILP runner for per-cluster feature selection using PADTAI.
Reads top-30 features from each cluster, executes PADTAI with timeout,
saves results and metadata in cluster_i/ilp_results/ subfolder.
"""

import os
import sys
import json
import subprocess
import argparse
import pandas as pd
import requests
import time
import re
from typing import List
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Defaults
DEFAULT_TIMEOUT = 1200  # 20 minutes per cluster
DEFAULT_TOP_N = 30


def find_repo_root() -> Path:
    """Find the repository root by walking up from this script."""
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "PADTAI").exists() and (candidate / "reports").exists():
            return candidate
    return current.parents[3]


REPO_ROOT = find_repo_root()
PADTAI_PATH = REPO_ROOT / "PADTAI" / "padtai.py"


def resolve_cluster_base_dir(explicit_dir: Path | None = None) -> Path:
    """Resolve the cluster base directory from an explicit path or repo-relative defaults."""
    candidates = []

    if explicit_dir is not None:
        candidates.append(explicit_dir)

    repo_root = REPO_ROOT
    candidates.extend(
        [
            repo_root / "reports" / "entropy_knn" / "analysis" / "per_cluster_feature_vs_method",
            Path.cwd() / "reports" / "entropy_knn" / "analysis" / "per_cluster_feature_vs_method",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find cluster base directory. Tried: "
        + ", ".join(str(path) for path in candidates)
    )

def send_discord(msg: str, url: str, user_id: str | None = None) -> None:
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
            else:
                logger.warning(
                    "Discord notification attempt %d failed: %s - %s",
                    attempt,
                    response.status_code,
                    response.text,
                )
        except Exception as exc:
            logger.warning("Discord notification attempt %d exception: %s", attempt, exc)

        # backoff before next attempt
        if attempt < max_retries:
            try:
                time.sleep(backoff * attempt)
            except Exception:
                pass

    logger.error("Discord notification failed after %d attempts", max_retries)

def get_cluster_dirs(base_dir: Path, cluster_ids: list = None):
    """List cluster directories, optionally filtered by IDs."""
    cluster_dirs = sorted(
        [d for d in base_dir.glob("cluster_*") if d.is_dir()],
        key=lambda x: int(x.name.split("_")[1])
    )
    if cluster_ids:
        cluster_dirs = [d for d in cluster_dirs if int(d.name.split("_")[1]) in cluster_ids]
    return cluster_dirs

def extract_top_features(csv_path: Path, top_n: int = DEFAULT_TOP_N):
    """Extract top-N feature names from cluster CSV."""
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
        # Extract feature names from first column
        features = df.iloc[:top_n, 0].tolist()
        return features
    except Exception as e:
        logger.error(f"Error reading {csv_path}: {e}")
        return None


def sanitize_feature_name(name: str) -> str:
    """
    Convert feature name to valid Prolog identifier.
    Replaces invalid characters with underscores.
    """
    # Replace invalid characters with underscores
    # Valid Prolog identifiers: start with lowercase/underscore, contain alphanumeric + underscore
    sanitized = re.sub(r'[^A-Za-z0-9_]', '_', name)

    # Ensure it starts with a letter or underscore (not a digit)
    if sanitized and sanitized[0].isdigit():
        sanitized = '_' + sanitized

    # Avoid empty strings
    if not sanitized:
        sanitized = '_unknown_'

    return sanitized


def is_probable_prolog_rule(line: str) -> bool:
    """Return True for lines that look like a valid Prolog rule."""
    candidate = line.strip()
    if not candidate or ':-' not in candidate:
        return False

    # Ignore obvious non-rules / artefacts.
    banned_prefixes = (
        'traceback', 'error', 'warning', 'exception', 'time', 'nohup',
        'padtai command', 'loading ', 'saved ', 'metadata ', 'summary',
        'cluster_', 'ilp runner', 'rowp =', 'import ', 'from '
    )
    lower = candidate.lower()
    if lower.startswith(banned_prefixes):
        return False

    if '=' in candidate and ':-' not in candidate.split('=', 1)[0]:
        return False

    # Basic Prolog rule shape: head(args) :- body(args).
    return bool(re.match(r'^[A-Za-z_][A-Za-z0-9_]*\([^()]*\)\s*:-\s*.+', candidate))


def extract_rules_from_output(output: str) -> List[str]:
    """Extract only valid Prolog rules from PADTAI output."""
    rules = []
    seen = set()

    # Prefer explicit rule lines.
    for raw_line in output.splitlines():
        candidate = raw_line.strip()
        if not is_probable_prolog_rule(candidate):
            continue

        # Normalize terminal period while keeping the rule text.
        candidate = candidate.rstrip('.')
        if candidate not in seen:
            seen.add(candidate)
            rules.append(candidate)

    return rules

def run_padtai(input_file: Path, output_dir: Path, timeout: int = DEFAULT_TIMEOUT):
    """
    Execute PADTAI with CSV containing binary features + label column.

    Args:
        input_file: Path to CSV with features (binary 0/1) + 'label' column
        output_dir: Output directory for results
        timeout: Timeout in seconds

    Returns:
        (success: bool, stdout: str, stderr: str)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prolog_artifacts_dir = output_dir / "padtai_artifacts"
    prolog_artifacts_dir.mkdir(parents=True, exist_ok=True)

    # PADTAI command with EXACT arguments per README:
    # --grounded none       → No arithmetic operations
    # --intcols none        → All columns as binary (feature columns)
    # --solver rc2          → SAT solver (stable)
    # --sample-size 100     → Use 100 samples
    # --max-timeout <int>   → Timeout in seconds
    # --debug none          → No debug output
    # --min-coverage 5      → Min coverage 5%
    # --min-recall 10       → Min recall 10%
    # --min-precision 75    → Min precision 75%

    cmd = [
        "python3",
        str(PADTAI_PATH),
        str(input_file),
        "--out", str(prolog_artifacts_dir),
        "--grounded", "none",
        "--solver", "rc2",
        "--sample-size", "100",
        "--max-timeout", str(timeout),
        "--debug", "none",
        "--min-coverage", "5",
        "--min-recall", "10",
        "--min-precision", "75",
    ]

    logger.info(f"PADTAI command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 30,  # Allow 30s overhead for cleanup
            cwd=str(REPO_ROOT / "PADTAI")
        )
        success = result.returncode == 0
        return success, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "PADTAI execution timed out"
    except Exception as e:
        return False, "", str(e)

def run_ilp_cluster(cluster_dir: Path, top_n: int, timeout: int, dry_run: bool = False):
    """
    Run ILP for a single cluster.

    Flow:
    1. Extract top-N feature names from cluster CSV
    2. Load binary feature data from extracted_features.parquet
    3. Load labels from training_set.csv
    4. Merge: keep only top-N features + label column
    5. Save as CSV (features must be 0/1, label in last column)
    6. Execute PADTAI

    Returns:
        dict with metadata and results
    """
    cluster_dir = cluster_dir.resolve()
    cluster_id = int(cluster_dir.name.split("_")[1])

    # Paths
    csv_path = cluster_dir / "top_feature_candidates.csv"
    ilp_output_dir = cluster_dir / "ilp_results"
    ilp_output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "cluster_id": cluster_id,
        "timestamp": datetime.now().isoformat(),
        "top_n": top_n,
        "timeout": timeout,
        "status": "pending",
        "features_selected": None,
        "n_features": 0,
        "n_samples": 0,
        "elapsed_seconds": None,
        "padtai_stdout": "",
        "padtai_stderr": "",
        "error": None
    }

    cluster_start = time.time()

    # Step 1: Extract top-N feature names
    features = extract_top_features(csv_path, top_n)
    if not features:
        result["status"] = "error"
        result["error"] = f"Could not extract features from {csv_path}"
        logger.error(f"[cluster_{cluster_id}] {result['error']}")
        return result

    result["features_selected"] = features
    result["n_features"] = len(features)
    logger.info(f"[cluster_{cluster_id}] ✓ Extracted {len(features)} feature names")

    if dry_run:
        result["status"] = "dry_run"
        return result

    # Step 2: Load extracted features (binary 0/1 values)
    features_file = REPO_ROOT / "reports" / "extracted_features.parquet"
    if not features_file.exists():
        result["status"] = "error"
        result["error"] = f"Features file not found: {features_file}"
        logger.error(f"[cluster_{cluster_id}] {result['error']}")
        return result

    try:
        logger.info(f"[cluster_{cluster_id}] Loading features from parquet...")
        features_df = pd.read_parquet(features_file)
        logger.info(f"[cluster_{cluster_id}] ✓ Loaded {features_df.shape[0]} samples × {features_df.shape[1]} columns")
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Error loading features: {str(e)[:200]}"
        logger.error(f"[cluster_{cluster_id}] {result['error']}")
        return result

    # Step 3: Load training labels
    labels_file = REPO_ROOT / "data" / "training_set.csv"
    if not labels_file.exists():
        result["status"] = "error"
        result["error"] = f"Labels file not found: {labels_file}"
        logger.error(f"[cluster_{cluster_id}] {result['error']}")
        return result

    try:
        logger.info(f"[cluster_{cluster_id}] Loading labels...")
        labels_df = pd.read_csv(labels_file)
        logger.info(f"[cluster_{cluster_id}] ✓ Loaded {len(labels_df)} labels")
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Error loading labels: {str(e)[:200]}"
        logger.error(f"[cluster_{cluster_id}] {result['error']}")
        return result

    # If cluster JSON lists sample indices, subset the datasets to those samples
    cluster_json = cluster_dir / f"cluster_{cluster_id}.json"
    if cluster_json.exists():
        try:
            with open(cluster_json, 'r') as cjf:
                cj = json.load(cjf)
            sample_indices = cj.get('sample_indices')
            if sample_indices:
                max_idx = len(features_df) - 1
                # keep only integer indices within bounds
                valid_indices = [int(i) for i in sample_indices if isinstance(i, int) and 0 <= i <= max_idx]
                if not valid_indices:
                    logger.warning(f"[cluster_{cluster_id}] cluster JSON has no valid sample indices; skipping subsetting")
                else:
                    features_df = features_df.iloc[valid_indices].reset_index(drop=True)
                    labels_df = labels_df.iloc[valid_indices].reset_index(drop=True)
                    logger.info(f"[cluster_{cluster_id}] ✓ Subset to {len(valid_indices)} cluster samples")
        except Exception as e:
            logger.warning(f"[cluster_{cluster_id}] Failed to parse cluster JSON {cluster_json}: {e}; proceeding without subsetting")

            cluster_start = time.time()

    # Step 4: Build final CSV with top-N features + label
    try:
        # Filter features to only those that exist in data
        available_features = [f for f in features if f in features_df.columns]
        missing = set(features) - set(available_features)

        if missing:
            logger.warning(f"[cluster_{cluster_id}] {len(missing)} features not found: {list(missing)[:3]}")

        if not available_features:
            result["status"] = "error"
            result["error"] = "No requested features found in dataset"
            logger.error(f"[cluster_{cluster_id}] {result['error']}")
            return result

        # Build dataset: select features + add label
        final_df = features_df[available_features].copy()

        # Match samples with labels (assume same row order)
        if len(final_df) != len(labels_df):
            logger.warning(f"[cluster_{cluster_id}] Row count mismatch: features={len(final_df)}, labels={len(labels_df)}")
            min_len = min(len(final_df), len(labels_df))
            final_df = final_df.iloc[:min_len]
            labels_df = labels_df.iloc[:min_len]

        # Add label column (PADTAI expects 'label' or last column as target)
        final_df['label'] = labels_df['label'].values

        # Remove any NaN values
        initial_count = len(final_df)
        final_df = final_df.dropna()
        if len(final_df) < initial_count:
            logger.warning(f"[cluster_{cluster_id}] Dropped {initial_count - len(final_df)} NaN rows")

        result["n_samples"] = len(final_df)
        logger.info(f"[cluster_{cluster_id}] ✓ Final dataset: {len(final_df)} samples × {len(available_features)} features + label")

        # Step 5: Sanitize feature names for Prolog compatibility and save to CSV for PADTAI
        final_df_prolog = final_df.copy()
        # Rename columns to valid Prolog identifiers (except 'label')
        rename_map = {col: sanitize_feature_name(col) for col in final_df_prolog.columns if col != 'label'}
        final_df_prolog = final_df_prolog.rename(columns=rename_map)

        padtai_input = ilp_output_dir / "padtai_input.csv"
        final_df_prolog.to_csv(padtai_input, index=False)
        logger.info(f"[cluster_{cluster_id}] ✓ Sanitized {len(rename_map)} feature names and saved input CSV: {padtai_input}")

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Error building dataset: {str(e)[:200]}"
        logger.error(f"[cluster_{cluster_id}] {result['error']}")
        return result

    # Step 6: Run PADTAI
    logger.info(f"[cluster_{cluster_id}] Starting PADTAI (timeout {timeout}s)...")
    success, stdout, stderr = run_padtai(padtai_input, ilp_output_dir, timeout)
    result["elapsed_seconds"] = round(time.time() - cluster_start, 2)

    result["padtai_stdout"] = stdout[:2000] if stdout else ""
    result["padtai_stderr"] = stderr[:2000] if stderr else ""
    result["status"] = "success" if success else "failed"

    # Save full stderr to file for debugging
    if stderr:
        stderr_file = ilp_output_dir / "padtai_stderr.log"
        with open(stderr_file, "w") as f:
            f.write(stderr)
        logger.info(f"[cluster_{cluster_id}] Full stderr → {stderr_file}")

    # Extract rules from PADTAI stdout/stderr and save
    try:
        full_out = (stdout or "") + "\n" + (stderr or "")
        rules = extract_rules_from_output(full_out)
        rules_file = ilp_output_dir / "padtai_rules.json"
        with open(rules_file, "w") as rf:
            json.dump({"n_rules": len(rules), "rules": rules, "elapsed_seconds": result["elapsed_seconds"]}, rf, indent=2)
        logger.info(f"[cluster_{cluster_id}] Saved extracted rules → {rules_file}")

        # Send Discord notification with rules summary
        webhook = os.getenv("DISCORD_WEBHOOK_URL")
        mention_id = os.getenv("DISCORD_USER_ID")
        if webhook:
            if rules:
                top_rules = "\n".join(rules[:10])
                msg = f"✅ PADTAI finished for cluster {cluster_id}: {len(rules)} rules found.\nTop rules:\n{top_rules}"
            else:
                msg = f"✅ PADTAI finished for cluster {cluster_id}: no rules found."
            send_discord(msg, url=webhook, user_id=mention_id or None)
    except Exception as e:
        logger.warning(f"[cluster_{cluster_id}] Failed to extract/send rules: {e}")

    if success:
        logger.info(f"[cluster_{cluster_id}] ✅ PADTAI completed successfully in {result['elapsed_seconds']}s")
    else:
        logger.warning(f"[cluster_{cluster_id}] ❌ PADTAI failed after {result['elapsed_seconds']}s")
        if stderr:
            logger.warning(f"    Error: {stderr[:300]}")

    return result



def main():
    parser = argparse.ArgumentParser(description="Run ILP per cluster (test mode)")
    parser.add_argument(
        "--cluster-dir",
        type=Path,
        default=None,
        help="Base directory containing cluster folders (optional; auto-detected if omitted)"
    )
    parser.add_argument(
        "--cluster-ids",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Cluster IDs to process (default: 0, 1)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of top features to use (default: {DEFAULT_TOP_N})"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"PADTAI timeout per cluster in seconds (default: {DEFAULT_TIMEOUT})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse input but don't execute PADTAI"
    )
    parser.add_argument(
        "--discord-webhook-url",
        default=os.getenv("DISCORD_WEBHOOK_URL", ""),
        help="Discord webhook URL for notifications (env: DISCORD_WEBHOOK_URL)"
    )
    parser.add_argument(
        "--discord-user-id",
        default=os.getenv("DISCORD_USER_ID", ""),
        help="Discord user ID to mention (env: DISCORD_USER_ID)"
    )

    args = parser.parse_args()

    print(f"ILP Runner (test mode)")
    print(f"Base dir: {args.cluster_dir}")
    print(f"Clusters: {args.cluster_ids}")
    print(f"Top-N features: {args.top_n}")
    print(f"Timeout: {args.timeout}s")
    print(f"Dry run: {args.dry_run}")
    print()

    try:
        args.cluster_dir = resolve_cluster_base_dir(args.cluster_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        send_discord(
            f"❌ ILP Runner ERROR: {exc}",
            url=args.discord_webhook_url,
            user_id=args.discord_user_id or None,
        )
        sys.exit(1)

    print(f"Resolved cluster dir: {args.cluster_dir}")

    # Discord notification: start
    send_discord(
        f"🚀 ILP Runner started for clusters {args.cluster_ids}",
        url=args.discord_webhook_url,
        user_id=args.discord_user_id or None,
    )

    if not args.cluster_dir.exists():
        print(f"Error: Cluster dir not found: {args.cluster_dir}")
        send_discord(
            f"❌ ILP Runner ERROR: Cluster dir not found {args.cluster_dir}",
            url=args.discord_webhook_url,
            user_id=args.discord_user_id or None,
        )
        sys.exit(1)

    # Find clusters
    cluster_dirs = get_cluster_dirs(args.cluster_dir, args.cluster_ids)
    if not cluster_dirs:
        print(f"Error: No clusters found matching IDs {args.cluster_ids}")
        send_discord(
            f"❌ ILP Runner ERROR: No clusters found for IDs {args.cluster_ids}",
            url=args.discord_webhook_url,
            user_id=args.discord_user_id or None,
        )
        sys.exit(1)

    # Process each cluster
    results = []
    for idx, cluster_dir in enumerate(cluster_dirs, 1):
        cluster_id = int(cluster_dir.name.split("_")[1])

        # Discord notification: cluster processing started
        send_discord(
            f"🔄 **Cluster {cluster_id}** ({idx}/{len(cluster_dirs)}): Starting feature selection & PADTAI processing...",
            url=args.discord_webhook_url,
            user_id=args.discord_user_id or None,
        )

        result = run_ilp_cluster(
            cluster_dir,
            args.top_n,
            args.timeout,
            dry_run=args.dry_run
        )
        results.append(result)

        # Save metadata
        metadata_file = cluster_dir / "ilp_results" / "ilp_metadata.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_file, "w") as f:
            json.dump(result, f, indent=2)

        # Discord notification: cluster completed with details
        n_features = result.get("n_features", 0)
        n_samples = result.get("n_samples", 0)
        n_rules = len(result.get("padtai_rules", []))
        elapsed = result.get("elapsed_seconds", 0)
        status_emoji = "✅" if result["status"] == "success" else "❌"

        details_msg = f"{status_emoji} **Cluster {cluster_id}** completed"
        if n_features > 0:
            details_msg += f"\n   • Features: {n_features}"
        if n_samples > 0:
            details_msg += f"\n   • Samples: {n_samples}"
        if n_rules > 0:
            details_msg += f"\n   • Rules discovered: {n_rules}"
        if elapsed:
            details_msg += f"\n   • Time: {elapsed:.1f}s"

        send_discord(
            details_msg,
            url=args.discord_webhook_url,
            user_id=args.discord_user_id or None,
        )

        print(f"[cluster_{result['cluster_id']}] Metadata saved to {metadata_file}")
        print()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    successful = sum(1 for r in results if r["status"] == "success")
    total_rules = sum(len(r.get('padtai_rules', [])) for r in results)
    total_features_used = sum(r.get('n_features', 0) for r in results)
    total_samples = sum(r.get('n_samples', 0) for r in results)

    for r in results:
        n_feats = len(r.get('features_selected') or [])
        n_samps = r.get('n_samples', 0)
        n_rules = len(r.get('padtai_rules', []))
        print(f"cluster_{r['cluster_id']:03d}: {r['status'].upper():12s} ({n_feats} features, {n_samps} samples, {n_rules} rules)")

    print(f"\nSuccessful: {successful}/{len(results)}")
    print(f"Total features selected: {total_features_used}")
    print(f"Total samples processed: {total_samples}")
    print(f"Total rules discovered: {total_rules}")

    # Discord notification: completion with detailed summary
    if successful == len(results):
        summary_msg = f"✅ **ILP Runner COMPLETED**\n   • Clusters: {successful}/{len(results)}\n   • Rules discovered: {total_rules}\n   • Features analyzed: {total_features_used}"
    else:
        failed = len(results) - successful
        summary_msg = f"⚠️ **ILP Runner COMPLETED with issues**\n   • Successful: {successful}/{len(results)}\n   • Failed: {failed}\n   • Rules discovered: {total_rules}"

    send_discord(
        summary_msg,
        url=args.discord_webhook_url,
        user_id=args.discord_user_id or None,
    )

if __name__ == "__main__":
    main()
