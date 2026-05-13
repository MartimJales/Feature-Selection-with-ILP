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
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Defaults
DEFAULT_TIMEOUT = 600  # 10 minutes per cluster
DEFAULT_TOP_N = 30
PADTAI_PATH = Path(__file__).parent.parent.parent.parent / "PADTAI" / "padtai.py"


def resolve_cluster_base_dir(explicit_dir: Path | None = None) -> Path:
    """Resolve the cluster base directory from an explicit path or repo-relative defaults."""
    candidates = []

    if explicit_dir is not None:
        candidates.append(explicit_dir)

    repo_root = Path(__file__).resolve().parents[3]
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
    """Extract top-N features from cluster CSV."""
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
        return df.head(top_n)[['feature']].values.flatten().tolist()
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def run_padtai(input_file: Path, output_dir: Path, timeout: int = DEFAULT_TIMEOUT):
    """
    Execute PADTAI with given input and timeout.
    Returns (success: bool, stdout: str, stderr: str)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        "python3",
        str(PADTAI_PATH),
        str(input_file),
        "--grounded", "none",
        "--timeout", str(timeout)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 30  # Allow 30s overhead
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "PADTAI execution timed out"
    except Exception as e:
        return False, "", str(e)

def run_ilp_cluster(cluster_dir: Path, top_n: int, timeout: int, dry_run: bool = False):
    """
    Run ILP for a single cluster.
    Returns dict with metadata and results.
    """
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
        "features": None,
        "padtai_stdout": "",
        "padtai_stderr": "",
        "error": None
    }

    # Extract features
    features = extract_top_features(csv_path, top_n)
    if not features:
        result["status"] = "error"
        result["error"] = f"Could not extract features from {csv_path}"
        return result

    result["features"] = features
    print(f"[cluster_{cluster_id}] Extracted {len(features)} features")

    if dry_run:
        result["status"] = "dry_run"
        return result

    # Format for PADTAI (simple CSV with features)
    temp_input = ilp_output_dir / "input_features.csv"
    try:
        df = pd.DataFrame({"feature": features})
        df.to_csv(temp_input, index=False)
        print(f"[cluster_{cluster_id}] Formatted input: {temp_input}")
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Error formatting input: {e}"
        return result

    # Run PADTAI
    print(f"[cluster_{cluster_id}] Starting PADTAI (timeout {timeout}s)...")
    success, stdout, stderr = run_padtai(temp_input, ilp_output_dir, timeout)

    result["padtai_stdout"] = stdout[:500] if stdout else ""  # Truncate
    result["padtai_stderr"] = stderr[:500] if stderr else ""
    result["status"] = "success" if success else "failed"

    if success:
        print(f"[cluster_{cluster_id}] PADTAI succeeded")
    else:
        print(f"[cluster_{cluster_id}] PADTAI failed: {stderr[:100]}")

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
    for cluster_dir in cluster_dirs:
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

        print(f"[cluster_{result['cluster_id']}] Metadata saved to {metadata_file}")
        print()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        print(f"cluster_{r['cluster_id']:03d}: {r['status'].upper():12s} ({len(r['features'] or [])} features)")

    successful = sum(1 for r in results if r["status"] == "success")
    print(f"\nSuccessful: {successful}/{len(results)}")

    # Discord notification: completion
    summary_msg = f"✅ ILP Runner completed: {successful}/{len(results)} successful"
    if successful < len(results):
        summary_msg = f"⚠️ ILP Runner completed with issues: {successful}/{len(results)} successful"

    send_discord(
        summary_msg,
        url=args.discord_webhook_url,
        user_id=args.discord_user_id or None,
    )

if __name__ == "__main__":
    main()
