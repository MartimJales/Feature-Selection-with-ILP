#!/usr/bin/env python3
"""Run PADTAI directly on the local sandbox cluster inputs.

This script only reads from `sandbox/cluster_0` and `sandbox/cluster_1` and
writes PADTAI outputs back into `sandbox/`.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


SANDBOX_DIR = Path(__file__).resolve().parent
REPO_ROOT = SANDBOX_DIR.parent
PADTAI_PATH = REPO_ROOT / "PADTAI" / "padtai.py"
DEFAULT_CLUSTERS = (0, 1)
DEFAULT_TIMEOUT = 600


def run_padtai(input_file: Path, output_dir: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3",
        str(PADTAI_PATH),
        str(input_file),
        "--out",
        str(output_dir),
        "--keep-prolog-files",
        "--grounded",
        "none",
        "--intcols",
        "none",
        "--solver",
        "rc2",
        "--sample-size",
        "100",
        "--max-timeout",
        str(timeout),
        "--debug",
        "none",
        "--min-coverage",
        "5",
        "--min-recall",
        "10",
        "--min-precision",
        "75",
    ]

    return subprocess.run(
        cmd,
        cwd=str(PADTAI_PATH.parent),
        capture_output=True,
        text=True,
        timeout=timeout + 30,
    )


def run_cluster(cluster_id: int, timeout: int) -> int:
    cluster_dir = SANDBOX_DIR / f"cluster_{cluster_id}"
    input_file = cluster_dir / "padtai_input.csv"
    if not input_file.exists():
        print(f"[cluster_{cluster_id}] missing input: {input_file}")
        return 1

    output_dir = cluster_dir / "padtai_output"
    result = run_padtai(input_file, output_dir, timeout)

    (cluster_dir / "padtai_stdout.log").write_text(result.stdout, encoding="utf-8")
    (cluster_dir / "padtai_stderr.log").write_text(result.stderr, encoding="utf-8")
    (cluster_dir / "padtai_returncode.txt").write_text(f"{result.returncode}\n", encoding="utf-8")

    status = "OK" if result.returncode == 0 else "FAIL"
    print(f"[cluster_{cluster_id}] {status} -> {output_dir}")
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PADTAI on sandbox cluster inputs.")
    parser.add_argument("--clusters", nargs="*", type=int, default=list(DEFAULT_CLUSTERS))
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    failures = 0
    for cluster_id in args.clusters:
        failures += 1 if run_cluster(cluster_id, args.timeout) != 0 else 0

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
