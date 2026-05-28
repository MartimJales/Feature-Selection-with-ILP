#!/usr/bin/env python3
"""Run PADTAI directly on the local sandbox cluster inputs.

This script only reads from `sandbox/cluster_0` and `sandbox/cluster_1` and
writes PADTAI outputs back into `sandbox/`.

It applies the same preprocessing steps the pipeline uses before calling PADTAI:
- load CSV
- drop rows with any NaN
- sanitize feature names (except `label`)
- coerce feature columns to int when possible
- coerce `label` to int (fallback numeric->fillna->int)
"""

from __future__ import annotations

import argparse
import subprocess
import re
from pathlib import Path
import hashlib
import time
import traceback

import pandas as pd


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

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PADTAI_PATH.parent),
            capture_output=True,
            text=True,
            timeout=timeout + 30,
        )
        return result
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        stderr = (e.stderr or "") + f"\n[ERROR] PADTAI timed out after {timeout + 30}s\n"
        return subprocess.CompletedProcess(cmd, returncode=124, stdout=stdout, stderr=stderr)
    except Exception as e:
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr=str(e))


def sanitize_feature_name(name: str) -> str:
    sanitized = re.sub(r'[^A-Za-z0-9_]', '_', str(name))
    if sanitized and sanitized[0].isdigit():
        sanitized = '_' + sanitized
    if not sanitized:
        sanitized = '_unknown_'
    return sanitized


def preprocess_padtai_input(cluster_dir: Path) -> Path:
    """Load cluster CSV, apply pipeline-like preprocessing, save preprocessed CSV and return its path."""
    src = cluster_dir / "padtai_input.csv"
    dst = cluster_dir / "padtai_input.prepared.csv"
    df = pd.read_csv(src)

    # Logging info about the raw input
    with open(cluster_dir / "run.log", "a", encoding="utf-8") as rl:
        rl.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - preprocess: loaded {src} size={src.stat().st_size} bytes\n")
        rl.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - preprocess: shape before dropna: {df.shape}\n")
        rl.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - preprocess: dtypes before: {dict(df.dtypes)}\n")

    # Pipeline uses dropna() before writing the CSV
    df = df.dropna()

    with open(cluster_dir / "run.log", "a", encoding="utf-8") as rl:
        rl.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - preprocess: shape after dropna: {df.shape}\n")

    # Sanitize feature column names (leave 'label' untouched)
    cols = list(df.columns)
    rename_map = {c: sanitize_feature_name(c) for c in cols if c != 'label'}
    if rename_map:
        with open(cluster_dir / "run.log", "a", encoding="utf-8") as rl:
            rl.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - preprocess: rename_map sample: {dict(list(rename_map.items())[:10])}\n")
        df = df.rename(columns=rename_map)

    # Coerce feature columns to ints where possible
    feature_cols = [c for c in df.columns if c != 'label']
    for c in feature_cols:
        try:
            df[c] = df[c].astype(int)
        except Exception:
            # leave as-is
            pass

    # Ensure label is integer
    if 'label' in df.columns:
        try:
            df['label'] = df['label'].astype(int)
        except Exception:
            df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)

    # write a small sample and checksum for quick inspection
    sample_csv = cluster_dir / "padtai_input.sample.csv"
    df.head(5).to_csv(sample_csv, index=False)
    sha256 = hashlib.sha256(df.to_csv(index=False).encode('utf-8')).hexdigest()
    with open(cluster_dir / "run.log", "a", encoding="utf-8") as rl:
        rl.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - preprocess: wrote sample to {sample_csv.name}, sha256={sha256}\n")

    df.to_csv(dst, index=False)
    return dst


def run_cluster(cluster_id: int, timeout: int) -> int:
    cluster_dir = SANDBOX_DIR / f"cluster_{cluster_id}"
    input_file = cluster_dir / "padtai_input.csv"
    if not input_file.exists():
        print(f"[cluster_{cluster_id}] missing input: {input_file}")
        return 1

    # Preprocess input to match pipeline behavior
    try:
        prepared = preprocess_padtai_input(cluster_dir)
    except Exception:
        with open(cluster_dir / "run.log", "a", encoding="utf-8") as rl:
            rl.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR: preprocessing failed:\n{traceback.format_exc()}\n")
        return 1

    output_dir = cluster_dir / "padtai_output"

    cmd_preview = f"python3 {PADTAI_PATH} {prepared} --out {output_dir}"
    with open(cluster_dir / "run.log", "a", encoding="utf-8") as rl:
        rl.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - running PADTAI cmd: {cmd_preview}\n")
        rl.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - PADTAI cwd: {PADTAI_PATH.parent}\n")
    try:
        result = run_padtai(prepared, output_dir, timeout)
    except Exception:
        with open(cluster_dir / "run.log", "a", encoding="utf-8") as rl:
            rl.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR: run_padtai raised exception:\n{traceback.format_exc()}\n")
        return 1

    (cluster_dir / "padtai_stdout.log").write_text(result.stdout or "", encoding="utf-8")
    (cluster_dir / "padtai_stderr.log").write_text(result.stderr or "", encoding="utf-8")
    (cluster_dir / "padtai_returncode.txt").write_text(f"{result.returncode}\n", encoding="utf-8")

    with open(cluster_dir / "run.log", "a", encoding="utf-8") as rl:
        rl.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - PADTAI returncode: {result.returncode}\n")
        rl.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - PADTAI stdout (truncated 1024 chars):\n{(result.stdout or '')[:1024]}\n")
        rl.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - PADTAI stderr (truncated 4096 chars):\n{(result.stderr or '')[:4096]}\n")

    status = "OK" if result.returncode == 0 else "FAIL"
    print(f"[cluster_{cluster_id}] {status} -> {output_dir}. See {cluster_dir / 'run.log'}")
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
