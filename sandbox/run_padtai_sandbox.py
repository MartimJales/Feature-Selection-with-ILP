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
from collections import Counter
import shutil

import json
import pandas as pd


SANDBOX_DIR = Path(__file__).resolve().parent
REPO_ROOT = SANDBOX_DIR.parent
PADTAI_PATH = REPO_ROOT / "PADTAI" / "padtai.py"
DEFAULT_CLUSTERS = (0, 1)
DEFAULT_TIMEOUT = 600

# Global output directory (set in main)
OUTPUT_DIR: Path = None


def is_probable_prolog_rule(line: str) -> bool:
    """Return True for lines that look like a valid Prolog rule."""
    candidate = line.strip()
    if not candidate or ':-' not in candidate:
        return False

    # Ignore obvious non-rules / artifacts.
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


def extract_rules_from_output(output: str) -> list[str]:
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


def run_padtai(input_file: Path, output_dir: Path, timeout: int, intcols: str = "none", grounded: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grounded_arg = ",".join(grounded) if grounded else "none"
    cmd = [
        "python3",
        str(PADTAI_PATH),
        str(input_file),
        "--out",
        str(output_dir),
        "--grounded",
        grounded_arg,
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
        def _to_text(value: object) -> str:
            if value is None:
                return ""
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return str(value)

        stdout = _to_text(e.stdout)
        stderr = _to_text(e.stderr) + f"\n[ERROR] PADTAI timed out after {timeout + 30}s\n"
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


def resolve_intcols_arg(prepared_csv: Path, intcols: str) -> str:
    """Convert sandbox intcols modes into PADTAI-compatible CSV index lists."""
    if not intcols or intcols == "none":
        return "none"

    if intcols != "auto":
        return intcols

    df = pd.read_csv(prepared_csv)
    feature_indices: list[str] = []
    for index, column in enumerate(df.columns):
        if column == "label":
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        unique_values = series.dropna().nunique()
        if unique_values > 2:
            feature_indices.append(str(index))

    return ",".join(feature_indices) if feature_indices else "none"


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


def run_cluster(cluster_id: int, timeout: int, intcols: str = "none", grounded: list[str] | None = None) -> int:
    global OUTPUT_DIR

    # Cluster dir is now in OUTPUT_DIR
    cluster_dir = OUTPUT_DIR / f"cluster_{cluster_id}"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    # Source input from original sandbox location
    src_input = SANDBOX_DIR / f"cluster_{cluster_id}" / "padtai_input.csv"
    if not src_input.exists():
        print(f"[cluster_{cluster_id}] missing input: {src_input}")
        return 1

    # Copy input to output dir
    input_file = cluster_dir / "padtai_input.csv"
    shutil.copy(src_input, input_file)

    # Preprocess input to match pipeline behavior
    try:
        prepared = preprocess_padtai_input(cluster_dir)
    except Exception:
        with open(cluster_dir / "run.log", "a", encoding="utf-8") as rl:
            rl.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR: preprocessing failed:\n{traceback.format_exc()}\n")
        return 1

    output_dir = cluster_dir / "padtai_output"
    resolved_intcols = resolve_intcols_arg(prepared, intcols)

    grounded_preview = f" --grounded {','.join(grounded)}" if grounded else " --grounded none"
    cmd_preview = f"python3 {PADTAI_PATH} {prepared} --out {output_dir} --intcols {resolved_intcols}{grounded_preview}"
    with open(cluster_dir / "run.log", "a", encoding="utf-8") as rl:
        rl.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - running PADTAI cmd: {cmd_preview}\n")
        rl.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - PADTAI cwd: {PADTAI_PATH.parent}\n")
        rl.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - intcols: {intcols} -> {resolved_intcols}\n")

    # Store start time for elapsed_seconds calculation
    start_time = time.time()

    try:
        result = run_padtai(prepared, output_dir, timeout, intcols=resolved_intcols, grounded=grounded)
    except Exception:
        with open(cluster_dir / "run.log", "a", encoding="utf-8") as rl:
            rl.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR: run_padtai raised exception:\n{traceback.format_exc()}\n")
        return 1

    elapsed_seconds = time.time() - start_time

    (cluster_dir / "padtai_stdout.txt").write_text(result.stdout or "", encoding="utf-8")
    (cluster_dir / "padtai_stderr.txt").write_text(result.stderr or "", encoding="utf-8")
    (cluster_dir / "padtai_returncode.txt").write_text(f"{result.returncode}\n", encoding="utf-8")

    with open(cluster_dir / "run.log", "a", encoding="utf-8") as rl:
        rl.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - PADTAI returncode: {result.returncode}\n")
        rl.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - PADTAI elapsed: {elapsed_seconds:.2f}s\n")
        rl.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - PADTAI stdout (truncated 1024 chars):\n{(result.stdout or '')[:1024]}\n")
        rl.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - PADTAI stderr (truncated 4096 chars):\n{(result.stderr or '')[:4096]}\n")

    # Extract rules, but keep only positive (label_1) rules for the binary classifier
    all_rules = extract_rules_from_output(result.stdout or "")
    rules = [rule for rule in all_rules if "attr_label_1" in rule]
    rules_json = {
        "n_rules": len(rules),
        "rules": rules,
        "elapsed_seconds": elapsed_seconds
    }

    rules_file = cluster_dir / "padtai_rules.json"
    with open(rules_file, "w", encoding="utf-8") as f:
        json.dump(rules_json, f, indent=2)

    with open(cluster_dir / "run.log", "a", encoding="utf-8") as rl:
        rl.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - extracted {len(rules)} rules\n")
        rl.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - saved rules JSON to {rules_file.name}\n")

    status = "OK" if result.returncode == 0 else "FAIL"
    print(f"[cluster_{cluster_id}] {status} -> {cluster_dir}. See {cluster_dir / 'run.log'}")
    return result.returncode



def split_rule_terms(rule: str) -> list[str]:
    """Split rule body into terms, respecting nested parentheses."""
    term_split_re = re.compile(r",(?=(?:[^()]*\([^()]*\))*[^()]*$)")
    if ":-" not in rule:
        return []
    body = rule.split(":-", 1)[1].strip().rstrip(".")
    return [term.strip() for term in term_split_re.split(body) if term.strip()]


def extract_rule_label(rule: str) -> int | None:
    """Extract the label (0 or 1) from a rule, if present."""
    for term in split_rule_terms(rule):
        if term.startswith("attr_label_"):
            if "attr_label_1" in term:
                return 1
            if "attr_label_0" in term:
                return 0
    return None


def extract_attr_constraint(term: str) -> tuple[str, str] | None:
    """Extract the core attribute name and value from attr_* term."""
    if not term.startswith("attr_"):
        return None
    if "(" not in term:
        return None
    core = term[len("attr_") : term.index("(")]
    if "_" not in core:
        return None
    return core, term


def match_term_to_column(core: str, columns: list[str]) -> tuple[str, str] | None:
    """Match attr_* core to a column and extract expected value."""
    candidates: list[tuple[int, str, str]] = []
    for column in columns:
        prefix = f"{column.lower()}_"
        if core.startswith(prefix):
            expected = core[len(prefix) :]
            candidates.append((len(column), column, expected))

    if not candidates:
        return None

    _, column, expected = max(candidates, key=lambda item: item[0])
    return column, expected


def rule_matches_row(rule: str, row: pd.Series, columns: list[str]) -> bool:
    """Check if a rule fires on a given row."""
    terms = split_rule_terms(rule)
    for term in terms:
        if term == "attr_label" or term.startswith("attr_label_"):
            continue
        extracted = extract_attr_constraint(term)
        if extracted is None:
            continue
        core, _ = extracted
        matched = match_term_to_column(core, columns)
        if matched is None:
            continue
        column, expected = matched
        actual = str(row[column]).strip().lower()
        if actual != expected.strip().lower():
            return False
    return True


def predict_from_rules(df: pd.DataFrame, rules: list[str]) -> pd.Series:
    """Predict malware if any positive rule fires; otherwise goodware."""
    feature_columns = [column for column in df.columns if column != "label"]

    predictions = []
    for _, row in df.iterrows():
        fired = any(rule_matches_row(rule, row, feature_columns) for rule in rules)
        predictions.append(1 if fired else 0)
    return pd.Series(predictions, index=df.index, dtype=int)


def evaluate_clusters(cluster_ids: list[int]) -> None:
    """Evaluate all clusters and save metrics to CSV."""
    global OUTPUT_DIR

    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

    results = []
    for cluster_id in cluster_ids:
        cluster_dir = OUTPUT_DIR / f"cluster_{cluster_id}"
        rules_file = cluster_dir / "padtai_rules.json"
        data_file = cluster_dir / "padtai_input.prepared.csv"

        if not rules_file.exists() or not data_file.exists():
            print(f"[eval] cluster_{cluster_id}: missing files (rules or data)")
            continue

        try:
            # Load rules
            with open(rules_file, encoding="utf-8") as f:
                rules_data = json.load(f)
            rules = rules_data.get("rules", [])

            # Load data
            df = pd.read_csv(data_file)
            if "label" not in df.columns:
                print(f"[eval] cluster_{cluster_id}: missing label column")
                continue

            # Compute predictions
            y_true = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
            y_pred = predict_from_rules(df, rules)

            # Compute metrics
            accuracy = float(accuracy_score(y_true, y_pred))
            recall = float(recall_score(y_true, y_pred, zero_division=0))
            precision = float(precision_score(y_true, y_pred, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

            results.append({
                "cluster_id": cluster_id,
                "n_samples": len(df),
                "n_rules": len(rules),
                "n_label1_rules": sum(1 for rule in rules if "attr_label_1" in rule),
                "n_label0_rules": 0,
                "accuracy": accuracy,
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            })
            print(f"[eval] cluster_{cluster_id}: accuracy={accuracy:.4f}, recall={recall:.4f}, precision={precision:.4f}, f1={f1:.4f}")

        except Exception as e:
            print(f"[eval] cluster_{cluster_id}: error: {e}")
            traceback.print_exc()

    # Save results
    if results:
        df_results = pd.DataFrame(results).sort_values("cluster_id")
        output_file = OUTPUT_DIR / "evaluation_results.csv"
        df_results.to_csv(output_file, index=False)
        print(f"\n✓ Evaluation results saved to {output_file}")
        print(f"\nSummary:")
        print(f"  Mean accuracy:  {df_results['accuracy'].mean():.4f}")
        print(f"  Mean recall:    {df_results['recall'].mean():.4f}")
        print(f"  Mean precision: {df_results['precision'].mean():.4f}")
        print(f"  Mean F1:        {df_results['f1'].mean():.4f}")



def main() -> int:
    global OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Run PADTAI on sandbox cluster inputs.")
    parser.add_argument("--clusters", nargs="*", type=int, default=list(DEFAULT_CLUSTERS))
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--intcols", type=str, default="none", help="PADTAI --intcols argument (e.g., 'none', 'auto', '4,5,6')")
    parser.add_argument("--grounded", nargs="*", type=str, default=None, help="PADTAI --grounded argument(s) (e.g., 'sum:SumOperator' 'lt:LTOperator')")
    parser.add_argument("--no-eval", action="store_true", help="Skip automatic evaluation after PADTAI")
    args = parser.parse_args()

    # Create timestamped output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = SANDBOX_DIR / f"output_{timestamp}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Run PADTAI on clusters
    failures = 0
    for cluster_id in args.clusters:
        failures += 1 if run_cluster(cluster_id, args.timeout, args.intcols, args.grounded) != 0 else 0

    # Evaluate clusters if requested
    if not args.no_eval:
        print("\n" + "="*80)
        print("STARTING EVALUATION")
        print("="*80)
        evaluate_clusters(args.clusters)

    print(f"\n✓ All outputs saved to: {OUTPUT_DIR}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
