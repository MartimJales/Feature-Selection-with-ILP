#!/usr/bin/env python3
"""
Run PADTAI 3 times and consolidate stable rules.

Executes PADTAI with:
- --grounded none (no arithmetic)
- --intcols none (all binary)
- Different sample orders for each run (to reduce random bias)

Consolidates rules that appear in 2+ runs.
"""

import subprocess
import json
import logging
import re
import sys
import csv
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def create_run_temp_dataset(
    source_dataset_path: str,
    run_id: int,
    output_dir: str,
    sample_size: int
) -> str:
    """Create and save the exact per-run CSV used as PADTAI input."""

    source_path = Path(source_dataset_path)
    temp_dir = Path(output_dir) / "temp_datasets"
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_dataset_path = temp_dir / f"run_{run_id}_input.csv"

    with open(source_path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Dataset is empty: {source_dataset_path}")

    header = rows[0]
    data_rows = rows[1:]
    if not data_rows:
        raise ValueError(f"Dataset has no data rows: {source_dataset_path}")

    rng = random.Random(1000 + run_id)
    if sample_size > 0 and len(data_rows) > sample_size:
        sampled_rows = rng.sample(data_rows, sample_size)
    else:
        sampled_rows = data_rows.copy()
        rng.shuffle(sampled_rows)

    with open(temp_dataset_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(sampled_rows)

    logger.info(
        f"[Run {run_id}] Saved temp dataset: {temp_dataset_path} "
        f"({len(sampled_rows)} rows from {len(data_rows)} total)"
    )

    return str(temp_dataset_path)


def run_padtai_once(
    dataset_path: str,
    run_id: int,
    padtai_dir: str = "./PADTAI",
    solver: str = "nuwls",
    max_timeout: int = 1800,
    sample_size: int = 3000,
    debug_level: str = "padtai"
) -> Tuple[List[str], str]:
    """
    Run PADTAI once and extract rules from output.

    Args:
        dataset_path: Path to CSV file
        run_id: Run number (for logging)
        padtai_dir: Path to PADTAI directory
        solver: Solver to use (rc2 or nuwls)
        max_timeout: Timeout in seconds
        sample_size: Sample size for PADTAI
        debug_level: Debug level (padtai, popper, none, all)

    Returns:
        (list of rules, full stdout)
    """

    logger.info(f"\n[Run {run_id}] Starting PADTAI...")

    cmd = [
        "python3",
        f"{padtai_dir}/padtai.py",
        dataset_path,
        "--grounded", "none",
        "--intcols", "none",
        "--solver", solver,
        "--max-timeout", str(max_timeout),
        "--sample-size", str(sample_size),
        "--debug", debug_level,
        "--min-coverage", "5",
        "--min-recall", "10",
        "--min-precision", "75"
    ]

    logger.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max_timeout + 60
        )

        stdout = result.stdout
        stderr = result.stderr

        logger.info(f"[Run {run_id}] PADTAI completed (exit code: {result.returncode})")

        if result.returncode != 0:
            logger.warning(f"[Run {run_id}] Non-zero exit code. stderr:")
            print(stderr[:500])

        # Extract rules from output
        rules = extract_rules_from_output(stdout)

        logger.info(f"[Run {run_id}] Found {len(rules)} rules")
        for i, rule in enumerate(rules[:3], 1):
            print(f"  {i}. {rule[:100]}...")
        if len(rules) > 3:
            print(f"  ... and {len(rules) - 3} more")

        return rules, stdout

    except subprocess.TimeoutExpired:
        logger.error(f"[Run {run_id}] PADTAI timed out (>{max_timeout}s)")
        return [], ""
    except Exception as e:
        logger.error(f"[Run {run_id}] Error running PADTAI: {e}")
        return [], ""


def extract_rules_from_output(output: str) -> List[str]:
    """
    Extract Prolog rules from PADTAI output.
    Looks for lines with format: "Rule: head :- body"
    """

    rules = []

    # Pattern for debug output: "Rule: ... :- ..."
    pattern = r"Rule:\s*(.+?)\s*:-\s*(.+?)(?:\n|$)"

    for match in re.finditer(pattern, output, re.MULTILINE | re.IGNORECASE):
        head = match.group(1).strip()
        body = match.group(2).strip()
        rule = f"{head} :- {body}"
        rules.append(rule)

    # Also check for solution lines
    if "Solution" in output:
        for line in output.split('\n'):
            if ":-" in line and ("Rule:" in line or line.strip().startswith("malware") or line.strip().startswith("positive")):
                rule = line.split("Rule:")[-1].strip() if "Rule:" in line else line.strip()
                if rule and rule not in rules:
                    rules.append(rule)

    return rules


def consolidate_rules(
    all_rules: Dict[int, List[str]],
    min_appearances: int = 2
) -> Dict[str, Dict]:
    """
    Consolidate rules from multiple runs.

    Args:
        all_rules: Dict mapping run_id -> list of rules
        min_appearances: Minimum number of runs in which rule must appear

    Returns:
        Dict with rule -> {count, run_ids, stability}
    """

    rule_counts = Counter()
    rule_runs = {}

    for run_id, rules in all_rules.items():
        for rule in rules:
            normalized = rule.strip()
            rule_counts[normalized] += 1
            if normalized not in rule_runs:
                rule_runs[normalized] = []
            rule_runs[normalized].append(run_id)

    # Filter by minimum appearances
    stable_rules = {
        rule: {
            'count': count,
            'stability': f"{count}/{len(all_rules)}",
            'run_ids': rule_runs[rule]
        }
        for rule, count in rule_counts.items()
        if count >= min_appearances
    }

    return stable_rules


def run_padtai_pipeline(
    dataset_path: str = "./data/ilp/top200_ig_malware.csv",
    n_runs: int = 3,
    padtai_dir: str = "./PADTAI",
    output_dir: str = "./reports/padtai",
    solver: str = "nuwls",
    max_timeout: int = 1800,
    sample_size: int = 3000
) -> None:
    """
    Run full PADTAI pipeline: 3 runs + consolidation.

    Args:
        dataset_path: Path to top-200 CSV
        n_runs: Number of independent runs
        padtai_dir: Path to PADTAI directory
        output_dir: Output directory for results
        solver: Solver choice
        max_timeout: Timeout per run
        sample_size: Sample size per run
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check dataset exists
    if not Path(dataset_path).exists():
        logger.error(f"✗ Dataset not found: {dataset_path}")
        logger.info("Run prepare_dataset.py first:")
        logger.info("  python3 src/padtai/prepare_dataset.py")
        sys.exit(1)

    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Number of runs: {n_runs}")
    logger.info(f"Solver: {solver}")
    logger.info(f"Max timeout: {max_timeout}s")
    logger.info(f"Sample size: {sample_size}")
    logger.info(f"Temp datasets: {output_path / 'temp_datasets'}")

    # Run PADTAI multiple times
    all_rules = {}
    all_outputs = {}

    start_time = time.time()

    for run_id in range(1, n_runs + 1):
        run_dataset_path = create_run_temp_dataset(
            source_dataset_path=dataset_path,
            run_id=run_id,
            output_dir=output_dir,
            sample_size=sample_size
        )

        rules, output = run_padtai_once(
            dataset_path=run_dataset_path,
            run_id=run_id,
            padtai_dir=padtai_dir,
            solver=solver,
            max_timeout=max_timeout,
            sample_size=sample_size
        )

        all_rules[run_id] = rules
        all_outputs[run_id] = output

        # Save individual run output
        run_output_file = output_path / f"run_{run_id}_output.txt"
        with open(run_output_file, 'w') as f:
            f.write(output)
        logger.info(f"✓ Saved output to {run_output_file}")

        if run_id < n_runs:
            logger.info(f"Waiting 10s before next run...")
            time.sleep(10)

    elapsed_time = time.time() - start_time
    logger.info(f"\n✓ All {n_runs} runs completed in {elapsed_time:.1f}s")

    # Consolidate rules
    logger.info("\n=== Consolidating rules ===")
    stable_rules = consolidate_rules(all_rules, min_appearances=2)

    logger.info(f"Stable rules (appearing in 2+ runs): {len(stable_rules)}")
    for i, (rule, info) in enumerate(list(stable_rules.items())[:5], 1):
        print(f"\n{i}. [{info['stability']}] {rule}")
    if len(stable_rules) > 5:
        print(f"\n... and {len(stable_rules) - 5} more")

    # Save consolidated results
    results = {
        'metadata': {
            'n_runs': n_runs,
            'dataset': dataset_path,
            'elapsed_time_sec': elapsed_time,
            'solver': solver,
            'max_timeout': max_timeout,
            'sample_size': sample_size
        },
        'rules_per_run': {str(k): v for k, v in all_rules.items()},
        'stable_rules': {
            rule: {k: v for k, v in info.items() if k != 'run_ids'}
            for rule, info in stable_rules.items()
        },
        'stable_rules_with_runs': {
            rule: info for rule, info in stable_rules.items()
        }
    }

    results_file = output_path / "consolidated_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✓ Saved consolidated results to {results_file}")

    # Save summary report
    summary_file = output_path / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=== PADTAI ILP Pipeline Summary ===\n\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Number of runs: {n_runs}\n")
        f.write(f"Total elapsed time: {elapsed_time:.1f}s\n")
        f.write(f"Solver: {solver}\n\n")

        f.write("=== Rules per run ===\n")
        for run_id, rules in all_rules.items():
            f.write(f"Run {run_id}: {len(rules)} rules\n")

        f.write(f"\n=== Stable rules (2+ runs) ===\n")
        f.write(f"Count: {len(stable_rules)}\n\n")
        for i, (rule, info) in enumerate(sorted(stable_rules.items(), key=lambda x: -x[1]['count']), 1):
            f.write(f"{i}. [{info['stability']}] {rule}\n")

    logger.info(f"✓ Saved summary to {summary_file}")

    logger.info(f"\n✓ Pipeline completed!")
    logger.info(f"Results: {output_path}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run PADTAI pipeline (3 runs + consolidation)")
    parser.add_argument('--dataset', default='./data/ilp/top200_ig_malware.csv',
                        help='Path to top-200 CSV dataset')
    parser.add_argument('--runs', type=int, default=3,
                        help='Number of PADTAI runs (default: 3)')
    parser.add_argument('--padtai-dir', default='./PADTAI',
                        help='Path to PADTAI directory')
    parser.add_argument('--output', default='./reports/padtai',
                        help='Output directory for results')
    parser.add_argument('--solver', choices=['rc2', 'nuwls'], default='nuwls',
                        help='Solver choice (default: nuwls)')
    parser.add_argument('--timeout', type=int, default=1800,
                        help='Max timeout per run in seconds (default: 1800)')
    parser.add_argument('--sample-size', type=int, default=3000,
                        help='Sample size for PADTAI (default: 3000)')

    args = parser.parse_args()

    run_padtai_pipeline(
        dataset_path=args.dataset,
        n_runs=args.runs,
        padtai_dir=args.padtai_dir,
        output_dir=args.output,
        solver=args.solver,
        max_timeout=args.timeout,
        sample_size=args.sample_size
    )
