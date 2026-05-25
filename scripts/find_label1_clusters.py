#!/usr/bin/env python3
"""Find clusters whose PADTAI rules mention attr_label_1.

Scans files like:
reports/entropy_knn/analysis/per_cluster_feature_vs_method/cluster_{i}/ilp_results/padtai_rules.json

For every cluster where at least one rule contains the substring 'attr_label_1',
write the cluster folder name (for example: cluster_0) to a txt file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

DEFAULT_BASE_DIR = Path("reports/entropy_knn/analysis/per_cluster_feature_vs_method")
DEFAULT_OUTPUT_FILE = Path("reports/entropy_knn/analysis/per_cluster_feature_vs_method/clusters_with_label1.txt")


def iter_cluster_dirs(base_dir: Path) -> Iterable[Path]:
    for cluster_dir in sorted(base_dir.glob("cluster_*")):
        if cluster_dir.is_dir():
            yield cluster_dir


def cluster_has_label1_rule(rules_file: Path) -> bool:
    if not rules_file.is_file():
        return False

    try:
        data = json.loads(rules_file.read_text(encoding="utf-8"))
    except Exception:
        return False

    rules = data.get("rules", [])
    if not isinstance(rules, list):
        return False

    return any("attr_label_1" in str(rule) for rule in rules)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find clusters with PADTAI rules that mention attr_label_1."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help=f"Base analysis directory (default: {DEFAULT_BASE_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"TXT file where matching clusters are written (default: {DEFAULT_OUTPUT_FILE})",
    )

    args = parser.parse_args()

    base_dir = args.base_dir
    output_file = args.output
    output_file.parent.mkdir(parents=True, exist_ok=True)

    matches: list[str] = []
    for cluster_dir in iter_cluster_dirs(base_dir):
        rules_file = cluster_dir / "ilp_results" / "padtai_rules.json"
        if cluster_has_label1_rule(rules_file):
            cluster_id = cluster_dir.name.split("_", 1)[1]
            matches.append(cluster_id)

    output_file.write_text("\n".join(matches) + ("\n" if matches else ""), encoding="utf-8")
    print(f"Found {len(matches)} clusters. Wrote list to: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
