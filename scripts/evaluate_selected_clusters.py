#!/usr/bin/env python3
"""Evaluate selected PADTAI clusters using the discovered positive rules.

This script reads a txt file with one cluster id per line, loads each cluster's
`padtai_rules.json` and `padtai_input.csv`, and computes binary classification
metrics by predicting label 1 when at least one positive rule fires.

Positive rules are the ones that contain `attr_label_1`.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

DEFAULT_BASE_DIR = Path("reports/entropy_knn/analysis/per_cluster_feature_vs_method")
DEFAULT_CLUSTERS_FILE = DEFAULT_BASE_DIR / "clusters_with_label1.txt"
DEFAULT_OUTPUT_CSV = DEFAULT_BASE_DIR / "label1_cluster_metrics.csv"

TERM_SPLIT_RE = re.compile(r",(?=(?:[^()]*\([^()]*\))*[^()]*$)")
LABEL_RULE_MARKER = "attr_label_1"


def read_cluster_ids(path: Path) -> list[int]:
    ids: list[int] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        text = raw_line.strip()
        if not text:
            continue
        if text.isdigit():
            ids.append(int(text))
    return ids


def load_rules(rules_path: Path) -> list[str]:
    data = json.loads(rules_path.read_text(encoding="utf-8"))
    rules = data.get("rules", [])
    return [str(rule) for rule in rules if isinstance(rule, (str, int, float))]


def split_rule_terms(rule: str) -> list[str]:
    if ":-" not in rule:
        return []
    body = rule.split(":-", 1)[1].strip().rstrip(".")
    return [term.strip() for term in TERM_SPLIT_RE.split(body) if term.strip()]


def extract_attr_constraint(term: str) -> tuple[str, str] | None:
    if not term.startswith("attr_"):
        return None
    if "(" not in term:
        return None

    core = term[len("attr_"): term.index("(")]
    if "_" not in core:
        return None

    return core, term


def extract_rule_label(rule: str) -> int | None:
    for term in split_rule_terms(rule):
        if term.startswith("attr_label_"):
            if "attr_label_1" in term:
                return 1
            if "attr_label_0" in term:
                return 0
    return None


def match_term_to_column(core: str, columns: list[str]) -> tuple[str, str] | None:
    """Return (column_name, expected_value) for an attr_* term core."""
    candidates: list[tuple[int, str, str]] = []
    for column in columns:
        prefix = f"{column.lower()}_"
        if core.startswith(prefix):
            expected = core[len(prefix):]
            candidates.append((len(column), column, expected))

    if not candidates:
        return None

    _, column, expected = max(candidates, key=lambda item: item[0])
    return column, expected


def rule_matches_row(rule: str, row: pd.Series, columns: list[str]) -> bool:
    terms = split_rule_terms(rule)
    for term in terms:
        if term == LABEL_RULE_MARKER or term.startswith("attr_label_"):
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
    feature_columns = [column for column in df.columns if column != "label"]

    predictions = []
    for _, row in df.iterrows():
        fired_labels = [
            extract_rule_label(rule)
            for rule in rules
            if rule_matches_row(rule, row, feature_columns)
        ]
        label_votes = Counter(label for label in fired_labels if label is not None)
        if not label_votes:
            predictions.append(0)
        elif label_votes[1] > label_votes[0]:
            predictions.append(1)
        elif label_votes[0] > label_votes[1]:
            predictions.append(0)
        else:
            predictions.append(1 if label_votes[1] else 0)
    return pd.Series(predictions, index=df.index, dtype=int)


def evaluate_cluster(cluster_dir: Path) -> dict:
    rules_path = cluster_dir / "ilp_results" / "padtai_rules.json"
    data_path = cluster_dir / "ilp_results" / "padtai_input.csv"

    df = pd.read_csv(data_path)
    if "label" not in df.columns:
        raise ValueError(f"Missing label column in {data_path}")

    rules = load_rules(rules_path)
    y_true = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    y_pred = predict_from_rules(df, rules)

    accuracy = float(accuracy_score(y_true, y_pred))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "cluster_id": int(cluster_dir.name.split("_", 1)[1]),
        "n_samples": int(len(df)),
        "n_rules": int(len(rules)),
        "n_label1_rules": int(sum(1 for rule in rules if "attr_label_1" in rule)),
        "n_label0_rules": int(sum(1 for rule in rules if "attr_label_0" in rule)),
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def iter_selected_clusters(base_dir: Path, cluster_ids: Iterable[int]) -> Iterable[Path]:
    wanted = {int(cluster_id) for cluster_id in cluster_ids}
    # Handle nested structure: base_dir/cluster_*/seed_*/cluster_*
    for cluster_outer_dir in sorted(base_dir.glob("cluster_*")):
        if not cluster_outer_dir.is_dir():
            continue
        for seed_dir in sorted(cluster_outer_dir.glob("seed_*")):
            if not seed_dir.is_dir():
                continue
            for cluster_dir in sorted(seed_dir.glob("cluster_*")):
                if not cluster_dir.is_dir():
                    continue
                try:
                    cluster_id = int(cluster_dir.name.split("_", 1)[1])
                except Exception:
                    continue
                if cluster_id in wanted:
                    yield cluster_dir
def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate selected clusters with label_1 rules.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--clusters-file", type=Path, default=DEFAULT_CLUSTERS_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()

    cluster_ids = read_cluster_ids(args.clusters_file)
    if not cluster_ids:
        raise SystemExit(f"No cluster ids found in {args.clusters_file}")

    rows = []
    for cluster_dir in iter_selected_clusters(args.base_dir, cluster_ids):
        try:
            rows.append(evaluate_cluster(cluster_dir))
        except Exception as exc:
            rows.append(
                {
                    "cluster_id": int(cluster_dir.name.split("_", 1)[1]),
                    "error": str(exc),
                }
            )

    df = pd.DataFrame(rows).sort_values("cluster_id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Wrote metrics to {args.output}")
    if not df.empty and "accuracy" in df.columns:
        ok = df.dropna(subset=["accuracy", "recall"])
        if not ok.empty:
            print(f"Mean accuracy: {ok['accuracy'].mean():.4f}")
            print(f"Mean recall:   {ok['recall'].mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
