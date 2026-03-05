import os
import json
import csv
import numpy as np
import pandas as pd
import sys
import argparse
from pathlib import Path

def load_labels(csv_path):
    """Load labels from CSV file."""
    if not os.path.exists(csv_path):
        print(f"ERROR: Labels file not found at {csv_path}")
        return {}

    labels = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        print(f"CSV headers: {headers}")

        for row in reader:
            # Tentar diferentes formatos de chave
            key_candidates = [
                row.get('sha256', ''),
                row.get('SHA256', ''),
                row.get('hash', ''),
                row.get('file', '').replace('.json', '')
            ]

            key = next((k for k in key_candidates if k), None)

            if key:
                label = row.get('label', row.get('Label', ''))
                # Normalize to uppercase for consistency
                labels[key.upper()] = 1 if label.lower() in ['malicious', '1', 'malware'] else 0

    print(f"Loaded {len(labels)} labels from CSV")
    if labels:
        print(f"Sample keys: {list(labels.keys())[:3]}")
    return labels

def load_data(json_dir, csv_path):
    """Load and process JSON files with their labels."""
    labels_dict = load_labels(csv_path)

    data = []
    labels = []
    matched = 0
    unmatched_samples = []

    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json'):
            # Normalize to uppercase for comparison
            key = json_file.replace('.json', '').upper()

            if key in labels_dict:
                matched += 1
                with open(os.path.join(json_dir, json_file)) as f:
                    features = json.load(f)
                    data.append(features)
                    labels.append(labels_dict[key])
            else:
                if len(unmatched_samples) < 3:
                    unmatched_samples.append(key)

    print(f"Matched {matched} samples with labels")
    if unmatched_samples:
        print(f"Sample unmatched keys: {unmatched_samples}")

    return np.array(data), np.array(labels)

def load_features_file(features_path, labels_path):
    """Load features from Parquet or CSV file with labels."""
    features_path = Path(features_path)

    print(f"Loading features from: {features_path}")

    # Load features (Parquet or CSV)
    if features_path.suffix == '.parquet':
        df = pd.read_parquet(features_path)
        print(f"✓ Loaded Parquet file: {df.shape[0]} rows x {df.shape[1]} cols")
    elif features_path.suffix == '.csv':
        df = pd.read_csv(features_path)
        print(f"✓ Loaded CSV file: {df.shape[0]} rows x {df.shape[1]} cols")
    else:
        raise ValueError(f"Unsupported file format: {features_path.suffix}")

    # Load labels
    labels_df = pd.read_csv(labels_path)
    print(f"✓ Loaded {len(labels_df)} labels")

    # Merge with labels
    if 'file_hash' in df.columns:
        df['file_hash'] = df['file_hash'].str.lower().str.strip()
        labels_df['sha256'] = labels_df['sha256'].str.lower().str.strip()

        df = df.merge(labels_df[['sha256', 'label']],
                     left_on='file_hash',
                     right_on='sha256',
                     how='left')

        df = df.dropna(subset=['label'])
        df = df.drop(columns=['file_hash', 'sha256'], errors='ignore')

        matched = len(df)
        print(f"✓ Matched {matched} samples with labels")

        X = df.drop(columns=['label'])
        y = df['label'].astype(int).values

        print(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Label distribution: {np.sum(y)} malicious, {len(y) - np.sum(y)} benign")

        return X, y
    else:
        raise ValueError("Features file must contain 'file_hash' column")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load features and labels")
    parser.add_argument("json_dir", nargs="?", default="./data/destino", help="Directory with JSON files (legacy)")
    parser.add_argument("labels_path", nargs="?", default="./data/training_set.csv", help="Path to labels CSV")
    parser.add_argument("--features", type=str, default=None, help="Path to extracted features file (.parquet or .csv)")

    args = parser.parse_args()

    # If features file is provided, use it (faster)
    if args.features and Path(args.features).exists():
        X, y = load_features_file(args.features, args.labels_path)
    else:
        # Legacy mode: load from JSON files
        print(f"Looking for JSON files in: {args.json_dir}")
        print(f"Looking for labels in: {args.labels_path}")

        data, labels = load_data(args.json_dir, args.labels_path)
        print(f"Final: Loaded {len(data)} samples with labels")
        if len(labels) > 0:
            print(f"Label distribution: {np.sum(labels)} malicious, {len(labels) - np.sum(labels)} benign")
