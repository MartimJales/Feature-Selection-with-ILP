import os
import json
import csv
import numpy as np
import sys

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

if __name__ == "__main__":
    json_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/destino"
    csv_path = sys.argv[2] if len(sys.argv) > 2 else "./data/training_set.csv"

    print(f"Looking for JSON files in: {json_dir}")
    print(f"Looking for labels in: {csv_path}")

    data, labels = load_data(json_dir, csv_path)
    print(f"Final: Loaded {len(data)} samples with labels")
    if len(labels) > 0:
        print(f"Label distribution: {np.sum(labels)} malicious, {len(labels) - np.sum(labels)} benign")
