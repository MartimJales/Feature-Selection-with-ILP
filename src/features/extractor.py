import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class JSONFeatureExtractor:
    """Extract and process features from Android malware JSON files."""

    def __init__(self, json_dir: str = "./data/destino"):
        self.json_dir = Path(json_dir)

    def load_json_files(self, limit: int = None) -> List[Dict[str, Any]]:
        """Load JSON files from directory. If limit is None, load all files."""
        json_files = sorted(list(self.json_dir.glob("*.json")))

        if limit is not None and limit > 0 and len(json_files) > limit:
            json_files = json_files[:limit]
            logger.info(f"Limiting to {limit} samples (total available: {len(list(self.json_dir.glob('*.json')))})")

        data = []
        for file_path in json_files:
            with open(file_path, 'r') as f:
                sample = json.load(f)
                sample['file_hash'] = file_path.stem
                data.append(sample)

        logger.info(f"Loaded {len(data)} JSON files")
        return data

    def extract_binary_features(self, samples: List[Dict]) -> pd.DataFrame:
        """
        Extract binary features: permission presence, suspicious API calls, etc.
        """
        # Coletar vocabulário único
        all_permissions = set()
        all_activities = set()
        all_services = set()
        all_receivers = set()
        all_api_calls = set()
        all_suspicious = set()
        all_intents = set()

        for sample in samples:
            all_permissions.update(sample.get('req_permissions', []))
            all_activities.update(sample.get('activities', []))
            all_services.update(sample.get('services', []))
            all_receivers.update(sample.get('receivers', []))
            all_api_calls.update(sample.get('api_calls', []))
            all_suspicious.update(sample.get('suspicious_calls', []))
            all_intents.update(sample.get('intent_filters', []))

        # Criar DataFrame binário
        rows = []
        for sample in samples:
            row = {'file_hash': sample['file_hash']}

            # Binary features: permissions
            for perm in all_permissions:
                row[f'perm_{perm}'] = int(perm in sample.get('req_permissions', []))

            # Binary features: suspicious calls
            for call in all_suspicious:
                row[f'suspicious_{call}'] = int(call in sample.get('suspicious_calls', []))

            # Binary features: intents
            for intent in all_intents:
                row[f'intent_{intent}'] = int(intent in sample.get('intent_filters', []))

            # Count features
            row['n_activities'] = len(sample.get('activities', []))
            row['n_services'] = len(sample.get('services', []))
            row['n_receivers'] = len(sample.get('receivers', []))
            row['n_providers'] = len(sample.get('providers', []))
            row['n_api_calls'] = len(sample.get('api_calls', []))
            row['n_permissions'] = len(sample.get('req_permissions', []))
            row['n_used_permissions'] = len(sample.get('used_permissions', []))
            row['n_urls'] = len(sample.get('urls', []))

            rows.append(row)

        df = pd.DataFrame(rows)
        logger.info(f"Extracted {df.shape[1]-1} features from {df.shape[0]} samples")
        return df

    def load_with_labels(self, labels_path: str) -> tuple:
        samples = self.load_json_files()
        df = self.extract_binary_features(samples)

        if Path(labels_path).exists():
            labels_df = pd.read_csv(labels_path)

            # Normalizar hashes
            df['file_hash'] = df['file_hash'].str.lower().str.strip()
            labels_df['sha256'] = labels_df['sha256'].str.lower().str.strip()

            # Merge
            df = df.merge(labels_df[['sha256', 'label']],
                        left_on='file_hash',
                        right_on='sha256',
                        how='left')

            df = df.dropna(subset=['label'])

        else:
            df['label'] = 1

        # Remover colunas auxiliares
        df = df.drop(columns=['file_hash', 'sha256', 'timestamp'], errors='ignore')

        X = df.drop(columns=['label'])
        y = df['label'].astype(int).values

        return X, y

if __name__ == "__main__":
    import sys
    json_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/destino"

    extractor = JSONFeatureExtractor(json_dir)
    samples = extractor.load_json_files()
    df = extractor.extract_binary_features(samples)

    Path("./reports").mkdir(parents=True, exist_ok=True)
    df.to_csv('./reports/extracted_features.csv', index=False)
    logger.info("Features exported to ./reports/extracted_features.csv")
