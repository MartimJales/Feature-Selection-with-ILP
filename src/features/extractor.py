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
        total_files = len(json_files)
        progress_step = max(1, total_files // 20)  # ~5% updates

        for idx, file_path in enumerate(json_files, start=1):
            with open(file_path, 'r') as f:
                sample = json.load(f)
                sample['file_hash'] = file_path.stem
                data.append(sample)

            if idx == 1 or idx % progress_step == 0 or idx == total_files:
                logger.info(f"Loading JSON files: {idx}/{total_files} ({(idx/total_files)*100:.1f}%)")

        logger.info(f"Loaded {len(data)} JSON files")
        return data

    def extract_binary_features(self, samples: List[Dict]) -> pd.DataFrame:
        """
        Extract binary features with memory optimization (chunked processing).
        """
        # Pass 1: Build vocabulary
        logger.info("Pass 1/2: Building feature vocabulary...")
        all_permissions = set()
        all_suspicious = set()
        all_intents = set()

        total_samples = len(samples)

        for idx, sample in enumerate(samples, start=1):
            all_permissions.update(sample.get('req_permissions', []))
            all_suspicious.update(sample.get('suspicious_calls', []))
            all_intents.update(sample.get('intent_filters', []))

            if idx % 5000 == 0 or idx == total_samples:
                logger.info(f"  Vocabulary: {idx}/{total_samples} ({(idx/total_samples)*100:.1f}%)")

        # Convert to sorted lists for consistent ordering
        permissions_list = sorted(list(all_permissions))
        suspicious_list = sorted(list(all_suspicious))
        intents_list = sorted(list(all_intents))

        logger.info(f"Vocabulary: {len(permissions_list)} permissions, {len(suspicious_list)} suspicious calls, {len(intents_list)} intents")

        # Pass 2: Extract features in chunks (memory-efficient)
        logger.info("Pass 2/2: Extracting features...")
        chunk_size = 5000  # Process in chunks to avoid OOM
        all_chunks = []
        processed = 0

        for chunk_start in range(0, total_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_samples)
            chunk_samples = samples[chunk_start:chunk_end]

            rows = []
            for sample in chunk_samples:
                row = {'file_hash': sample['file_hash']}

                # Binary features: permissions
                perms = sample.get('req_permissions', [])
                for perm in permissions_list:
                    row[f'perm_{perm}'] = int(perm in perms)

                # Binary features: suspicious calls
                suspicious = sample.get('suspicious_calls', [])
                for call in suspicious_list:
                    row[f'suspicious_{call}'] = int(call in suspicious)

                # Binary features: intents
                intents = sample.get('intent_filters', [])
                for intent in intents_list:
                    row[f'intent_{intent}'] = int(intent in intents)

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

            chunk_df = pd.DataFrame(rows)
            all_chunks.append(chunk_df)
            processed = chunk_end

            progress_pct = (processed / total_samples) * 100
            logger.info(f"  Extracted: {processed}/{total_samples} ({progress_pct:.1f}%)")

        # Concatenate all chunks
        df = pd.concat(all_chunks, ignore_index=True)
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
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Extract features from Android malware JSON files")
    parser.add_argument("json_dir", nargs="?", default="./data/destino", help="Directory containing JSON files")
    parser.add_argument("--output", "-o", default="./reports/extracted_features.csv", help="Output CSV file path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to load")

    args = parser.parse_args()

    json_dir = args.json_dir
    output_file = args.output

    extractor = JSONFeatureExtractor(json_dir)
    samples = extractor.load_json_files(limit=args.limit)
    df = extractor.extract_binary_features(samples)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    logger.info(f"Features exported to {output_file}")
