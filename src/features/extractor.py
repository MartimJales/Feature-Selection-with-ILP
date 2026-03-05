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
        Extract binary features with vectorized processing (fast).
        """
        from scipy.sparse import lil_matrix

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

        # Create fast lookup dictionaries
        perm_to_idx = {p: i for i, p in enumerate(permissions_list)}
        susp_to_idx = {s: i + len(permissions_list) for i, s in enumerate(suspicious_list)}
        intent_to_idx = {it: i + len(permissions_list) + len(suspicious_list) for i, it in enumerate(intents_list)}

        n_binary_features = len(permissions_list) + len(suspicious_list) + len(intents_list)

        # Pass 2: Extract features with sparse matrix
        logger.info("Pass 2/2: Extracting features...")

        sparse_binary = lil_matrix((total_samples, n_binary_features), dtype=np.int8)
        count_features = np.zeros((total_samples, 8), dtype=np.int16)
        file_hashes = []

        for idx, sample in enumerate(samples):
            # Binary features: permissions (vectorized with lookup)
            perms = sample.get('req_permissions', [])
            for perm in perms:
                if perm in perm_to_idx:
                    sparse_binary[idx, perm_to_idx[perm]] = 1

            # Binary features: suspicious calls
            suspicious = sample.get('suspicious_calls', [])
            for call in suspicious:
                if call in susp_to_idx:
                    sparse_binary[idx, susp_to_idx[call]] = 1

            # Binary features: intents
            intents = sample.get('intent_filters', [])
            for intent in intents:
                if intent in intent_to_idx:
                    sparse_binary[idx, intent_to_idx[intent]] = 1

            # Count features (vectorized)
            count_features[idx, 0] = len(sample.get('activities', []))
            count_features[idx, 1] = len(sample.get('services', []))
            count_features[idx, 2] = len(sample.get('receivers', []))
            count_features[idx, 3] = len(sample.get('providers', []))
            count_features[idx, 4] = len(sample.get('api_calls', []))
            count_features[idx, 5] = len(sample.get('req_permissions', []))
            count_features[idx, 6] = len(sample.get('used_permissions', []))
            count_features[idx, 7] = len(sample.get('urls', []))

            file_hashes.append(sample['file_hash'])

            if (idx + 1) % 5000 == 0 or idx == total_samples - 1:
                progress_pct = ((idx + 1) / total_samples) * 100
                logger.info(f"  Extracted: {idx + 1}/{total_samples} ({progress_pct:.1f}%)")

        # Build DataFrame efficiently (avoid converting entire sparse matrix at once)
        logger.info("Building DataFrame from features...")

        # Convert sparse matrix to CSR for efficient column access
        sparse_binary = sparse_binary.tocsr()

        # Build dictionary with file_hash and count features first (memory-efficient)
        df_dict = {
            'file_hash': file_hashes,
            'n_activities': count_features[:, 0],
            'n_services': count_features[:, 1],
            'n_receivers': count_features[:, 2],
            'n_providers': count_features[:, 3],
            'n_api_calls': count_features[:, 4],
            'n_permissions': count_features[:, 5],
            'n_used_permissions': count_features[:, 6],
            'n_urls': count_features[:, 7]
        }

        # Add binary features efficiently (convert only needed columns)
        col_idx = 0
        for i, perm in enumerate(permissions_list):
            df_dict[f'perm_{perm}'] = sparse_binary.getcol(col_idx).toarray().ravel().astype(np.int8)
            col_idx += 1

        for i, call in enumerate(suspicious_list):
            df_dict[f'suspicious_{call}'] = sparse_binary.getcol(col_idx).toarray().ravel().astype(np.int8)
            col_idx += 1

        for i, intent in enumerate(intents_list):
            df_dict[f'intent_{intent}'] = sparse_binary.getcol(col_idx).toarray().ravel().astype(np.int8)
            col_idx += 1

        logger.info("Creating DataFrame...")
        df = pd.DataFrame(df_dict)
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

    logger.info(f"Saving DataFrame ({df.shape[0]} rows x {df.shape[1]} cols) to {output_file}...")
    logger.info("This may take several minutes for large datasets...")

    # Use Parquet format if available (much faster), otherwise CSV
    if output_file.endswith('.csv'):
        # Try to use Parquet instead
        parquet_file = output_file.replace('.csv', '.parquet')
        try:
            df.to_parquet(parquet_file, index=False, engine='pyarrow', compression='snappy')
            logger.info(f"✓ Features saved as Parquet: {parquet_file} (also saving CSV...)")
        except ImportError:
            logger.warning("pyarrow not available, skipping Parquet format")

        # Save CSV with progress indication
        df.to_csv(output_file, index=False)
        logger.info(f"✓ Features exported to {output_file}")
    else:
        df.to_parquet(output_file, index=False, engine='pyarrow', compression='snappy')
        logger.info(f"✓ Features exported to {output_file}")
