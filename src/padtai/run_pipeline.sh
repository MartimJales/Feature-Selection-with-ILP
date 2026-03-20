#!/bin/bash
# Complete pipeline: prepare dataset + run PADTAI 3 times

set -e

echo "=== PADTAI ILP Pipeline ==="
echo ""

DATASET_PATH="./data/ilp/top200_ig_malware.csv"
OUTPUT_DIR="./reports/padtai"
PADTAI_DIR="./PADTAI"
TOP_K="${1:-200}"
TIMEOUT="${2:-1800}"
SAMPLE_SIZE="${3:-3000}"

# Step 1: Prepare dataset
echo "[1/2] Preparing top-${TOP_K} features..."
python3 -m src.padtai.prepare_dataset \
    --top-k "$TOP_K" \
    --output "$DATASET_PATH"

echo ""
echo "[2/2] Running PADTAI pipeline (3 runs)..."
python3 -m src.padtai.run_padtai \
    --dataset "$DATASET_PATH" \
    --runs 3 \
    --padtai-dir "$PADTAI_DIR" \
    --output "$OUTPUT_DIR" \
    --timeout "$TIMEOUT" \
    --sample-size "$SAMPLE_SIZE"

echo ""
echo "✓ Pipeline completed!"
echo "Results: $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR"
