#!/bin/bash

# Script to run the balanced 1:1 Entropy KNN pipeline overnight
# Keeps the original pipeline intact and writes to separate balanced outputs

set -e

echo "========================================"
echo "Balanced Entropy KNN Pipeline - Overnight Run"
echo "========================================"
echo ""

echo "[1/6] Setting up directories..."
cd ..
REPO_DIR="Feature-Selection-with-ILP"
if [ ! -d "$REPO_DIR" ]; then
    echo "Error: Repository directory '$REPO_DIR' not found!"
    exit 1
fi

echo "[2/6] Initializing conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh

echo "[3/6] Activating 'malware-ilp' conda environment..."
conda activate malware-ilp

echo "[4/6] Entering repository: $REPO_DIR"
cd "$REPO_DIR"

echo "[5/6] Pulling latest changes from git..."
git pull

echo "[6/6] Loading Discord environment variables..."
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✓ Environment variables loaded"
    echo "  DISCORD_WEBHOOK_URL: ${DISCORD_WEBHOOK_URL:-(not set)}"
    echo "  DISCORD_USER_ID: ${DISCORD_USER_ID:-(not set)}"
else
    echo "Warning: .env file not found. Discord notifications may not work."
fi

echo ""
echo "========================================"
echo "Starting BALANCED 1:1 Entropy KNN execution"
echo "Timeout: 900 seconds per cluster"
echo "Output dir: reports/entropy_knn_balanced"
echo "========================================"
echo ""

python3 src/entropy_knn_balanced/runners/run_entropy_knn_balanced.py \
    --mode score-only \
    --cluster-sizes 500 \
    --seeds 42 \
    --top-features-global 1000 \
    --output-dir ./reports/entropy_knn_balanced \
    --discord-webhook-url "${DISCORD_WEBHOOK_URL}" \
    --discord-user-id "${DISCORD_USER_ID}"

echo ""
echo "========================================"
echo "Balanced pipeline execution completed!"
echo "========================================"
