#!/bin/bash

# Script to run ILP for all clusters overnight
# This script ensures proper environment setup and executes the full pipeline

set -e  # Exit on error

echo "========================================"
echo "ILP All Clusters Pipeline - Overnight Run"
echo "========================================"
echo ""

# Step 1: Navigate to parent directory and back into repo
echo "[1/6] Setting up directories..."
cd ..
REPO_DIR="Feature-Selection-with-ILP"
if [ ! -d "$REPO_DIR" ]; then
    echo "Error: Repository directory '$REPO_DIR' not found!"
    exit 1
fi

# Step 2: Initialize conda
echo "[2/6] Initializing conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh

# Step 3: Activate conda environment
echo "[3/6] Activating 'malware-ilp' conda environment..."
conda activate malware-ilp

# Step 4: Enter repository
echo "[4/6] Entering repository: $REPO_DIR"
cd "$REPO_DIR"

# Step 5: Update repository
echo "[5/6] Pulling latest changes from git..."
git pull

# Step 6: Load Discord environment variables from .env file
echo "[6/6] Loading Discord environment variables..."
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
    echo "✓ Environment variables loaded"
    echo "  DISCORD_WEBHOOK_URL: ${DISCORD_WEBHOOK_URL:+set}"
    echo "  DISCORD_USER_ID: ${DISCORD_USER_ID:+set}"
else
    echo "Warning: .env file not found. Discord notifications may not work."
fi

echo ""
echo "========================================"
echo "Starting ILP execution for all clusters"
echo "Timeout: 900 seconds per cluster"
echo "========================================"
echo ""

# Run the ILP pipeline for all clusters (0-99)
python3 src/ilp_pipeline/runners/run_ilp_per_cluster_test.py \
    --cluster-ids $(seq 0 99) \
    --timeout 900

echo ""
echo "========================================"
echo "Pipeline execution completed!"
echo "========================================"
