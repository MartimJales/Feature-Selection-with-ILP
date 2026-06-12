#!/bin/bash

# Complete balanced 1:1 pipeline: clustering + PADTAI
# Runs the full pipeline from feature selection to rule discovery

set -e

echo "========================================"
echo "Complete Balanced 1:1 Pipeline"
echo "Clustering + PADTAI Rule Discovery"
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
    set -a
    source .env
    set +a
    echo "✓ Environment variables loaded"
    echo "  DISCORD_WEBHOOK_URL: ${DISCORD_WEBHOOK_URL:+set}"
    echo "  DISCORD_USER_ID: ${DISCORD_USER_ID:+set}"
else
    echo "Warning: .env file not found. Discord notifications may not work."
fi

RUN_TIMESTAMP="$(date '+%Y-%m-%d_%H')"
OUTPUT_DIR="./reports_parallel/entropy_knn_balanced/run_${RUN_TIMESTAMP}"

echo ""
echo "========================================"
echo "PHASE 1: Balanced 1:1 Clustering"
echo "PHASE 2: PADTAI Rule Discovery"
echo "Output: $OUTPUT_DIR"
echo "========================================"
echo ""

# Send Discord notification - pipeline started
if [ -n "$DISCORD_WEBHOOK_URL" ] && [ -n "$DISCORD_USER_ID" ]; then
    python3 scripts/test_discord_notifications.py \
        --webhook-url "$DISCORD_WEBHOOK_URL" \
        --user-id "$DISCORD_USER_ID" \
        --message "🚀 **Complete Balanced 1:1 Pipeline** - Starting clustering + PADTAI rule discovery" \
        2>/dev/null || echo "Warning: Failed to send Discord notification"
fi

echo ""

python3 src/entropy_knn_balanced/runners/run_complete_pipeline_balanced.py \
    --output-dir "$OUTPUT_DIR" \
    --cluster-sizes 500 \
    --seeds 42 \
    --top-features-global 1000 \
    --balance-seed 42 \
    --max-clusters 10 \
    --ilp-top-n 30 \
    --ilp-timeout 0 \
    --ilp-workers 10 \
    --discord-webhook-url "$DISCORD_WEBHOOK_URL" \
    --discord-user-id "$DISCORD_USER_ID"

echo ""
echo "========================================"
echo "Complete pipeline execution finished!"
echo "========================================"
