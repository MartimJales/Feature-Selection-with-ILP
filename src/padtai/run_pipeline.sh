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
SAMPLE_SIZE="${3:--1}"

# Step 0: Preflight checks
echo "[0/3] Checking PADTAI runtime dependencies..."
python3 - <<'PY'
import sys

missing = []
for mod in ("pkg_resources", "janus_swi", "pyarrow"):
    try:
        __import__(mod)
    except Exception:
        missing.append(mod)

try:
    from popper.loop import learn_solution  # noqa: F401
except Exception:
    missing.append("popper")

if missing:
    print("Missing dependencies for PADTAI runtime:", ", ".join(sorted(set(missing))))
    print("Hint: activate the correct conda env and reinstall PADTAI requirements.")
    sys.exit(1)

print("✓ Runtime dependencies OK")
PY

# Step 1: Prepare dataset
echo "[1/3] Preparing top-${TOP_K} features..."
python3 -m src.padtai.prepare_dataset \
    --top-k "$TOP_K" \
    --output "$DATASET_PATH"

# Step 2: Validate generated dataset header for PADTAI syntax
echo ""
echo "[2/3] Validating generated dataset format..."
python3 - <<PY
import csv
import re
import sys

dataset_path = "$DATASET_PATH"
with open(dataset_path, "r", newline="") as f:
    reader = csv.reader(f)
    rows = list(reader)

if len(rows) < 2:
    print(f"Dataset has no data rows: {dataset_path}")
    sys.exit(1)

header = rows[0]
if header[-1] != "label":
    print("Last column must be 'label'.")
    print(f"Found: {header[-1]}")
    sys.exit(1)

bad = [c for c in header[:-1] if not re.match(r"^[a-z_][a-z0-9_]*$", c)]
if bad:
    print("Found invalid PADTAI column names (showing first 5):")
    for col in bad[:5]:
        print(" -", col)
    sys.exit(1)

print(f"✓ Dataset OK: {len(rows)-1} rows, {len(header)-1} features + label")
PY

echo ""
echo "[3/3] Running PADTAI pipeline (3 runs)..."
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
