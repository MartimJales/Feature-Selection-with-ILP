#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/decision_tree}"
REPORTS_DIR="${REPORTS_DIR:-$ROOT_DIR/reports/decision_tree}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="$LOG_DIR/decision_tree_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/decision_tree_${TIMESTAMP}.pid"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
FEATURES_PATH="${FEATURES_PATH:-$ROOT_DIR/reports/extracted_features.parquet}"
LABELS_PATH="${LABELS_PATH:-$ROOT_DIR/data/training_set.csv}"
RANKINGS_PATH="${RANKINGS_PATH:-$ROOT_DIR/reports/feature_analysis/feature_rankings_all.parquet}"
TOP_K_VALUES="${TOP_K_VALUES:-200 500 1000}"
DEPTH_VALUES="${DEPTH_VALUES:-2 3 4 5 6 8 10 12 15 20 25 30 None}"
PLOT_MAX_DEPTH="${PLOT_MAX_DEPTH:-6}"
TOP_N_IMPORTANCES="${TOP_N_IMPORTANCES:-30}"
SEED="${SEED:-42}"

RUN_IN_BACKGROUND=1
if [[ "${1:-}" == "--foreground" ]]; then
	RUN_IN_BACKGROUND=0
	shift
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
	PYTHON_BIN="$(command -v python3 || true)"
fi

if [[ -z "$PYTHON_BIN" ]]; then
	echo "[ERRO] Python não encontrado. Defina PYTHON_BIN ou instale python3."
	exit 1
fi

mkdir -p "$LOG_DIR" "$REPORTS_DIR"

read -r -a TOP_K_ARR <<< "$TOP_K_VALUES"
read -r -a DEPTH_ARR <<< "$DEPTH_VALUES"

CMD=(
	"$PYTHON_BIN"
	"$ROOT_DIR/src/decision_tree/decision_tree.py"
	--features "$FEATURES_PATH"
	--labels "$LABELS_PATH"
	--rankings "$RANKINGS_PATH"
	--seed "$SEED"
	--plot-max-depth "$PLOT_MAX_DEPTH"
	--top-n-importances "$TOP_N_IMPORTANCES"
	--output-dir "$REPORTS_DIR"
	--export-all-trees
	--top-k "${TOP_K_ARR[@]}"
	--depths "${DEPTH_ARR[@]}"
)

if [[ $# -gt 0 ]]; then
	CMD+=("$@")
fi

echo "=================================================" | tee -a "$LOG_FILE"
echo "=== Decision Tree batch started: $TIMESTAMP ===" | tee -a "$LOG_FILE"
echo "Python: $PYTHON_BIN" | tee -a "$LOG_FILE"
echo "Output dir: $REPORTS_DIR" | tee -a "$LOG_FILE"
echo "Top-k: $TOP_K_VALUES" | tee -a "$LOG_FILE"
echo "Depths: $DEPTH_VALUES" | tee -a "$LOG_FILE"
echo "=================================================" | tee -a "$LOG_FILE"

if [[ "$RUN_IN_BACKGROUND" -eq 1 ]]; then
	nohup "${CMD[@]}" >> "$LOG_FILE" 2>&1 &
	PID=$!
	echo "$PID" > "$PID_FILE"
	echo "[OK] Processo iniciado em background"
	echo "PID: $PID"
	echo "PID file: $PID_FILE"
	echo "Log: $LOG_FILE"
	echo "Para acompanhar: tail -f $LOG_FILE"
else
	echo "[INFO] A correr em foreground..."
	echo "Log: $LOG_FILE"
	"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
fi
