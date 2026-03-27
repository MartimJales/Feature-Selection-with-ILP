#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/decision_tree}"
REPORTS_DIR="${REPORTS_DIR:-$ROOT_DIR/reports/decision_tree}"
TIMESTAMP="${TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/decision_tree_${TIMESTAMP}.log}"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
FEATURES_PATH="${FEATURES_PATH:-$ROOT_DIR/reports/extracted_features.parquet}"
LABELS_PATH="${LABELS_PATH:-$ROOT_DIR/data/training_set.csv}"
RANKINGS_PATH="${RANKINGS_PATH:-$ROOT_DIR/reports/feature_analysis/feature_rankings_all.parquet}"
TOP_K_VALUES="${TOP_K_VALUES:-200 500 1000}"
DEPTH_VALUES="${DEPTH_VALUES:-2 3 4 5 6 8 10 12 15 20 25 30 None}"
PLOT_MAX_DEPTH="${PLOT_MAX_DEPTH:-6}"
TOP_N_IMPORTANCES="${TOP_N_IMPORTANCES:-30}"
SEED="${SEED:-42}"

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
echo "Mode: foreground" | tee -a "$LOG_FILE"
echo "Output dir: $REPORTS_DIR" | tee -a "$LOG_FILE"
echo "Top-k: $TOP_K_VALUES" | tee -a "$LOG_FILE"
echo "Depths: $DEPTH_VALUES" | tee -a "$LOG_FILE"
echo "=================================================" | tee -a "$LOG_FILE"

run_job() {
	local started_at finished_at duration status
	started_at="$(date +"%Y-%m-%d %H:%M:%S")"
	echo "[INFO] Job started at $started_at"
	echo "[INFO] Command: ${CMD[*]}"

	set +e
	"${CMD[@]}"
	status=$?
	set -e

	finished_at="$(date +"%Y-%m-%d %H:%M:%S")"
	duration="$SECONDS"
	if [[ "$status" -eq 0 ]]; then
		echo "[SUCESSO] Decision Tree batch terminado com sucesso em $finished_at (duracao: ${duration}s)"
		echo "[SUCESSO] Tabela por depth: $REPORTS_DIR/decision_tree_comparison_by_depth.csv"
	else
		echo "[ERRO] Decision Tree batch terminou com erro (exit code $status) em $finished_at (duracao: ${duration}s)"
	fi
	return "$status"
}

echo "[INFO] A correr em foreground..." | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
run_job 2>&1 | tee -a "$LOG_FILE"
