#!/bin/bash
set -e

# Configurações
DATA_DIR="./data/destino"
LABELS_CSV="${LABELS_CSV:-./data/training_set.csv}"
LOG_DIR="./logs"
REPORTS_DIR="./reports"
NUM_SECTIONS="${NUM_SECTIONS:-6}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/multi_analysis_$TIMESTAMP.log"

mkdir -p "$DATA_DIR" "$LOG_DIR" "$REPORTS_DIR"

echo "=== Análise Múltipla iniciada em $TIMESTAMP ===" | tee "$LOG_FILE"

# Instalar dependências
echo "[0/3] Instalando dependências..." | tee -a "$LOG_FILE"
pip install -q -r requirements.txt >> "$LOG_FILE" 2>&1
echo "✓ Dependências instaladas" | tee -a "$LOG_FILE"

# Extrair features uma única vez
echo "[1/3] Extraindo features dos JSONs..." | tee -a "$LOG_FILE"
python3 -m src.features.extractor "$DATA_DIR" >> "$LOG_FILE" 2>&1

# Contar features
MAX_FEATURES=$(head -1 ./reports/extracted_features.csv | tr ',' '\n' | wc -l)
echo "✓ Features extraídas - Total: $MAX_FEATURES features" | tee -a "$LOG_FILE"

# Gerar valores de top-k dinâmicos
TOP_K_VALUES=()
for ((i=1; i<NUM_SECTIONS; i++)); do
    TOP_K_VALUES+=($((MAX_FEATURES * i / NUM_SECTIONS)))
done
TOP_K_VALUES+=($MAX_FEATURES)  # Última secção com todas as features

# Executar análise IG + MI para cada valor de top-k
echo "[2/3] Executando análises com $NUM_SECTIONS secções..." | tee -a "$LOG_FILE"
for top_k in "${TOP_K_VALUES[@]}"; do
    echo "  → Analisando com top-k=$top_k..." | tee -a "$LOG_FILE"
    python3 -m src.analysis.feature_analysis \
        --json-dir "$DATA_DIR" \
        --labels-path "$LABELS_CSV" \
        --output-dir "$REPORTS_DIR/feature_analysis" \
        --top-k "$top_k" >> "$LOG_FILE" 2>&1
    echo "    ✓ Análise concluída para top-k=$top_k" | tee -a "$LOG_FILE"
done

# Relatório final
echo "" | tee -a "$LOG_FILE"
echo "=== Pipeline concluído com sucesso ===" | tee -a "$LOG_FILE"
echo "✓ Secções: $NUM_SECTIONS | Valores de top-k: ${TOP_K_VALUES[@]}" | tee -a "$LOG_FILE"
echo "✓ Relatórios gerados em: $REPORTS_DIR/feature_analysis/" | tee -a "$LOG_FILE"
echo "✓ Log completo: $LOG_FILE" | tee -a "$LOG_FILE"
