#!/bin/bash
# Script de extração e carregamento de dados

set -e

# Configurações
LOG_DIR="./logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/extract_and_load_$TIMESTAMP.log"
EXTRACT_MODE="${1:-sample}"  # "sample" (padrão: 100 arquivos) ou "full" (dataset inteiro)

mkdir -p "$LOG_DIR"

echo "=== Pipeline iniciado em $TIMESTAMP ===" | tee "$LOG_FILE"
echo "Modo de extração: $EXTRACT_MODE" | tee -a "$LOG_FILE"

# Descompactar dataset
echo "[1/2] Descompactando arquivos..." | tee -a "$LOG_FILE"
if [ -f "data/training_set_features.zip" ]; then
    if [ "$EXTRACT_MODE" = "full" ]; then
        echo "Extraindo dataset completo..." | tee -a "$LOG_FILE"
        unzip -q data/training_set_features.zip -d data/destino >> "$LOG_FILE" 2>&1
    else
        echo "Extraindo amostra de 100 arquivos aleatórios..." | tee -a "$LOG_FILE"
        zipinfo -1 data/training_set_features.zip | shuf | head -n 100 | xargs unzip -q data/training_set_features.zip -d data/destino >> "$LOG_FILE" 2>&1
    fi
    EXTRACTED=$(ls data/destino | wc -l)
    echo "✓ Arquivos descompactados: $EXTRACTED arquivos em data/destino" | tee -a "$LOG_FILE"
else
    echo "⚠ data/training_set_features.zip não encontrado" | tee -a "$LOG_FILE"
    exit 1
fi

# Carregar dados
echo "[2/2] Carregando dados..." | tee -a "$LOG_FILE"
python3 -m src.data.loader >> "$LOG_FILE" 2>&1
echo "✓ Dados carregados com sucesso" | tee -a "$LOG_FILE"

echo "=== Pipeline concluído com sucesso ===" | tee -a "$LOG_FILE"
