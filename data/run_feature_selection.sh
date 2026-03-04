#!/bin/bash
# filepath: /home/jales/IST/Tese/elsa-cybersecurity/run_pipeline.sh

set -e

# Configurações
DATA_DIR="./data/raw"
LOG_DIR="./logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/pipeline_$TIMESTAMP.log"
EXTRACT_MODE="${1:-sample}"  # "sample" (padrão: 100 arquivos) ou "full" (dataset inteiro)

mkdir -p "$DATA_DIR" "$LOG_DIR"

echo "=== Pipeline iniciado em $TIMESTAMP ===" | tee "$LOG_FILE"
echo "Modo de extração: $EXTRACT_MODE" | tee -a "$LOG_FILE"

# Descompactar dataset
echo "[1/3] Descompactando arquivos..." | tee -a "$LOG_FILE"
if [ -f "training_set_features.zip" ]; then
    if [ "$EXTRACT_MODE" = "full" ]; then
        echo "Extraindo dataset completo..." | tee -a "$LOG_FILE"
        unzip training_set_features.zip -d "$DATA_DIR/track_2" >> "$LOG_FILE" 2>&1
    else
        echo "Extraindo amostra de 100 arquivos aleatórios..." | tee -a "$LOG_FILE"
        unzip training_set_features.zip $(zipinfo -1 training_set_features.zip | shuf | head -n 100) -d "$DATA_DIR/track_2" >> "$LOG_FILE" 2>&1
    fi
    echo "✓ Arquivos descompactados em $DATA_DIR/track_2" | tee -a "$LOG_FILE"
else
    echo "⚠ training_set_features.zip não encontrado" | tee -a "$LOG_FILE"
fi

# Carregar dados e executar pipeline
echo "[2/3] Carregando dados..." | tee -a "$LOG_FILE"
python3 -m src.data.loader >> "$LOG_FILE" 2>&1

echo "[3/3] Executando treinamento..." | tee -a "$LOG_FILE"
python3 -m src.models.train >> "$LOG_FILE" 2>&1

echo "=== Pipeline concluído com sucesso ===" | tee -a "$LOG_FILE"
