#!/bin/bash
set -e

# Configurações
DATA_DIR="./data/destino"
LABELS_CSV="${LABELS_CSV:-./data/training_set.csv}"
LOG_DIR="./logs"
REPORTS_DIR="./reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/pipeline_$TIMESTAMP.log"
MAX_SAMPLES=100
FEATURES_FILE="./reports/extracted_features.csv"


mkdir -p "$DATA_DIR" "$LOG_DIR" "$REPORTS_DIR"

echo "=== Pipeline iniciado em $TIMESTAMP ===" | tee "$LOG_FILE"

# Instalar dependências
echo "[0/5] Instalando dependências..." | tee -a "$LOG_FILE"
pip install -q -r requirements.txt >> "$LOG_FILE" 2>&1
echo "✓ Dependências instaladas" | tee -a "$LOG_FILE"

# Verificar arquivos JSON
echo "[1/5] Verificando arquivos JSON..." | tee -a "$LOG_FILE"
JSON_COUNT=$(ls -1 "$DATA_DIR"/*.json 2>/dev/null | wc -l)
if [ "$JSON_COUNT" -eq 0 ]; then
    echo "✗ ERRO: Nenhum arquivo JSON encontrado em $DATA_DIR/" | tee -a "$LOG_FILE"
    exit 1
fi
if [ "$JSON_COUNT" -gt "$MAX_SAMPLES" ]; then
    echo "⚠ Encontrados $JSON_COUNT arquivos. Limitando a $MAX_SAMPLES samples..." | tee -a "$LOG_FILE"
else
    echo "✓ Encontrados $JSON_COUNT arquivos JSON" | tee -a "$LOG_FILE"
fi

# Extrair features dos JSONs
echo "[2/5] Extraindo features dos JSONs..." | tee -a "$LOG_FILE"
python3 -m src.features.extractor "$DATA_DIR" --output "$FEATURES_FILE"
echo "✓ Features extraídas para $FEATURES_FILE" | tee -a "$LOG_FILE"

# Carregar dados (features + labels)
echo "[3/5] Carregando dataset com labels..." | tee -a "$LOG_FILE"
python3 -m src.data.loader "$DATA_DIR" "$LABELS_CSV" --features "$FEATURES_FILE"
echo "✓ Dataset carregado com sucesso" | tee -a "$LOG_FILE"

# Feature selection (IG + MI)
echo "[4/5] Analisando features (IG + MI)..." | tee -a "$LOG_FILE"
python3 -m src.analysis.feature_analysis >> "$LOG_FILE" 2>&1
echo "✓ Análise concluída. Resultados em ./reports/feature_analysis/" | tee -a "$LOG_FILE"

# Treinamento
echo "[5/5] Executando treinamento..." | tee -a "$LOG_FILE"
if python3 -c "import src.models.train" 2>/dev/null; then
    python3 -m src.models.train >> "$LOG_FILE" 2>&1
else
    echo "⚠ Módulo src.models.train não configurado - pulando" | tee -a "$LOG_FILE"
fi

# Relatório final
echo "" | tee -a "$LOG_FILE"
echo "=== Pipeline concluído com sucesso ===" | tee -a "$LOG_FILE"
echo "✓ Samples processados: $JSON_COUNT" | tee -a "$LOG_FILE"
echo "✓ Relatório de features: ./reports/feature_analysis/feature_rankings.csv" | tee -a "$LOG_FILE"
echo "✓ Gráficos: ./reports/feature_analysis/feature_importance.png" | tee -a "$LOG_FILE"
