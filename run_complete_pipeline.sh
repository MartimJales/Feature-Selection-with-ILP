#!/bin/bash
# Pipeline completo otimizado (sem tarefas duplicadas)
set -e

# Configurações
DATA_DIR="./data/destino"
LABELS_CSV="${LABELS_CSV:-./data/training_set.csv}"
LOG_DIR="./logs"
REPORTS_DIR="./reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/complete_pipeline_$TIMESTAMP.log"
FEATURES_FILE="./reports/extracted_features.parquet"  # Parquet é 50x mais rápido que CSV
EXTRACT_MODE="${1:-sample}"  # "sample" (100 arquivos) ou "full" (dataset completo)

mkdir -p "$DATA_DIR" "$LOG_DIR" "$REPORTS_DIR"

echo "=============================================" | tee "$LOG_FILE"
echo "=== Pipeline Completo iniciado em $TIMESTAMP ===" | tee -a "$LOG_FILE"
echo "Modo de extração: $EXTRACT_MODE" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"

# [0/6] Instalar dependências
echo "[0/6] Instalando dependências..." | tee -a "$LOG_FILE"
pip install -q -r requirements.txt >> "$LOG_FILE" 2>&1
echo "✓ Dependências instaladas" | tee -a "$LOG_FILE"

# [1/6] Descompactar dataset
echo "[1/6] Verificando arquivos..." | tee -a "$LOG_FILE"
EXISTING_JSON=$(find "$DATA_DIR" -maxdepth 1 -type f -name "*.json" 2>/dev/null | wc -l)

if [ "$EXISTING_JSON" -ge 75000 ]; then
    echo "✓ Dataset completo já existe ($EXISTING_JSON arquivos JSON encontrados)" | tee -a "$LOG_FILE"
    EXTRACTED=$EXISTING_JSON
else
    echo "Descompactando arquivos..." | tee -a "$LOG_FILE"
    if [ -f "data/training_set_features.zip" ]; then
        if [ "$EXTRACT_MODE" = "full" ]; then
            echo "Extraindo dataset completo..." | tee -a "$LOG_FILE"
            unzip -oq data/training_set_features.zip -d "$DATA_DIR" >> "$LOG_FILE" 2>&1
        else
            echo "Extraindo amostra de 100 arquivos aleatórios..." | tee -a "$LOG_FILE"
            zipinfo -1 data/training_set_features.zip | shuf | head -n 100 | xargs unzip -oq data/training_set_features.zip -d "$DATA_DIR" >> "$LOG_FILE" 2>&1
        fi
        EXTRACTED=$(find "$DATA_DIR" -maxdepth 1 -type f -name "*.json" 2>/dev/null | wc -l)
        echo "✓ Arquivos descompactados: $EXTRACTED em $DATA_DIR" | tee -a "$LOG_FILE"
    else
        echo "✗ ERRO: data/training_set_features.zip não encontrado" | tee -a "$LOG_FILE"
        exit 1
    fi
fi

# [2/6] Verificar arquivos JSON
echo "[2/6] Verificando arquivos JSON..." | tee -a "$LOG_FILE"
JSON_COUNT=$(find "$DATA_DIR" -maxdepth 1 -type f -name "*.json" | wc -l)
if [ "$JSON_COUNT" -eq 0 ]; then
    echo "✗ ERRO: Nenhum arquivo JSON encontrado em $DATA_DIR/" | tee -a "$LOG_FILE"
    exit 1
fi
echo "✓ Encontrados $JSON_COUNT arquivos JSON" | tee -a "$LOG_FILE"

# [3/6] Extrair features dos JSONs
echo "[3/6] Extraindo features dos JSONs..." | tee -a "$LOG_FILE"
python3 -m src.features.extractor "$DATA_DIR" --output "$FEATURES_FILE" >> "$LOG_FILE" 2>&1
echo "✓ Features extraídas para $FEATURES_FILE" | tee -a "$LOG_FILE"

# [4/6] Carregar dados (features + labels) - UMA VEZ SÓ
echo "[4/6] Carregando dataset com labels..." | tee -a "$LOG_FILE"
python3 -m src.data.loader "$DATA_DIR" "$LABELS_CSV" --features "$FEATURES_FILE" >> "$LOG_FILE" 2>&1
echo "✓ Dataset carregado com sucesso" | tee -a "$LOG_FILE"

# [5/6] Feature selection (IG + MI)
echo "[5/6] Analisando features (IG + MI)..." | tee -a "$LOG_FILE"
python3 -m src.analysis.feature_analysis >> "$LOG_FILE" 2>&1
echo "✓ Análise concluída. Resultados em ./reports/feature_analysis/" | tee -a "$LOG_FILE"

# [6/6] Treinamento (comentado)
# echo "[6/6] Executando treinamento..." | tee -a "$LOG_FILE"
# if python3 -c "import src.models.train" 2>/dev/null; then
#     python3 -m src.models.train >> "$LOG_FILE" 2>&1
#     echo "✓ Treinamento concluído" | tee -a "$LOG_FILE"
# else
#     echo "⚠ Módulo src.models.train não configurado - pulando" | tee -a "$LOG_FILE"
# fi

# Relatório final
echo "" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"
echo "=== Pipeline Completo finalizado com sucesso ===" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"
echo "✓ Samples processados: $JSON_COUNT" | tee -a "$LOG_FILE"
echo "✓ Features extraídas: $FEATURES_FILE" | tee -a "$LOG_FILE"
echo "✓ Análise de features: ./reports/feature_analysis/feature_rankings.csv" | tee -a "$LOG_FILE"
echo "✓ Gráficos: ./reports/feature_analysis/feature_importance.png" | tee -a "$LOG_FILE"
echo "✓ Log completo: $LOG_FILE" | tee -a "$LOG_FILE"
