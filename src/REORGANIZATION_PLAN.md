# Proposta de Reorganização da Pasta `src/` e do Repositório Inteiro

> Documento único de trabalho para este problema. Esta versão substitui a análise separada que existia noutro ficheiro.

## Problema Atual

A pasta `src/` contém múltiplas ideias e workflows em paralelo, muitas delas ultrapassadas ou parcialmente implementadas:

```
src/
├── analysis/              # Análises globais (incompletas)
├── data/                  # Loaders de dados
├── decision_tree/         # Validação com decision tree (não integrado)
├── entropy_knn/           # ✅ Pipeline ATIVO (cluster scoring)
├── features/              # Helpers de features (pouco claro)
├── idea1/                 # ❌ KNN clustering (deprecated, substitute para entropy_knn)
├── idea2/                 # ⚠️ Feature windows + ILP (reutilizável, mas desorganizado)
└── padtai/                # ⚠️ Wrapper PADTAI (útil se idea2 for mantido)
```

**Resultado**: confusão sobre qual é o "primary workflow" e muito código duplicado/obsoleto.

---

## Análise Honesta do Repositório Inteiro

### O que é realmente necessário manter

- `src/entropy_knn/` — é o workflow principal actual.
- `src/analysis/entropy_knn_visualizations/` — contém a análise per-cluster que já estivemos a consolidar.
- `src/idea2/` — vale a pena reaproveitar para a futura camada de ILP.
- `src/decision_tree/` — vale a pena reaproveitar para a etapa de validação.
- `PADTAI/` — deve ficar como submodule, porque é a base externa do ILP.
- `README.md`, `requirements.txt`, `data/training_set.csv` — fundamentais.

### O que está extra ou desactualizado

- `main.py` e `config.py` — são para tracks de avaliação/adversarial, não para o workflow de tese.
- `run_idea1_knn.py`, `run_idea1_knn_sweep.py` — ideia antiga substituída por `entropy_knn`.
- `run_complete_pipeline.sh`, `run_pipeline.sh`, `run_multiple_analysis.sh`, `extract_and_load.sh` — orquestrações antigas.
- `plot_features_horizontal.py`, `venn_ig_mi_features.py`, `convert_report_format.py` — análises ad-hoc que não fazem parte do fluxo actual.
- `BrainStorming/` — notas históricas que já não são fonte de verdade.
- `track_1/`, `track_2/`, `track_3/` — código de adversarial attacks, fora do âmbito do workflow actual.
- `logs/` e `data/destino/` — artefactos temporários.

### O que pode ser aproveitado sem recriar do zero

- A lógica de clustering/scoring de `entropy_knn`.
- O loader unificado baseado em JSON/parquet na análise.
- A camada de validação com `DecisionTreeClassifier`.
- A pipeline de ILP e sampling de `idea2`.
- O executor `PADTAI` como motor externo.

### Decisão honesta

Se o objectivo for a tese sobre seleção de features + ILP, então **não precisamos** das tracks 1/2/3 nem do código de adversarial attacks. O que precisamos mesmo é:

1. `entropy_knn` como Stage 1.
2. Validação como Stage 2.
3. ILP como Stage 3.

Tudo o resto deve ser tratado como legado, referência histórica, ou removido.

---

## Solução Proposta

Criar uma estrutura clara e linear que reflects o workflow científico real:

```
src/
├── entropy_knn/                    # ✅ MAIN PIPELINE: cluster feature scoring
│   ├── __init__.py
│   ├── clustering/
│   │   ├── __init__.py
│   │   └── clustering.py            # Cluster logic (já existe em src/entropy_knn)
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── scorer.py                # Score calculation (5 methods)
│   │   └── common.py                # METHODS, METHOD_LABELS, helpers (reutilizar)
│   └── visualizations/
│       ├── __init__.py
│       └── common.py                # (já existe)
│
├── analysis/
│   └── entropy_knn_analysis/        # ✅ Per-cluster feature-vs-method analysis
│       ├── __init__.py
│       ├── runners/
│       │   ├── __init__.py
│       │   └── run_cluster_analysis.py  # MAIN RUNNER (renomear run_cluster_top_feature_analysis.py)
│       ├── loaders/
│       │   ├── __init__.py
│       │   └── data_sources.py       # (reutilizar existente)
│       └── visualizations/
│           ├── __init__.py
│           ├── heatmap.py
│           ├── bars.py
│           ├── scatter.py
│           └── helpers.py             # normalize_scores, build_feature_summary, etc.
│
├── validation/                       # ✅ STAGE 2: Validar top-features com classifier
│   ├── __init__.py
│   ├── classifier.py                 # (adaptar de src/decision_tree/decision_tree.py)
│   ├── metrics.py                    # (extrair métricas: AUC, F1, ACC, etc.)
│   └── runners/
│       ├── __init__.py
│       └── run_validation.py          # Treina classifier com top-N features
│
├── ilp_pipeline/                     # ⚠️ STAGE 3 (future): Send to ILP
│   ├── __init__.py
│   ├── data_loader.py                # (reutilizar de src/idea2/data_loader.py)
│   ├── ilp_runner.py                 # (reutilizar de src/idea2/ilp_runner.py)
│   ├── sampling.py                   # (reutilizar de src/idea2/sampling.py)
│   ├── window_generator.py            # (reutilizar de src/idea2/window_generator.py)
│   └── runners/
│       ├── __init__.py
│       └── run_ilp_pipeline.py        # Orchestrator: entropy_knn → validation → ILP
│
├── common/                           # Reutilizável
│   ├── __init__.py
│   ├── loaders.py                    # Data loading (unificar src/data/ e idea2/data_loader.py)
│   └── paths.py                      # Path management (defaults para reports/, data/)
│
└── legacy/                           # ❌ Deprecated (manter para referência histórica)
    ├── __init__.py
    ├── idea1_knn_experiment.py        # (mover de src/idea1/knn_experiment.py)
    ├── padtai_wrapper.py              # (mover de src/padtai/)
    └── README_LEGACY.md               # "These were exploratory, use entropy_knn instead"
```

---

## Workflow Linear (Como Usar)

```
STAGE 1: Feature Scoring per Cluster
    python3 src/analysis/entropy_knn_analysis/runners/run_cluster_analysis.py \
        --cluster-json-dir reports/entropy_knn/score_only/cluster_500/seed_42 \
        --output-dir reports/entropy_knn/analysis/per_cluster_feature_vs_method

    Output:
    - reports/entropy_knn/analysis/per_cluster_feature_vs_method/cluster_0/
        - top_feature_candidates.csv
        - feature_method_summary.csv
        - visualizations/ (heatmap, bars, agreement, scatter)

STAGE 2: Validação com Classifier (NEW)
    python3 src/validation/runners/run_validation.py \
        --features reports/entropy_knn/analysis/per_cluster_feature_vs_method/cluster_0/top_feature_candidates.csv \
        --training-data data/training_set.csv \
        --output-dir reports/validation/cluster_0 \
        --top-k 10,20,30

    Output:
    - reports/validation/cluster_0/
        - metrics_top10.csv (AUC, F1, ACC, etc.)
        - metrics_top20.csv
        - metrics_top30.csv
        - confusion_matrices.png

STAGE 3: ILP Pipeline (FUTURE, when needed)
    python3 src/ilp_pipeline/runners/run_ilp_pipeline.py \
        --features-dir reports/entropy_knn/analysis/per_cluster_feature_vs_method \
        --top-k 15 \
        --ilp-timeout 300

    Output:
    - reports/ilp/cluster_0/
        - ilp_rules.csv
        - ilp_metrics.csv
```

---

## Código a Reutilizar (Mapeamento Explícito)

| Origem | Destino | Notas |
|--------|---------|-------|
| `src/entropy_knn/visualizations/common.py` | `src/entropy_knn/scoring/common.py` | METHODS, METHOD_LABELS constants |
| `src/analysis/entropy_knn_visualizations/data_sources.py` | `src/analysis/entropy_knn_analysis/loaders/data_sources.py` | JSON/parquet loader |
| `src/analysis/entropy_knn_visualizations/run_cluster_top_feature_analysis.py` | `src/analysis/entropy_knn_analysis/runners/run_cluster_analysis.py` | **Renomear**, import helpers from visualizations/ |
| `src/idea2/data_loader.py` | `src/common/loaders.py` | Unificar loaders (treino/teste split) |
| `src/idea2/ilp_runner.py` | `src/ilp_pipeline/ilp_runner.py` | Executor PADTAI |
| `src/idea2/sampling.py` | `src/ilp_pipeline/sampling.py` | Estratégia de sampling |
| `src/decision_tree/decision_tree.py` | `src/validation/classifier.py` | Extrair DecisionTreeClassifier + metrics |

---

## Passos de Migração (Para o Colega)

### Phase 1: Cleanup (Removar ambiguidade)
1. Mover `src/idea1/` → `src/legacy/idea1_knn_experiment/`
2. Criar `src/legacy/README_LEGACY.md` com descrição do que foi tentado
3. Confirmar que `src/entropy_knn/` é a source of truth para clustering/scoring

### Phase 2: Reorganizar Entropy KNN (Não mexer na lógica, só em paths)
1. Criar `src/entropy_knn/scoring/` → mover constants de `src/entropy_knn/visualizations/common.py`
2. Manter visualizations como estão, mas documentar que são "for reporting only"
3. Atualizar imports em `src/analysis/entropy_knn_visualizations/` para apontar para novo local

### Phase 3: Criar Analysis Pipeline
1. Renomear `src/analysis/entropy_knn_visualizations/run_cluster_top_feature_analysis.py` → `src/analysis/entropy_knn_analysis/runners/run_cluster_analysis.py`
2. Mover `data_sources.py` para `src/analysis/entropy_knn_analysis/loaders/`
3. Extrair helpers de visualização para `src/analysis/entropy_knn_analysis/visualizations/helpers.py`
4. Atualizar todos os imports relativos

### Phase 4: Criar Validation Pipeline (NEW)
1. Copiar `src/decision_tree/decision_tree.py` → `src/validation/classifier.py`
2. Limpar (manter só DecisionTreeClassifier + metrics calculation)
3. Criar `src/validation/runners/run_validation.py`:
   - Input: CSV de top-features (de STAGE 1)
   - Training data: `data/training_set.csv`
   - Output: Métricas de accuracy/AUC/F1 para diferentes top-K thresholds
4. Gerar relatório: "How many features do we need for 90% AUC?"

### Phase 5: Preparar ILP Pipeline (Não implementar ainda, just reorganize)
1. Mover `src/idea2/` → `src/ilp_pipeline/`
2. Renomear `idea2_pipeline.py` → `ilp_orchestrator.py`
3. Criar stub em `src/ilp_pipeline/runners/run_ilp_pipeline.py`
4. Documentar: "Ready to integrate after validation stage validates top-K"

### Phase 6: Documentação Final
1. Criar `src/README.md` com workflow diagram (texto ASCII)
2. Criar `src/WORKFLOW.md` com exemplos de comando end-to-end
3. Atualizar imports em `requirements.txt` (se alguma dependência foi perdida)

---

## Checklist para o Colega

- [ ] Phase 1: Move idea1 to legacy, confirm entropy_knn is primary
- [ ] Phase 2: Reorganize entropy_knn (paths only, logic unchanged)
- [ ] Phase 3: Reorganize analysis pipeline (rename + move helpers)
- [ ] Phase 4: Create validation pipeline (new stage 2)
- [ ] Phase 5: Reorganize ILP as future stage (no impl. needed yet)
- [ ] Phase 6: Document workflow with ASCII diagram + example commands
- [ ] Test: Run end-to-end STAGE 1 + STAGE 2 with 1 cluster (--max-clusters 1)
- [ ] Commit: Message = "refactor: reorganize src/ into linear workflow (entropy_knn → validation → ilp)"

---

## Expected Result

```bash
# Single command to run the full pipeline (STAGE 1 + STAGE 2)
./run_full_analysis.sh --cluster-id 0 --top-k 15 --validation-only

# Or step by step
python3 src/analysis/entropy_knn_analysis/runners/run_cluster_analysis.py ...  # STAGE 1
python3 src/validation/runners/run_validation.py ...                           # STAGE 2
python3 src/ilp_pipeline/runners/run_ilp_pipeline.py ...                       # STAGE 3 (future)
```

**Outcome**: A colleague can understand the workflow in 5 minutes by reading the directory structure.
