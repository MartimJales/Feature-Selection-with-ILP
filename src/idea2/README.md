# Idea 2: Feature Windows + ILP com Sampling Global

## Descrição Geral

Implementação completa da **Ideia 2** — um baseline rápido e controlado para validar a viabilidade da ILP sob forte limitação de dimensionalidade.

A estratégia é simples mas poderosa:
1. Ordenar features por IG (Information Gain) global
2. Criar janelas fixas de 30 features consecutivas
3. Para cada janela + tamanho de amostra + seed, executar ILP com PADTAI
4. Consolidar métricas e regras para análise

## Estrutura de Código

```
src/idea2/
├── __init__.py           # Exports principais
├── pipeline.py           # Orquestrador principal (Idea2Pipeline)
├── data_loader.py        # Carregamento e validação de dados
├── window_generator.py   # Geração de janelas de features
├── sampling.py           # Estratégia de sampling estratificado
├── ilp_runner.py         # Executor PADTAI com logging
└── aggregator.py         # Consolidação e reporting
```

## Uso Rápido

### Phase A (Prova de viabilidade - ~5-10 min)

```bash
python run_idea2.py --phase a --seeds-a 2
```

Testa:
- 3 janelas: [1-30, 31-60, 61-90]
- 1 tamanho de amostra (médio 50%)
- 2 seeds
- **Total: 6 runs**

### Phase B (Benchmark principal - ~1-2 horas)

```bash
python run_idea2.py --phase b --seeds-b 3
```

Testa:
- 10 janelas: [1-30, 31-60, ..., 271-300]
- 3 tamanhos de amostra: [10%, 50%, 100%]
- 3 seeds
- **Total: 90 runs**

### Full (A + B - ~2-3 horas)

```bash
python run_idea2.py --phase full --seeds-a 2 --seeds-b 3
```

## Opções de Linha de Comando

```
--phase {a|b|full}       Fase a correr (default: a)
--seeds-a INT            Nº de seeds para Phase A (default: 2)
--seeds-b INT            Nº de seeds para Phase B (default: 3)
--window-size INT        Tamanho de cada janela (default: 30)
--timeout INT            Timeout max por run em segundos (default: 1800)
```

## Outputs

### Tabelas Consolidadas

- `reports/idea2/summary.csv`
  - Cada linha = 1 run
  - Colunas: window_id, sample_size, seed, n_rules, status, elapsed_time, métricas

- `reports/idea2/by_window.csv`
  - Agregado por janela
  - Mostra: média/min/max de rules, success rate, tempo médio

- `reports/idea2/by_sample_size.csv`
  - Agregado por tamanho de amostra
  - Mostra impacto da escala nos resultados

- `reports/idea2/top_runs.csv`
  - Top-15 runs por qualidade (success × n_rules × F1)

### Regras e Logs

- `reports/idea2/rules/`
  - Ficheiros `.txt` com regras de cada run bem-sucedida
  - Naming: `rules_w{window_id}_s{seed}.txt`

- `logs/idea2/idea2.log`
  - Log completo de execução

## Fluxo de Dados

```
training_set.csv + feature_rankings_all.parquet
         ↓
   Idea2Pipeline.initialize()
         ↓
   FeatureWindowGenerator: 5 janelas (primeira fase)
         ↓
   Para cada (window, sample_size, seed):
       ├→ SamplingStrategy: stratified sample
       ├→ ILPRunner: executa PADTAI
       └→ ResultsAggregator: guarda resultado
         ↓
   ResultsAggregator.consolidate_results()
   ├→ summary.csv
   ├→ by_window.csv
   ├→ by_sample_size.csv
   ├→ top_runs.csv
   ├→ rules/
   └→ config.json
```

## Métricas Capturadas

### Por run:
- **window_id**: Índice da janela
- **sample_size**: Nº de samples usados
- **seed**: Random seed
- **n_features**: 30 (fixo)
- **n_rules**: Nº de regras descobertas
- **status**: success|timeout|error|no_solution
- **elapsed_time**: Tempo total
- **solver_time**: Tempo só do solver PADTAI
- **train_accuracy / precision / recall / f1**: Métricas (se calculadas)

### Agregadas:
- **success_rate**: % de runs que encontraram solução
- **mean_n_rules**: Nº médio de regras por janela
- **mean_elapsed_time**: Tempo médio

## Perguntas que Responde

1. ✓ Com 30 features por corrida, ILP consegue produzir regras estáveis?
2. ✓ Top-30 é claramente melhor que blocos seguintes?
3. ✓ Até onde blocos 31-60, 61-90, etc. trazem sinal útil?
4. ✓ Melhor compromisso entre qualidade e custo?
5. ✓ Ponto de saturação?

## Critérios de Decisão (go/no-go)

### ✓ Avançar com Idea2:
- ILP encontra solução em >70% das runs
- Tempos aceitáveis para execução em lote
- Qualidade estável entre seeds

### ↻ Pivotar para Idea 1 (clustering):
- Instabilidade forte entre seeds
- Degradação rápida após top-30
- Regras frágeis ou pouco interpretáveis
- Custo computacional inviável

## Workflow Recomendado

1. **Comece com Phase A** (rápido, ~10 min)
   - Valida pipeline
   - Detecta problemas cedo

2. **Analise resultados Phase A**
   - Tempo médio por run?
   - Taxa de sucesso?
   - Se <30s por run, prossiga; se >5 min, ajuste timeout

3. **Execute Phase B** (comprehensive)
   - Mantém os mesmos parâmetros
   - Expande a mais janelas e tamanhos

4. **Analise padrões**
   - Qualidade melhora com mais features?
   - Melhora com samples maiores?
   - Janelas top-30 são melhores?

5. **Decida próximos passos**
   - Se resultados bons: usar top-K janelas em production
   - Se instável: pivotar para Idea 1 (clustering local)
   - Se promissor: combinar ambas (híbrido)

## Troubleshooting

### PADTAI não encontrado
```
FileNotFoundError: PADTAI directory not found
```
→ Verificar se `./PADTAI/` existe e tem `padtai.py`

### Timeout frequente
```
Status: timeout
```
→ Aumentar `--timeout` ou reduzir sample size

### Memória insuficiente
→ Reduzir `--window-size` ou usar samples menores

### Label column não detectada
```
ValueError: Could not detect label column
```
→ Verificar nome da coluna em training_set.csv (esperado: Label, label, target, y)

## Integração com Idea 1

Depois de Phase A/B concluídos, resultados podem informar:
- Quais janelas são mais promissoras
- Se clustering local agregado pode melhorar
- Threshold realista para IG local dentro de clusters

## Referências

- [Idea2.md](../../Idea2.md) - Proposta exaustiva
- [Idea1.md](../../idea1.md) - Proposta de clustering (avançada)
- PADTAI: `./PADTAI/README.md`
