# Idea 2 — Quick Start Guide

## O que foi implementado?

Uma pipeline completa e modular para testar a viabilidade da ILP com limitação de dimensionalidade, usando janelas de 30 features e sampling global.

## Estrutura de Código

```
src/idea2/
├── __init__.py              # Exports
├── README.md                # Documentação detalhada
├── pipeline.py              # Classe Idea2Pipeline (orquestrador)
├── data_loader.py           # Carregamento e validação Idea2DataLoader
├── window_generator.py      # Geração de janelas FeatureWindowGenerator
├── sampling.py              # Sampling estratificado SamplingStrategy
├── ilp_runner.py            # Executor PADTAI ILPRunner + ILPRunResult
└── aggregator.py            # Consolidação ResultsAggregator

run_idea2.py                 # Script principal executável

reports/idea2/               # (criado na execução)
├── summary.csv             # Tabela com todos os runs
├── by_window.csv           # Agregado por window
├── by_sample_size.csv      # Agregado por tamanho
├── top_runs.csv            # Top-15 runs por qualidade
├── rules/                  # Regras descobertas
├── config.json             # Configuração usada
└── (logs em logs/idea2/idea2.log)
```

## Uso Rápido

### Pré-requisitos

✓ Python 3.10+
✓ Dependências instaladas (pandas, numpy, scikit-learn, etc.)
✓ PADTAI funcional em `./PADTAI/`
✓ `data/training_set.csv` existente
✓ `reports/feature_analysis/feature_rankings_all.parquet` existente

### Opção 1: Prova de Viabilidade (5-10 minutos)

```bash
python run_idea2.py --phase a --seeds-a 2
```

✓ Testa 3 janelas × 1 tamanho × 2 seeds = 6 runs
✓ Outputs em `reports/idea2/`

### Opção 2: Benchmark Principal (1-2 horas)

```bash
python run_idea2.py --phase b --seeds-b 3
```

✓ Testa 10 janelas × 3 tamanhos × 3 seeds = 90 runs
✓ Mais abrangente, dados para decisão final

### Opção 3: Full Pipeline (2-3 horas)

```bash
python run_idea2.py --phase full --seeds-a 2 --seeds-b 3
```

✓ Combina A + B
✓ Máxima cobertura

## O que esperar?

### Outputs Principais

Depois de executar, terás:

1. **summary.csv** — Cada linha é um run
   ```
   window_id, sample_size, seed, n_rules, status, elapsed_time, ...
   0, 5000, 0, 12, success, 45.3
   1, 5000, 1, 8, success, 38.2
   ...
   ```

2. **by_window.csv** — Agregado por janela
   ```
   window_id, n_rules_mean, success_rate, elapsed_time_mean
   0, 10.5, 1.0, 41.75
   1, 9.2, 0.67, 52.1
   ...
   ```

3. **top_runs.csv** — Melhores runs por qualidade
   - Janelas mais promissoras
   - Tamanhos de amostra que funcionam bem
   - Estabilidade entre seeds

4. **rules/** — Ficheiros com regras de cada run bem-sucedido

## Interpretação Rápida de Resultados

### Pergunta 1: "Funciona a ILP com 30 features?"

Olha para **success_rate** em `by_window.csv`:
- > 80% → ✓ Sim, muito bem
- 50-80% → ~ Parcialmente, pode melhorar
- < 50% → ✗ Não, rever estratégia

### Pergunta 2: "Top-30 é melhor que resto?"

Compara **n_rules_mean** entre janelas:
- W0 (features 1-30) >> W1 (31-60) → ✓ Top-30 claramente melhor
- Degradação gradual → ~ Ambos trazem sinal útil
- W0 ≈ W1 → ~ IG global tem limitações

### Pergunta 3: "Qual é o custo?"

Olha para **elapsed_time_mean**:
- < 60s → ✓ Viável para execução em lote
- 60-300s → ~ Aceitável mas lento
- > 300s → ✗ Timeout ou muito pesado

### Pergunta 4: "Estável entre seeds?"

Compara entre diferentes seeds em `summary.csv`:
- Mesmas features + tamanho → mesmos rules? → ✓ Estável
- Variação grande → ✗ Frágil, revisar

## Próximos Passos

### Se Phase A Corre Bem (Success Rate > 70%)

1. Execute Phase B para dados mais abrangentes
2. Analise curvas de qualidade vs IG ranking
3. Identifique janelas "ótimas"

### Se Tudo Funcionar Bem (Phase A + B OK)

1. ✓ Idea 2 validada como baseline
2. Proceda com **Idea 1** (clustering + ILP local)
3. Ou combine: use Idea 2 para filtrar, Idea 1 para refinar

### Se Houver Problemas

1. **Timeout frequente?** → Aumentar `--timeout` ou reduzir samples
2. **Poucas regras?** → Dataset pode não ter padrões, rever features
3. **Muito lento?** → Reduzir `--window-size` ou aumentar `--timeout`
4. **Memória?** → Usar samples menores

## Ficheiros Importantes

| Ficheiro | Criado | Descrição |
|----------|--------|-----------|
| `src/idea2/*.py` | ✓ | Módulos core |
| `run_idea2.py` | ✓ | Script executável |
| `src/idea2/README.md` | ✓ | Docs completas |
| `reports/idea2/` | (~) | Criado na 1ª execução |
| `logs/idea2/` | (~) | Criado na 1ª execução |

## Debugging

### Ver logs em tempo real

```bash
tail -f logs/idea2/idea2.log
```

### Testar pipeline sem PADTAI (mock)

Adicione em `ilp_runner.py` um modo mock (não implementado ainda, mas fácil de adicionar)

### Verificar dados carregados

Cada módulo tem `.log` chamadas, vê no ficheiro de log.

## Próxima Integração: Idea 1

Depois de Idea 2 concluída com sucesso:

- Use os dados gerados para decidir parâmetros de clustering
- Implemente Idea 1 em `src/idea1/`
- Combine resultados de ambas

---

**Status**: ✓ Implementação completa
**Pronto para**: Execução Phase A → análise → Phase B → decisão

Boa sorte! 🚀
