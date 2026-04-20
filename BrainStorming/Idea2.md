# Ideia 2 — Proposta Exaustiva (Feature Windows + ILP com Sampling Global)

## 1) Contexto e objetivo

A **Ideia 2** pretende validar, de forma rápida e controlada, a viabilidade da ILP sob forte limitação de dimensionalidade, sem depender já de clustering local.

A estratégia é:

1. Ordenar features por importância global (IG).
2. Definir janelas fixas de tamanho 30 (ex.: 1–30, 31–60, 61–90, ...).
3. Para cada janela, correr ILP sobre amostras do dataset completo.
4. Comparar desempenho, custo computacional e qualidade lógica das regras.

Esta abordagem funciona como **baseline operacional** para:
- medir limites práticos da ILP;
- identificar rapidamente zonas do ranking IG com maior utilidade;
- recolher evidência para orientar a abordagem mais avançada (Ideia 1).

---

## 2) Princípios da abordagem

- **Prioridade à viabilidade ILP**: o foco é manter a dimensão por execução pequena e estável (30 features).
- **Cobertura progressiva do ranking IG**: testar blocos consecutivos para perceber onde está o ganho real.
- **Sampling estratificado**: controlar custo mantendo representatividade malware/benign.
- **Reprodutibilidade**: seeds fixas, versionamento de outputs e logs comparáveis.
- **Critérios explícitos de decisão**: definir antes quando avançar, parar ou ajustar.

---

## 3) Perguntas que esta ideia responde

1. Com 30 features por corrida, a ILP consegue produzir regras estáveis em tempo útil?
2. O bloco top-30 global é claramente melhor que blocos seguintes?
3. Até que ponto blocos 31–60, 61–90, etc. ainda trazem sinal útil?
4. Qual é o melhor compromisso entre qualidade das regras e custo computacional?
5. Qual é o ponto de saturação (mais blocos testados já não melhoram resultados)?

---

## 4) Desenho experimental

## 4.1 Unidades experimentais

- **Janela de features**: blocos consecutivos de 30 features no ranking IG global.
- **Amostra de dados**: subconjunto do dataset completo (estratificado por classe).
- **Run ILP**: uma execução da ILP para um par (janela, amostra, seed).

## 4.2 Matriz de testes recomendada (fase inicial)

Começar com:
- Janelas: [1–30], [31–60], [61–90], [91–120], [121–150]
- Tamanhos de amostra: 3 níveis (ex.: pequeno, médio, grande)
- Seeds: 3 seeds por configuração

Total inicial = 5 janelas × 3 tamanhos × 3 seeds = **45 runs**.

Depois expandir para mais janelas conforme resultados (até onde o orçamento computacional permitir).

---

## 5) Pipeline proposto (sem implementação ainda)

## 5.1 Preparação de inputs

1. Carregar ranking IG global (ordenado decrescente).
2. Alinhar nomes de features com dataset final.
3. Verificar consistência:
   - features em falta;
   - duplicados;
   - colunas constantes;
   - distribuição de classes.

## 5.2 Construção de janelas

Gerar janelas de tamanho fixo 30:
- W1 = features[1:30]
- W2 = features[31:60]
- ...

Regras:
- sem overlap na fase base;
- opcionalmente testar overlap (deslocamento de 15) numa fase posterior.

## 5.3 Sampling global

Para cada run:
1. Fazer sampling estratificado por classe.
2. Preservar proporção malware/benign (ou usar balanceamento controlado, se necessário).
3. Guardar IDs/índices da amostra para reprodutibilidade.

## 5.4 Execução ILP por janela

Para cada janela e amostra:
1. Criar dataset reduzido com 30 features + target.
2. Configurar ILP com limites de tempo/memória definidos.
3. Treinar/induzir regras ILP.
4. Avaliar regras em treino e validação.
5. Guardar artefactos (regras, métricas, logs, tempos, status do solver).

## 5.5 Consolidação

Agregação de resultados por:
- janela;
- tamanho de amostra;
- seed.

Produzir tabelas comparativas e ranking de janelas.

---

## 6) Métricas obrigatórias

## 6.1 Qualidade preditiva

- Accuracy
- Precision
- Recall
- F1
- (Opcional) Balanced Accuracy

## 6.2 Qualidade lógica das regras (ILP-centric)

- nº de regras geradas
- comprimento médio das regras
- cobertura média por regra
- conflito/sobreposição entre regras
- interpretabilidade qualitativa (inspeção manual de top runs)

## 6.3 Robustez

- variância entre seeds
- estabilidade de métricas por janela

## 6.4 Custo computacional

- tempo total por run
- tempo do solver
- memória pico (se disponível)
- taxa de runs que terminam com solução válida

---

## 7) Critérios de decisão (go/no-go)

## 7.1 Critérios para avançar com Ideia 2

Avançar se:
- ILP encontra solução em alta percentagem das runs;
- tempos são aceitáveis para execução em lote;
- pelo menos algumas janelas têm qualidade estável e interpretável.

## 7.2 Critérios para pivotar para Ideia 1 (ou híbrida)

Pivotar se:
- forte instabilidade entre seeds;
- degradação rápida após top-30;
- regras pouco interpretáveis ou muito frágeis;
- custo computacional inviável em escala.

## 7.3 Critério híbrido recomendado

Usar Ideia 2 como filtro inicial para selecionar intervalos IG promissores e aplicar Ideia 1 apenas nesses intervalos.

---

## 8) Riscos e mitigação

1. **Bias do IG global** (não captura contexto local)
   - Mitigação: testar múltiplas janelas e não só top-30.

2. **Amostras não representativas**
   - Mitigação: sampling estratificado + múltiplas seeds + registo de índices.

3. **ILP com timeout frequente**
   - Mitigação: reduzir tamanho de amostra, ajustar parâmetros do solver, limitar complexidade.

4. **Overfitting em runs pequenas**
   - Mitigação: separar validação, repetir por seed e comparar estabilidade.

5. **Explosão de combinações experimentais**
   - Mitigação: plano em fases (curto → médio → expandido), com gate por resultados.

---

## 9) Plano faseado recomendado

## Fase A — Prova de viabilidade (rápida)

- 3 janelas: [1–30], [31–60], [61–90]
- 1 tamanho de amostra (médio)
- 2 seeds

Objetivo: validar pipeline, runtime e capacidade da ILP produzir regras úteis.

## Fase B — Benchmark principal

- 5 a 10 janelas
- 3 tamanhos de amostra
- 3 seeds

Objetivo: mapear performance vs custo e identificar melhores janelas.

## Fase C — Expansão controlada

- mais janelas (até limite computacional)
- eventual overlap de janelas

Objetivo: refinar conclusões e preparar integração com Ideia 1.

---

## 10) Estrutura de outputs sugerida

- `reports/idea2/summary.csv`
  - colunas: janela, sample_size, seed, métricas, tempo, status
- `reports/idea2/by_window.csv`
  - agregações por janela
- `reports/idea2/by_sample_size.csv`
  - agregações por tamanho de amostra
- `reports/idea2/top_runs.csv`
  - melhores configurações
- `logs/idea2/...`
  - logs detalhados por execução
- `reports/idea2/rules/...`
  - regras ILP exportadas por run

---

## 11) Boas práticas de reprodutibilidade

- Fixar seeds em todos os passos aleatórios.
- Versionar configuração experimental (JSON/YAML).
- Guardar hash da lista de features usada por janela.
- Guardar versão de dados, versão de código e timestamp.
- Registar falhas de solver explicitamente (não omitir runs falhadas).

---

## 12) Resultado esperado desta ideia

No final da Ideia 2 deves ter:

1. uma curva clara de **qualidade vs posição no ranking IG**;
2. uma curva de **qualidade vs custo computacional**;
3. um conjunto de janelas candidatas para uso posterior;
4. uma noção objetiva de quanta liberdade a ILP ganha com blocos de 30 features;
5. base sólida para decidir entre:
   - continuar com baseline por janelas,
   - migrar para clustering local (Ideia 1),
   - ou aplicar estratégia híbrida.

---

## 13) Decisão estratégica recomendada

Tratar a Ideia 2 como **baseline obrigatório antes da Ideia 1**:
- é mais simples;
- gera evidência rapidamente;
- reduz risco de engenharia prematura;
- ajuda a definir thresholds realistas para o pipeline de clustering+ILP.

Se os resultados da Ideia 2 forem promissores, ela pode inclusive permanecer como benchmark oficial para comparar qualquer abordagem mais complexa.
