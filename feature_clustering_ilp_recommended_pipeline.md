# Feature Clustering + ILP (Pipeline Recomendado)

## Objetivo
Definir uma pipeline **mais robusta que K-Means global** para reduzir dimensionalidade e melhorar interpretabilidade antes de ILP, mantendo foco em:
- escalabilidade
- qualidade das regras
- clareza metodológica

---

## 1) Princípios orientadores (ILP-first)

1. **Evitar redução opaca** (ex.: PCA/autoencoder) na fase principal, para não perder interpretabilidade.
2. **Trabalhar por grupos semânticos de features** (prefixos/tipos DREBIN), não tudo misturado.
3. **Avaliar clustering pelo impacto no ILP**, não apenas por métricas geométricas.
4. **Reduzir redundância mantendo cobertura** dos padrões úteis.

---

## 2) Porque o K-Means global falhou (esperado)

Num espaço binário/esparso e de alta dimensão, K-Means tende a falhar por:
- pressupor clusters aproximadamente esféricos
- depender de distância euclidiana
- baixa robustez a sparsidade extrema
- pouca correspondência com estrutura semântica das features

Conclusão: faz sentido abandonar K-Means como abordagem principal.

---

## 3) Estratégia recomendada (em camadas)

## 3.1 Camada A — Agrupamento semântico inicial
Separar top-K features por tipo/origem:
- `perm_*` (requested permissions)
- `used_permissions`
- `api_calls`
- `suspicious_calls`
- `intent_filters`
- contagens (`n_*`)
- restantes

Objetivo: criar subproblemas menores e semanticamente coerentes.

## 3.2 Camada B — Similaridade feature-feature por grupo
Dentro de cada grupo, usar similaridades adequadas a dados binários:
- **Jaccard** (presença/ausência)
- **Cosine** (sparse vectors)
- **Mutual Information normalizada** (dependências não lineares)

Nota: Pearson/Spearman ficam como comparação secundária.

## 3.3 Camada C — Clustering de features por grupo
Testar, por ordem de prioridade:
1. **Hierarchical clustering** (com corte por distância)
2. **Graph-based (Louvain/Leiden)** sobre grafo de similaridade
3. HDBSCAN (opcional, fase posterior)

---

## 4) Redução dentro de cada cluster

Para cada cluster de features, construir representação compacta:

1. **Representante principal**: feature com maior MI com `y`
2. **Backup opcional**: medoid do cluster
3. **Predicado composto simples** (quando fizer sentido):
   - co-ocorrência forte (`A ∧ B`)
   - presença mínima em subconjuntos relevantes

Resultado: novo conjunto reduzido de variáveis interpretáveis.

---

## 5) Integração com ILP

## 5.1 ILP local (por grupo/cluster)
Executar ILP em blocos menores para reduzir explosão combinatória.

## 5.2 Agregação de regras
Combinar regras locais num nível global:
- deduplicação
- simplificação
- validação cruzada de cobertura

## 5.3 ILP global final
Executar ILP final sobre:
- representantes de cluster
- predicados compostos estáveis
- features de contagem relevantes

---

## 6) Plano de experiências (mínimo viável)

## Fase 1 — Baselines curtos
1. Baseline atual: top-1000 direto para ILP
2. Top-1000 → agrupamento semântico apenas (sem clustering interno)

## Fase 2 — Clustering por grupo
3. Semântico + Hierarchical
4. Semântico + Graph (Louvain)

## Fase 3 — Consolidação
5. Escolher melhor pipeline para ILP final

---

## 7) Métricas de decisão (prioridade)

## 7.1 Primárias (ILP)
- tempo de execução
- nº de regras
- tamanho médio das regras
- cobertura
- precisão/F1

## 7.2 Secundárias (clustering)
- estabilidade entre runs
- redução de redundância
- coesão/separação (apoio, não critério principal)

---

## 8) Critério de sucesso

Uma pipeline é melhor se conseguir:
1. reduzir significativamente dimensionalidade
2. manter ou melhorar qualidade preditiva
3. gerar regras mais curtas e interpretáveis
4. diminuir custo computacional do ILP

---

## 9) Riscos e mitigação

- **Risco:** perder sinais úteis ao comprimir clusters
  **Mitigação:** manter 1 representante + 1 backup em clusters grandes

- **Risco:** clusters instáveis por escolha de métrica
  **Mitigação:** repetir com 2-3 métricas e comparar estabilidade

- **Risco:** predicados compostos excessivos
  **Mitigação:** limitar a padrões simples e com suporte mínimo

---

## 10) Recomendação prática (arranque)

Começar por:
1. top-1000
2. separação por tipo de feature
3. hierarchical por grupo com Jaccard/Cosine
4. 1 representante por cluster (maior MI)
5. ILP por grupo + agregação

Só depois comparar com Louvain/Leiden.

---

## Resumo executivo

Para o teu contexto, a estratégia mais robusta é **feature clustering semântico + ILP hierárquico**, em vez de clustering global com K-Means.
Isto maximiza interpretabilidade, reduz complexidade e melhora a viabilidade prática do ILP.
