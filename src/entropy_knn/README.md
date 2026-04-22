# Entropy KNN Pipeline — Especificação

## 1. Objetivo

Esta pipeline existe para transformar o raciocínio experimental e matemático discutido em `Math_Entropy.md` numa abordagem operacional, reprodutível e comparável com as ideias já exploradas em `Idea1.md` e `Idea2.md`.

O objetivo principal é construir uma pipeline de seleção de features que:

1. comece com uma redução global por Information Gain (IG);
2. agrupe features semanticamente semelhantes ou redundantes;
3. avalie, dentro de cada grupo, quanta incerteza sobre a classe é removida por cada feature;
4. retenha apenas as features mais informativas por grupo;
5. permita um refinamento por interações de ordem superior quando necessário;
6. produza conjuntos reduzidos de features adequados para consumo por ILP.

Em termos práticos, esta pipeline pretende responder a uma pergunta central:

> Como reduzir um espaço de features grande e redundante sem perder as variáveis mais informativas para separar malware de benign e sem inviabilizar o uso de ILP?

---

## 2. Motivação

O projeto já tem duas linhas de pensamento complementares:

- **Idea2**: testar janelas fixas de features globais por IG, como baseline rápido e controlado.
- **Idea1**: explorar clustering local com verificação de qualidade dentro de grupos, para adaptar a seleção às relações locais entre features.

A nova pipeline `entropy_knn` nasce da combinação destas duas ideias com foco adicional na matemática da entropia:

- usa o ranking global de IG como ponto de partida;
- usa a lógica de clustering para organizar features em subconjuntos analisáveis;
- usa entropia condicional para decidir o que manter dentro de cada cluster;
- e prepara o dataset final para ILP com um controlo mais fino da dimensionalidade.

Esta pipeline não substitui automaticamente a Idea1 ou a Idea2. Em vez disso, serve como uma formulação mais geral e mais explícita do que queremos testar.

---

## 3. Base conceptual

A pipeline segue a intuição seguinte:

- features com IG alto são candidatas fortes a nível global;
- features próximas ou redundantes devem ser analisadas em conjunto;
- dentro de cada grupo, nem todas as features têm o mesmo valor explicativo;
- a métrica mais adequada para esse refinamento local é a **redução de entropia condicional**;
- a retenção final deve equilibrar dois objetivos:
  - reduzir dimensionalidade;
  - manter sinal discriminativo suficiente para ILP.

A lógica operacional é:

1. ordenar features por IG;
2. selecionar o top-$m'$;
3. agrupar por similaridade;
4. calcular $H(Y \mid X_j)$ dentro de cada cluster;
5. reter as melhores features locais;
6. opcionalmente analisar pares ou triplos;
7. exportar o subconjunto final.

---

## 4. Relação com os trabalhos anteriores

### 4.1 Relação com Idea2

A Idea2 introduz uma estrutura simples e robusta de exploração por janelas fixas. A sua utilidade principal é experimental:

- medir o impacto de subconjuntos pequenos e estáveis;
- perceber se o ranking global por IG já contém sinal suficiente nos primeiros blocos;
- comparar custo e qualidade de forma direta.

A `entropy_knn` reaproveita desta ideia:

- a preocupação com controlo de dimensionalidade;
- a necessidade de experimentos comparáveis;
- a lógica de varrer configurações com diferentes parâmetros.

### 4.2 Relação com Idea1

A Idea1 já mostrou que faz sentido olhar para grupos locais de features e medir qualidade da seleção dentro desses grupos. O report em `reports/idea1/knn_cluster_results.csv` é especialmente relevante porque mostra que uma análise local pode distinguir clusters com forte separação e clusters fracos.

A `entropy_knn` prolonga esse raciocínio:

- em vez de medir apenas MI médio ou performance de top-30, passa a medir também a redução de entropia condicional;
- em vez de usar apenas uma configuração fixa, passa a suportar sweep de tamanhos de cluster e thresholds;
- em vez de ser apenas uma análise exploratória, passa a produzir uma especificação de seleção reutilizável para ILP.

### 4.3 Relação com os reports existentes

Esta pipeline deve reutilizar a evidência já existente em:

- `reports/feature_analysis/incremental_ig_analysis_detailed.csv`
- `reports/feature_analysis/incremental_ig_analysis_detailed_top1000_step1.csv`
- `reports/idea1/knn_cluster_results.csv`
- `reports/feature_analysis/feature_rankings_all.parquet`

Esses artefactos já provam que existe uma ordenação global por IG e que há variação significativa entre zonas do ranking. A pipeline nova não deve recalcular a mesma informação sem necessidade; deve reaproveitá-la como input e evidência.

---

## 5. Entradas esperadas

A pipeline deve assumir, no mínimo, os seguintes inputs:

- matriz de features extraídas, por exemplo `reports/extracted_features.parquet`;
- labels em `data/training_set.csv`;
- ranking global de features em `reports/feature_analysis/feature_rankings_all.parquet`;
- opcionalmente, reports já calculados de incremental IG para orientar thresholds;
- opcionalmente, parâmetros de execução e seeds.

Se o ranking global já estiver calculado, a pipeline deve lê-lo em vez de o recomputar.

---

## 6. Saídas esperadas

A pipeline deve produzir:

1. **Tabelas de sweep** com resultados por combinação de parâmetros;
2. **Resumo agregado** por tamanho de cluster e threshold;
3. **Lista de features selecionadas por cluster**;
4. **Export final** de datasets reduzidos para ILP;
5. **Logs e artefactos de reprodutibilidade**.

Exemplos de outputs:

- `reports/entropy_knn/sweep_results.csv`
- `reports/entropy_knn/sweep_summary.csv`
- `reports/entropy_knn/selected_features_by_cluster.csv`
- `reports/entropy_knn/reduced_datasets/`
- `logs/entropy_knn/entropy_knn.log`

---

## 7. Pipeline funcional proposta

## 7.1 Etapa 1 — Carregamento e alinhamento de dados

- carregar features e labels;
- alinhar nomes de colunas entre matriz e ranking global;
- remover features em falta, duplicadas ou constantes, se necessário;
- verificar consistência da coluna alvo.

## 7.2 Etapa 2 — Filtragem global por IG

- ordenar features por IG global;
- selecionar os top-$m'$;
- guardar esta lista como base de todo o experimento;
- permitir variar $m'$ em sweep.

## 7.3 Etapa 3 — Agrupamento por similaridade

- agrupar features selecionadas em clusters;
- o clustering pode começar com KNN como lógica conceptual;
- se a implementação exigir melhor escalabilidade, pode ser substituída por uma variante aproximada, desde que preserve o objetivo conceptual.

## 7.4 Etapa 4 — Cálculo de entropia condicional local

Para cada cluster:

- calcular $H(Y \mid X_j)$ para cada feature $X_j$;
- medir a redução relativa face à entropia base $H(Y)$;
- ordenar features por capacidade de reduzir incerteza.

Uma forma útil de expressar o score é:

$$r_j = \frac{H(Y) - H(Y \mid X_j)}{H(Y)}$$

Este score pode ser usado como critério de retenção.

## 7.5 Etapa 5 — Seleção local

- manter as top-$k$ features por cluster;
- ou manter apenas features com $r_j \geq \tau$;
- ou aplicar ambos os critérios em conjunto.

A pipeline deve suportar sweep sobre:

- `cluster_size`
- `top_k`
- `threshold` ($\tau$)
- `top_features_global` ($m'$)

## 7.6 Etapa 6 — Refinamento por interações

Se o conjunto final ainda for grande:

- analisar pares de features;
- opcionalmente analisar triplos;
- aplicar refinamento apenas onde houver ganho claro.

Esta etapa deve ser opcional e controlada por parâmetros.

## 7.7 Etapa 7 — Export para ILP

- gerar datasets reduzidos;
- manter o mapeamento entre features originais e features selecionadas;
- exportar cada subconjunto pronto para execução no sistema ILP;
- garantir reprodutibilidade por seed e por configuração.

---

## 8. Sweep experimental

A pipeline deve suportar uma análise estilo sweep, semelhante à feita na Idea1, para comparar várias configurações.

### 8.1 Parâmetros a varrer

- tamanho do cluster;
- número de clusters ou anchors;
- top-$m'$ global;
- top-$k$ local;
- threshold de retenção $\tau$;
- seed aleatória.

### 8.2 Objetivo do sweep

O sweep serve para responder:

- que tamanho de cluster preserva melhor o sinal útil;
- qual threshold local é suficientemente conservador sem eliminar features relevantes;
- que combinação produz a melhor relação entre compactação e qualidade;
- quais configurações merecem ser passadas ao ILP.

### 8.3 Métricas a recolher

- número de features retidas por cluster;
- redução média de entropia;
- entropia residual média;
- proporção de features acima do threshold;
- estabilidade entre seeds;
- custo computacional;
- eventualmente, impacto downstream na ILP.

---

## 9. Critérios de sucesso

A pipeline será considerada útil se conseguir:

- reduzir significativamente a dimensionalidade;
- preservar features com forte relevância discriminativa;
- produzir subconjuntos estáveis entre seeds;
- mostrar separação clara entre configurações boas e fracas;
- gerar entradas viáveis para ILP sem explosão combinatória.

---

## 10. Princípios de implementação

Antes de escrever código, a implementação deve respeitar estes princípios:

- **Separação de responsabilidades**: leitura de dados, clustering, scoring, seleção e exportação devem ser módulos distintos;
- **Reutilização**: aproveitar loaders, rankings e estruturas já existentes;
- **Reprodutibilidade**: seeds, logs e outputs versionados;
- **Comparabilidade**: resultados sempre guardados em CSVs agregados;
- **Flexibilidade**: parâmetros ajustáveis sem alterar a lógica central;
- **Escalabilidade**: a arquitetura deve permitir futura adaptação a datasets maiores.

---

## 11. Resultado esperado desta documentação

Esta pasta e este ficheiro existem para garantir que qualquer pessoa que abra o projeto mais tarde consiga perceber rapidamente:

- qual é o objetivo científico da pipeline;
- como ela se relaciona com Idea1 e Idea2;
- que métricas e thresholds estamos a usar;
- porque é que a entropia condicional é central na seleção local;
- e como a pipeline se encaixa no uso final com ILP.

Em resumo: esta pipeline é a formalização operacional do raciocínio descrito em `Math_Entropy.md`, apoiada pelos experimentos já feitos e preparada para uma implementação comparável, repetível e extensível.
