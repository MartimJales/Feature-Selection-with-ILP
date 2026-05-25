# Balanced Entropy KNN Pipeline — 1:1 Malware/Goodware

## 1. Objetivo

Esta pasta contém a variante **balanced** da pipeline `entropy_knn`, desenhada para executar a mesma lógica de análise e seleção de features, mas com uma proporção **1:1 entre malware e goodware** antes da fase de clustering.

O objetivo é reduzir o viés introduzido pela forte desproporção original do dataset e produzir clusters mais equilibrados para análise posterior, sem alterar a pipeline original 1:9.

Em termos práticos, esta variante existe para responder à pergunta:

> O que acontece se mantivermos a mesma metodologia da pipeline original, mas forçarmos uma distribuição balanceada antes de fazer clustering, consensus analysis e PADTAI?

---

## 2. Motivação

A pipeline original `src/entropy_knn/` foi pensada para trabalhar com o dataset completo, onde a classe benign/goodware domina fortemente.

Essa distribuição é útil para certos estudos, mas tem dois problemas quando o objetivo é descobrir regras ou features com comportamento mais simétrico entre classes:

- o clustering pode ficar fortemente influenciado pela classe maioritária;
- clusters sem malware tornam-se pouco úteis para rule discovery com PADTAI.

A variante balanced resolve isso por **undersampling controlado da classe maioritária**, criando um input 1:1 antes do clustering. Assim:

- mantemos a metodologia conceptual da pipeline original;
- tornamos os clusters mais comparáveis entre si;
- reduzimos o tempo desperdiçado em clusters irrelevantes;
- facilitamos a análise de features e regras em clusters que realmente têm sinal de malware.

---

## 3. Relação com a pipeline original

Esta implementação **não substitui** `src/entropy_knn/`. Ela existe em paralelo.

Diferenças principais:

- `src/entropy_knn/` → pipeline original, sem balanceamento explícito;
- `src/entropy_knn_balanced/` → mesma ideia, mas com balanceamento 1:1 antes do clustering.

Isto é importante por duas razões:

1. permite comparação direta entre as duas abordagens;
2. evita introduzir alterações regressivas no pipeline original.

Se alguém no futuro quiser validar hipóteses experimentais, ambas as variantes podem ser executadas separadamente e comparadas com outputs distintos.

---

## 4. Estrutura da pasta

Conteúdo principal:

- `pipeline.py` — implementação da pipeline balanced, derivada de `EntropyKNNPipeline`;
- `runners/run_entropy_knn_balanced.py` — runner para executar a fase de clustering/score sweep da versão balanced;
- `runners/run_complete_pipeline_balanced.py` — runner end-to-end: clustering balanceado + consensus analysis + PADTAI;
- `runners/__init__.py` — pacote Python.

A pipeline usa também código partilhado de outras áreas do repositório:

- `src/entropy_knn/` para a base lógica;
- `src/analysis/entropy_knn_visualizations/run_cluster_top_feature_analysis.py` para consensus analysis;
- `src/ilp_pipeline/runners/run_ilp_per_cluster_test.py` para execução de PADTAI por cluster;
- `scripts/analyze_clusters_with_malware.py` para filtrar clusters que realmente contêm malware.

---

## 5. Fluxo completo da pipeline

A pipeline balanced segue esta sequência:

### 5.1 Fase 1 — Balanced clustering + feature selection

1. carregar features, labels e rankings globais;
2. aplicar balanceamento 1:1 por undersampling da classe maioritária;
3. executar o clustering sobre o conjunto balanceado;
4. guardar os artefactos de score sweep e as saídas por configuração.

O balanceamento é feito na classe `BalancedEntropyKNNPipeline`, que sobrescreve o carregamento da bundle original.

### 5.2 Fase 2 — Consensus analysis por cluster

Depois do clustering, a pipeline gera a análise por cluster:

- ranking de features por cluster;
- métricas como entropy reduction ratio, mutual information, chi-square, F-statistic e Pearson correlation;
- tabela agregada `top_feature_candidates.csv` por cluster.

Esta fase cria a base para a seleção das features que vão ser usadas no PADTAI.

### 5.3 Fase 3 — Filtro por malware

Antes de correr PADTAI, a pipeline analisa os clusters criados para identificar quais têm pelo menos um sample com `label = 1`.

A ideia é simples:

- se um cluster não tiver malware, não interessa gastar tempo com ILP/PADTAI nesse cluster;
- se tiver malware, continua para a fase de rule discovery.

Este passo é implementado com o helper `scripts/analyze_clusters_with_malware.py` e foi integrado no runner principal para evitar trabalho desnecessário.

### 5.4 Fase 4 — PADTAI por cluster

Para cada cluster elegível:

1. selecionam-se as top-N features do cluster;
2. prepara-se o input binário para PADTAI;
3. executa-se PADTAI com timeout e parâmetros definidos;
4. guardam-se as regras, metadata e artefactos intermédios.

---

## 6. Algoritmo de balanceamento

O balanceamento é intencionalmente simples e conservador.

A regra é:

- contar quantos samples de malware existem;
- contar quantos samples de goodware existem;
- escolher `min(malware_count, goodware_count)`;
- fazer undersampling da classe maioritária até essa quantidade;
- embaralhar o resultado;
- devolver uma bundle balanceada 1:1.

Isto significa que a pipeline não inventa dados nem faz oversampling. Ela apenas reduz a classe maioritária para igualar a minoritária.

### Consequência importante

O `cluster_size` do runner continua a ser o objetivo da configuração de clustering, mas o tamanho final de cada cluster pode variar após balanceamento e após filtragem por malware.

Ou seja:

- `cluster_size=500` não significa “250 malware + 250 goodware garantidos”;
- significa que a configuração de clustering está a usar essa escala como referência;
- o tamanho real depende da composição do dataset e da distribuição local das classes.

---

## 7. Entrypoints principais

### 7.1 Runner completo recomendado

```bash
bash run_complete_pipeline_balanced_overnight.sh
```

Este é o script recomendado quando se quer executar tudo de ponta a ponta:

- setup do ambiente;
- carregamento do `.env`;
- execução da pipeline balanced;
- notificações Discord;
- clustering;
- consensus analysis;
- filtragem de clusters com malware;
- PADTAI.

### 7.2 Runner Python principal

```bash
python3 src/entropy_knn_balanced/runners/run_complete_pipeline_balanced.py
```

Este runner faz a orquestração completa dentro do Python.

### 7.3 Runner de clustering/score sweep

```bash
python3 src/entropy_knn_balanced/runners/run_entropy_knn_balanced.py
```

Útil quando se quer apenas executar a parte de clustering/seleção global e não a parte completa com PADTAI.

---

## 8. Saídas esperadas

A pipeline escreve os artefactos em `reports/entropy_knn_balanced/`.

Estrutura típica esperada:

- `reports/entropy_knn_balanced/score_only/`
- `reports/entropy_knn_balanced/analysis/`
- `reports/entropy_knn_balanced/analysis/per_cluster_feature_vs_method/`
- `reports/entropy_knn_balanced/.../ilp_results/`

Outputs relevantes:

- tabelas de sweep;
- summaries por cluster;
- `top_feature_candidates.csv`;
- `ilp_metadata.json`;
- `padtai_rules.json` e artefactos associados;
- logs da execução.

---

## 9. Notificações Discord

A pipeline foi desenhada para enviar feedback contínuo por Discord durante a execução.

Isto inclui mensagens sobre:

- início da pipeline;
- fim da fase 1;
- início da fase de consensus;
- clusters processados;
- features selecionadas;
- resultados do PADTAI;
- summary final.

As credenciais são lidas a partir do `.env` ou passadas explicitamente por linha de comandos.

Variáveis esperadas:

- `DISCORD_WEBHOOK_URL`
- `DISCORD_USER_ID`

O objetivo é que se possa acompanhar a execução remotamente, sem abrir logs locais o tempo todo.

---

## 10. Script de análise de clusters com malware

O ficheiro `scripts/analyze_clusters_with_malware.py` foi criado para responder a uma necessidade prática: descobrir rapidamente quais os clusters que valem a pena para PADTAI.

Ele:

- percorre os clusters produzidos pela pipeline;
- lê o `padtai_input.csv` guardado em `ilp_results/`;
- conta quantos samples têm `label = 1`;
- gera uma lista de cluster IDs com malware.

Para a variante unbalanced e balanced, este script é útil porque evita correr ILP em clusters sem sinal útil.

---

## 11. O que futuros programadores devem ter em mente

### 11.1 Não misturar balanced com unbalanced

A regra mais importante é manter as duas pipelines separadas.

Se precisares de alterar a lógica da versão balanced, tenta não mexer na versão original a menos que a mudança seja realmente comum às duas.

### 11.2 Balanceamento é undersampling, não oversampling

A pipeline atual assume uma escolha metodológica simples e controlada:

- não cria exemplos sintéticos;
- não replica amostras para aumentar a classe minoritária;
- apenas reduz a maioritária.

### 11.3 Os clusters com malware são filtrados antes do PADTAI

Esse detalhe é intencional e importante para performance.

Se alterares esta lógica, a pipeline pode voltar a desperdiçar tempo em clusters que não produzem regras úteis.

### 11.4 Os artefactos podem mudar de nome/pasta

A estrutura de output foi pensada para ser previsível, mas pode evoluir.

Ao alterar a pipeline, confirma sempre:

- onde ficam os clusters;
- onde fica o consensus output;
- onde o runner do PADTAI espera encontrar os inputs;
- onde os logs são escritos.

### 11.5 Preservar reprodutibilidade

Sempre que possível:

- manter seeds explícitas;
- guardar logs;
- guardar metadata por run;
- evitar dependências implícitas em estado global.

---

## 12. Resumo curto

A pipeline balanced existe para executar a mesma ideia da `entropy_knn`, mas com um dataset balanceado 1:1 antes do clustering.

Ela serve para:

- reduzir viés de classe;
- melhorar a utilidade dos clusters;
- filtrar clusters sem malware;
- acelerar o PADTAI;
- manter uma alternativa comparável à pipeline original.

Se estiveres a continuar este trabalho, esta é a versão a usar quando o foco for análise mais equilibrada e discovery de regras em clusters com sinal real de malware.
