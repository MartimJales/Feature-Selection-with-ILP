# Feature Selection + Clustering for ILP

## Contexto
- Dataset binário (label: positivo/negativo)
- ~75k linhas, ~85k features
- Redução inicial: IG / Mutual Information → top 1000 features
- Objetivo: reduzir ainda mais dimensionalidade e estruturar features para ILP

---

# 1. Definição do Problema

Queremos agrupar **features (colunas)** com comportamento semelhante, de forma a:
- reduzir redundância
- melhorar interpretabilidade
- criar subproblemas mais pequenos para ILP

---

# 2. Métricas de Similaridade entre Features

## 2.1 Correlação
- Pearson (linear)
- Spearman (monótono)

## 2.2 Mutual Information (feature-feature)
- MI(fi, fj)
- Capta relações não lineares

## 2.3 Similaridade baseada na label
- |MI(fi, y) − MI(fj, y)|
- Agrupa features com impacto semelhante

## 2.4 Cosine Similarity
- Especialmente útil para dados esparsos

## 2.5 Distância baseada em distribuição
- Comparar distribuições condicionais P(fi | y)

---

# 3. Técnicas de Clustering

## 3.1 Hierarchical Clustering
- Baseado em matriz de distância
- Não requer k fixo
- Permite corte posterior (dendrograma)

## 3.2 Spectral Clustering
- Baseado em matriz de afinidade
- Captura estruturas não lineares

## 3.3 DBSCAN / HDBSCAN
- Detecta clusters densos
- Remove ruído automaticamente

## 3.4 K-Means
- Baseline simples
- Assume clusters esféricos

---

# 4. Alternativa: Feature Graph

## Construção do grafo
- Nós: features
- Arestas: similaridade (MI, correlação)

## Algoritmos
- Louvain
- Leiden

## Vantagens
- Captura melhor relações complexas
- Não exige número de clusters

---

# 5. Estratégias após Clustering

## 5.1 Representante por Cluster
- Feature com maior MI com label
- Medoid do cluster

## 5.2 ILP por Cluster
- Executar ILP em cada cluster separadamente

## 5.3 Ensemble de ILP
- Combinar regras de múltiplos clusters

## 5.4 Construção de Features Derivadas
- Co-ocorrência
- Implicações
- Features compostas

---

# 6. Pipeline Sugerido

1. IG / MI → top 1000 features
2. Calcular similaridade feature-feature
3. Construir matriz de distância
4. Aplicar clustering (hierarchical ou graph-based)
5. Selecionar features por cluster
6. Executar ILP por cluster
7. Agregar regras

---

# 7. Pré-processamento para ILP

## 7.1 Discretização
- Binning
- Thresholding

## 7.2 Criação de Predicados
- high_feature_X
- feature_X_present

---

# 8. Ideias Avançadas

## 8.1 Meta-ILP
- ILP hierárquico
- Combinação de regras

## 8.2 Feature Structuring
- Organização semântica de features

## 8.3 Interpretable Pipelines
- Combinação de clustering + ILP + XAI

---

# 9. Experiências a Testar

- Comparar MI vs correlação
- Hierarchical vs Louvain
- Número de clusters
- Impacto na performance do ILP
- Interpretabilidade das regras

---

# 10. Possíveis Extensões

- Dimensionality reduction híbrida (PCA + clustering)
- Autoencoders (como baseline)
- Comparação com modelos não simbólicos

---

# Notas

Este documento serve como base iterativa para experimentação e pode ser refinado conforme resultados.

