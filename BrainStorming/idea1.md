# Análise Revisada — Partindo de 2000 Features com 95% IG

Isto muda **bastante** o panorama. Com 2000 features já pré-selecionadas por IG, os problemas mais críticos que apontei desaparecem ou reduzem-se muito. Deixa-me rever tudo.

## O que muda com este facto

O problema da **maldição da dimensionalidade** — que era o argumento mais forte contra o Método 2 — fica largamente mitigado. 2000 features com IG alto são features *estruturadas e informativas*, não ruído. As métricas de distância voltam a ter significado. O KNN torna-se computacionalmente viável.

Além disso, o facto de 2000 features explicarem 95% da variância indica que **o espaço de features já é relativamente compacto** — o que sugere que dentro de clusters locais, chegar a ≤ 30 features com IG ≥ 90% é um objetivo realista, não wishful thinking.

---

## Reavaliação dos dois métodos

### Método 1: Árvore de Decisão — continua com o mesmo problema estrutural

A redução para 2000 features ajuda a árvore a ser mais limpa, mas o **problema fundamental mantém-se**: os 11 níveis superiores da árvore usam no máximo ~11 features globais para criar as 2048 partições. Isso não garante que dentro de cada partição as features relevantes sejam ≤ 30 — pode haver partições onde a separação malware/benign dentro do grupo ainda dependa de 200 features locais.

Outro problema que persiste: com milhões de instâncias e 2048 grupos, a **distribuição vai ser muito desigual** — alguns grupos terão centenas de milhares de instâncias, outros terão dezenas. O ILP não escala bem para grupos enormes.

### Método 2: KNN com IG adaptativo — agora genuinamente viável

Com 2000 features informativas, o KNN recupera a sua lógica. A distância entre instâncias volta a ser semanticamente significativa. O algoritmo de paragem baseado em IG por cluster **é agora a abordagem mais direta para garantir a constraint do ILP**.

O problema que resta é de **implementação e eficiência**, não de princípio:

```
Com milhões de instâncias e 2000 features:
- KNN exato → ainda caro (O(n²) para encontrar vizinhos)
- KNN aproximado (FAISS, Annoy, HNSW) → muito mais rápido,
  e suficientemente preciso para o objetivo de clustering
```

---

## Recomendação revisada: Método 2 com ajustes de implementação

O Método 2 é agora o caminho mais promissor, mas com algumas modificações importantes:

### Substituir KNN puro por Mini-Batch K-Means com verificação de IG

Em vez de crescer clusters a partir de K vizinhos (que é custoso), inverte a lógica:

```
1. Começa com K clusters razoável (ex: K=500)
   usando Mini-Batch K-Means (escala bem para milhões de instâncias)
        ↓
2. Para cada cluster, mede o IG das features dentro do cluster
        ↓
3. Clusters que precisam de mais de 30 features → subdividir (bisect)
   Clusters com instâncias a menos para ILP → fundir com vizinho mais próximo
        ↓
4. Repete até todos os clusters satisfazerem:
   - ≤ 30 features com IG ≥ 90%
   - ≥ N instâncias mínimas para o ILP (ex: 200-500)
```

### Por que não usar o Método 1 como inicializador?

Podes — e faz sentido usar a **árvore como warm start**. Mas não como método principal:

```
Árvore de decisão (profundidade limitada)
        ↓
  Grupos iniciais razoáveis (não os 2048, talvez 50-100)
        ↓
  Verificação de IG por grupo
        ↓
  Subdivisão onde necessário (K-Means local)
```

Isto aproveita o facto de a árvore já fazer seleção de features implicitamente nos primeiros níveis, e usa o K-Means só onde a árvore não consegue garantir a constraint do ILP.

---

## O pipeline completo revisado

```
85.000 features
      ↓  [já feito]
2.000 features (IG global ≥ 95%)
      ↓
Clustering iterativo (Mini-Batch K-Means + verificação de IG)
      ↓
Por cada cluster:
   • Top-30 features locais (IG local)
   • Dataset local: N instâncias × 30 features
   • Aplicar ILP  →  regras lógicas explicáveis
      ↓
Modelo final:
   Assignment de cluster + Regras ILP por cluster
```

---

## O que ainda precisas de validar empiricamente

Há uma questão que a teoria não resolve e que é central para o sucesso da tua abordagem:

> **Dentro de cada cluster, as 30 features com maior IG local são suficientes para separar malware de benign com boa accuracy?**

Isto podes testar *antes* de sequer aplicar ILP — treina um classificador simples (regressão logística, ou até um threshold) dentro de cada cluster usando apenas as top-30 features locais, e mede a accuracy. Se for alta → o ILP vai conseguir aprender regras úteis. Se for baixa → o cluster precisa de ser subdividido mais ou o threshold de 30 features precisa de ser revisto.

Este teste de sanidade é barato e pode poupar-te semanas de trabalho com o ILP.
