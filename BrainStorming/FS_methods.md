# Relatório técnico sobre Feature Selection (métodos filter) para deteção de malware

## 1. Objetivo

Este relatório foca **apenas** métodos de Feature Selection do tipo **filter** para um dataset de malware com dimensionalidade muito elevada (aprox. 85 mil features).

O objetivo principal é reduzir o espaço de features para um conjunto muito menor, mantendo o máximo possível da capacidade preditiva dos modelos de classificação.

A preocupação principal é:

1. reduzir drasticamente a dimensionalidade;
2. manter boa performance de classificação;
3. evitar features redundantes;
4. preservar alguma interpretabilidade;
5. evitar métodos demasiado caros computacionalmente;
6. produzir um conjunto final de features suficientemente pequeno para análise posterior.

A parte de ILP não é o foco deste relatório. A ILP pode ser usada posteriormente sobre o conjunto reduzido de features.

---

## 2. Contexto do problema

Em datasets de malware, é comum ter features binárias e esparsas, por exemplo:

1. permissões Android;
2. chamadas de API;
3. strings;
4. intents;
5. componentes;
6. padrões estáticos extraídos de ficheiros;
7. características de comportamento;
8. indicadores de presença ou ausência.

Este tipo de dados tem vários desafios:

1. muitas features são raras;
2. muitas features são redundantes;
3. algumas features só são úteis em conjunto;
4. algumas features podem estar associadas apenas a famílias específicas de malware;
5. o dataset pode sofrer de drift temporal;
6. algumas features podem parecer muito fortes no treino mas generalizar mal.

Por isso, uma estratégia de Feature Selection robusta deve combinar vários critérios, e não depender apenas de uma métrica univariada.

---

## 3. Métodos filter

Este relatório fica restrito aos métodos **filter**, que seleccionam features antes do treino do modelo final, usando critérios estatísticos entre feature e classe (ou entre features).

Exemplos:

1. Information Gain;
2. Mutual Information;
3. Chi Square;
4. mRMR;
5. JMI;
6. CMIM;
7. FCBF;
8. correlação;
9. Symmetrical Uncertainty.

Vantagens:

1. rápidos;
2. escaláveis;
3. independentes do classificador;
4. bons para filtragem inicial;
5. úteis em datasets com muitas features.

Desvantagens:

1. podem ignorar interacções complexas;
2. alguns métodos são essencialmente univariados;
3. a performance final nem sempre corresponde ao ranking estatístico.

---

## 4. Information Gain e Mutual Information

## 4.1 Ideia principal

Information Gain mede a redução de incerteza sobre a classe quando conhecemos uma feature. Em termos de informação mútua, para uma feature $X_j$ e classe $Y$:

$$
IG(X_j) = H(Y) - H(Y \mid X_j)
$$

Para variáveis discretas, este valor coincide com a informação mútua entre a feature e a classe:

$$
I(X_j;Y) = H(Y) - H(Y \mid X_j) = H(X_j) - H(X_j \mid Y)
$$

Se quiseres um score normalizado para comparar features com classes de entropia diferente, podes usar:

$$
IG_{norm}(X_j) = \frac{I(X_j;Y)}{H(Y)} = 1 - \frac{H(Y \mid X_j)}{H(Y)}
$$

Assim, o texto fica alinhado com a fórmula: Information Gain é o termo não normalizado, e Mutual Information é a mesma quantidade vista como dependência entre $X_j$ e $Y$.

## 4.2 Vantagens

1. é simples;
2. é rápida;
3. funciona bem com features binárias;
4. é interpretável;
5. é adequada para screening inicial.

## 4.3 Limitações

1. é univariada;
2. não detecta bem interacções;
3. não penaliza redundância;
4. pode seleccionar muitas features parecidas;
5. pode sobrevalorizar features raras;
6. pode eliminar features fracas individualmente mas fortes em conjunto.

## 4.4 Recomendação

Usar Mutual Information apenas como primeira etapa.

Exemplo:

1. dataset original com 85 mil features;
2. remover features raras;
3. calcular MI entre cada feature e a classe;
4. manter top 1000, top 3000 ou top 5000;
5. aplicar métodos mais fortes depois.

---

## 5. mRMR

## 5.1 Ideia principal

mRMR significa minimum Redundancy Maximum Relevance.

O objectivo é seleccionar features que sejam:

1. relevantes para a classe;
2. pouco redundantes entre si.

Uma forma comum de pontuação é:

$$
score(X_j) = I(X_j;Y) - \frac{1}{|S|}\sum_{X_s \in S} I(X_j;X_s)
$$

onde:

1. $S$ é o conjunto de features já seleccionadas;
2. $I(X_j;Y)$ é a **informação mútua** entre a feature e a classe (relevância);
3. $I(X_j;X_s)$ é a **informação mútua** entre duas features (redundância).

## 5.2 Vantagens

1. melhor do que MI simples;
2. reduz redundância;
3. é interpretável;
4. funciona bem com features binárias;
5. é adequado para datasets de malware.

## 5.3 Limitações

1. continua a ser essencialmente greedy;
2. não captura interacções complexas de ordem alta;
3. pode ser caro se calculado para muitas features;
4. depende da qualidade da estimativa de MI.

## 5.4 Quando usar

mRMR é uma excelente segunda etapa depois de MI.

Exemplo:

1. reduzir de 85 mil para 5000 features com MI;
2. aplicar mRMR;
3. reduzir de 5000 para 500 ou 1000;
4. passar para uma etapa embedded, como Elastic Net.

---

## 6. JMI

## 6.1 Ideia principal

JMI significa Joint Mutual Information.

A ideia é seleccionar features que acrescentem informação em conjunto com as features já seleccionadas.

Em vez de olhar apenas para:

$$
I(X_j;Y)
$$

JMI tenta aproximar a utilidade conjunta de:

$$
I(X_j, X_s;Y)
$$

para features \(X_s\) já seleccionadas.

## 6.2 Vantagens

1. captura melhor complementaridade do que mRMR;
2. é mais sensível a interacções entre pares;
3. continua mais barato do que testar todos os pares;
4. é útil quando features individualmente moderadas se tornam fortes em conjunto.

## 6.3 Limitações

1. é mais caro do que mRMR;
2. pode ser difícil de estimar bem em dados muito esparsos;
3. ainda não resolve interacções de ordem alta.

## 6.4 Quando usar

JMI é uma boa opção quando se suspeita que há interacções relevantes entre features.

Recomendação prática:

1. aplicar JMI apenas depois de reduzir para 1000 a 5000 features;
2. comparar com mRMR e CMIM;
3. avaliar com o mesmo classificador final.

---

## 7. CMIM

## 7.1 Ideia principal

CMIM significa Conditional Mutual Information Maximization.

A ideia é escolher features que continuam informativas mesmo depois de considerar features já seleccionadas.

Uma formulação típica é:

$$
score(X_j) = \min_{X_s \in S} I(X_j;Y \mid X_s)
$$

Isto significa que uma feature é boa se a sua informação sobre a classe não desaparecer quando condicionamos noutras features seleccionadas.

## 7.2 Vantagens

1. muito adequado para features binárias;
2. reduz redundância;
3. captura alguma complementaridade;
4. é mais forte do que MI univariada;
5. é uma boa escolha para malware.

## 7.3 Limitações

1. mais caro do que MI simples;
2. não substitui validação com classificador;
3. pode depender da qualidade das estimativas de probabilidade.

## 7.4 Recomendação

CMIM é provavelmente um dos melhores métodos filter para o teu caso.

Pipeline possível:

1. remover features raras;
2. top 5000 por MI;
3. CMIM para reduzir para 500;
4. Elastic Net para reduzir para 50 a 200;
5. validação final com classificadores.

---

## 8. ReliefF e variantes

## 8.1 Ideia principal

ReliefF atribui peso a uma feature com base na sua capacidade de distinguir amostras próximas de classes diferentes.

A ideia é:

1. escolher uma amostra;
2. encontrar vizinhos próximos da mesma classe;
3. encontrar vizinhos próximos da classe oposta;
4. aumentar o peso das features que diferenciam classes;
5. diminuir o peso das features que variam dentro da mesma classe.

## 8.2 Vantagens

1. consegue detectar interacções;
2. não é puramente univariado;
3. pode funcionar bem quando features combinadas são importantes;
4. é útil como método complementar.

## 8.3 Limitações

1. pode ser pesado em datasets grandes;
2. depende de cálculo de vizinhos;
3. em dados muito esparsos, a escolha da métrica é crítica;
4. não deve ser usado directamente nas 85 mil features.

## 8.4 Quando usar

Usar apenas depois de uma filtragem inicial.

Exemplo:

1. MI reduz 85 mil para 3000;
2. ReliefF reduz 3000 para 500;
3. Elastic Net ou classificador final faz a selecção final.

---

## 9. L1 Regularization

## 9.1 Ideia principal

Modelos com penalização L1 forçam muitos coeficientes a zero.

Exemplo com Logistic Regression:

$$
\min_{\beta} Loss(Y,X\beta) + \lambda \|\beta\|_1
$$

Se:

$$
\beta_j = 0
$$

então a feature \(X_j\) é removida.

## 9.2 Vantagens

1. muito útil para alta dimensionalidade;
2. funciona bem com matrizes esparsas;
3. é eficiente;
4. produz modelos interpretáveis;
5. liga a selecção de features à performance do modelo.

## 9.3 Limitações

1. pode ser instável quando há features correlacionadas;
2. pode escolher uma feature arbitrária dentro de um grupo redundante;
3. depende da escolha de \(\lambda\);
4. pode eliminar features úteis se a regularização for demasiado forte.

## 9.4 Quando usar

## 10. Graph based redundancy pruning

## 13.1 Ideia principal

Depois de teres um conjunto candidato de features, constróis um grafo onde:

1. cada nó é uma feature;
2. existe uma aresta entre duas features se forem muito semelhantes;
3. dentro de cada grupo de features semelhantes, escolhes uma representante.

## 13.2 Métricas possíveis

Para features binárias e esparsas, boas métricas são:

1. Jaccard similarity;
2. cosine similarity;
3. Symmetrical Uncertainty;
4. Mutual Information entre features.

Jaccard é particularmente intuitiva para features binárias:

$$
J(X_a,X_b) = \frac{|X_a \cap X_b|}{|X_a \cup X_b|}
$$

## 13.3 Como usar

Criar uma aresta se:

$$
J(X_a,X_b) \geq \theta
$$

Valores típicos:

$$
\theta = 0.7
$$

$$
\theta = 0.8
$$

$$
\theta = 0.9
$$

Depois, para cada componente do grafo, escolher a feature com maior score supervisionado.

O score pode combinar:

1. MI com a classe;
2. frequência de estabilidade;
3. coeficiente absoluto no Elastic Net;
4. importância em árvores.

## 13.4 Vantagens

1. reduz redundância;
2. melhora interpretabilidade;
3. evita seleccionar várias features equivalentes;
4. é mais controlado do que clustering genérico.

## 13.5 Recomendação

Usar depois de CMIM e Elastic Net, não antes.

---

## 14. Clustering de features

## 14.1 Pode ser útil?

Sim, mas com cuidado.

Clustering pode ajudar a agrupar features parecidas. No entanto, se for feito antes da selecção supervisionada, pode agrupar features que são semelhantes mas têm valor preditivo diferente.

## 14.2 Problemas

1. clustering é muitas vezes não supervisionado;
2. pode perder features úteis;
3. depende muito da métrica;
4. pode ser caro;
5. pode ser difícil escolher o número de clusters.

## 14.3 Métricas recomendadas

Para features binárias e esparsas:

1. Jaccard;
2. cosine similarity;
3. Mutual Information entre features;
4. Symmetrical Uncertainty.

Evitar distância Euclidiana simples.

## 14.4 Recomendação

Eu não colocaria clustering como etapa central.

Preferia:

1. ranking supervisionado;
2. redundancy aware selection;
3. grafo de redundância;
4. selecção de representantes.

Ou seja, substituir clustering por graph based redundancy pruning.

---


## 15. Pipeline recomendada

A pipeline recomendada é:

```text
Input:
    X com aproximadamente 85000 features
    Y com labels malware ou benigno

Passo 1:
    Separar treino, validação e teste

Passo 2:
    Remover features raras e quase constantes

Passo 3:
    Calcular Mutual Information entre cada feature e Y

Passo 4:
    Manter top m features
    Testar m igual a 1000, 3000 e 5000

Passo 5:
    Aplicar CMIM, mRMR ou JMI

Passo 6:
    Reduzir para q features
    Testar q igual a 300, 500 e 1000

Passo 7:
    Aplicar Elastic Net com Stability Selection

Passo 8:
    Manter features com frequência de selecção superior a 0.6 ou 0.7

Passo 9:
    Aplicar graph based redundancy pruning

Passo 10:
    Produzir conjuntos finais com 25, 50, 100 e 200 features

Passo 11:
    Avaliar classificadores finais

Output:
    conjunto final compacto, robusto e interpretável de features
