# Entropy-Based Feature Selection Pipeline for Malware Detection

## 1. Objective

The goal of this pipeline is to perform feature selection in
high-dimensional datasets by combining:

- Univariate information-theoretic metrics
- Redundancy-aware selection methods
- Limited exploration of feature interactions
- Preparation of a reduced feature set for Inductive Logic Programming

---

## 2. Step 1: Univariate Feature Evaluation

For each feature $X_i$, compute its relevance with respect to the class
variable $Y$.

### 2.1 Entropy of the Class

$$
H(Y) = - \sum_{y \in \mathcal{Y}} P(y)\log_2 P(y)
$$

Example 1 (balanced classes, 50% / 50%):

$$
H(Y) = -\big(0.5\log_2(0.5) + 0.5\log_2(0.5)\big) = 1
$$

This shows why we use $\log_2$: entropy is measured in bits, and a perfectly balanced
binary class has exactly 1 bit of uncertainty.

Example 2 (rare malware, 5% / 95%):

$$
H(Y) = -\big(0.05\log_2(0.05) + 0.95\log_2(0.95)\big) \approx 0.286
$$

When malware represents only about $5\%$ of the dataset, the class entropy is much
lower, indicating a more predictable (more imbalanced) target distribution.

### 2.2 Conditional Entropy

$$
H(Y \mid X_i) = \sum_{x \in \mathcal{X}_i} P(x)\, H(Y \mid X_i = x)
$$

with:

$$
H(Y \mid X_i = x) = - \sum_{y \in \mathcal{Y}} P(y \mid x)\log_2 P(y \mid x)
$$

### 2.2.1 Dummy Dataset (Manual Conditional Entropy)

To make the conditional entropy concrete, consider a small binary dataset with
2 features and a binary class ($malware$ / $benign$):

| $X_1$ | $X_2$ | $Y$ |
|---:|---:|---|
| 1 | 1 | malware |
| 1 | 1 | malware |
| 1 | 0 | malware |
| 1 | 0 | benign |
| 0 | 1 | benign |
| 0 | 0 | benign |
| 0 | 1 | benign |
| 0 | 0 | malware |

Class counts: 4 malware, 4 benign.

For $X_1$:

- $X_1 = 1$: 4 samples $\rightarrow$ 3 malware, 1 benign
  $$
  H(Y \mid X_1=1) = -\left(\frac{3}{4}\log_2\frac{3}{4} + \frac{1}{4}\log_2\frac{1}{4}\right) \approx 0.811
  $$
- $X_1 = 0$: 4 samples $\rightarrow$ 1 malware, 3 benign
  $$
  H(Y \mid X_1=0) = -\left(\frac{1}{4}\log_2\frac{1}{4} + \frac{3}{4}\log_2\frac{3}{4}\right) \approx 0.811
  $$

Therefore:

$$
H(Y \mid X_1) = P(X_1=1)H(Y \mid X_1=1) + P(X_1=0)H(Y \mid X_1=0)
= 0.5\cdot0.811 + 0.5\cdot0.811 = 0.811
$$

For $X_2$:

- $X_2 = 1$: 4 samples $\rightarrow$ 2 malware, 2 benign
  $$
  H(Y \mid X_2=1) = -\left(\frac{1}{2}\log_2\frac{1}{2} + \frac{1}{2}\log_2\frac{1}{2}\right) = 1
  $$
- $X_2 = 0$: 4 samples $\rightarrow$ 2 malware, 2 benign
  $$
  H(Y \mid X_2=0) = 1
  $$

Therefore:

$$
H(Y \mid X_2) = 0.5\cdot1 + 0.5\cdot1 = 1
$$

In this toy example, $X_1$ has lower conditional entropy than $X_2$, so $X_1$
is more informative about the class.

### 2.2.2 Conditional Entropy with Both Features: $H(Y \,|\, X_1, X_2)$

Now we condition on the pair $(X_1, X_2)$ and compute:

$$
H(Y \mid X_1, X_2) = \sum_{x_1,x_2} P(x_1,x_2)\,H(Y\mid x_1,x_2)
$$

From the same dummy dataset:

- $(X_1,X_2)=(1,1)$: 2 samples $\rightarrow$ 2 malware, 0 benign
  $$
  H(Y\mid 1,1)=0
  $$
- $(X_1,X_2)=(1,0)$: 2 samples $\rightarrow$ 1 malware, 1 benign
  $$
  H(Y\mid 1,0)=1
  $$
- $(X_1,X_2)=(0,1)$: 2 samples $\rightarrow$ 0 malware, 2 benign
  $$
  H(Y\mid 0,1)=0
  $$
- $(X_1,X_2)=(0,0)$: 2 samples $\rightarrow$ 1 malware, 1 benign
  $$
  H(Y\mid 0,0)=1
  $$

Each pair appears 2 times in 8 samples, so:

$$
P(1,1)=P(1,0)=P(0,1)=P(0,0)=\tfrac{1}{4}
$$

Therefore:

$$
\begin{aligned}
H(Y \mid X_1, X_2)
&= \tfrac{1}{4}H(Y\mid 1,1) + \tfrac{1}{4}H(Y\mid 1,0) + \tfrac{1}{4}H(Y\mid 0,1) + \tfrac{1}{4}H(Y\mid 0,0) \\
&= \tfrac{1}{4}\cdot0 + \tfrac{1}{4}\cdot1 + \tfrac{1}{4}\cdot0 + \tfrac{1}{4}\cdot1 = 0.5
\end{aligned}
$$

So, using both features together reduces class uncertainty to $0.5$ bits.

---

### 2.3 Information Gain (IG)

$$
IG(X_i) = H(Y) - H(Y \mid X_i)
$$

Using the dummy dataset:

$$
H(Y)=1, \quad H(Y \mid X_1)=0.811, \quad H(Y \mid X_2)=1
$$

So:

$$
IG(X_1)=H(Y)-H(Y \mid X_1)=1-0.811=0.189
$$

$$
IG(X_2)=H(Y)-H(Y \mid X_2)=1-1=0
$$

Interpretation: $X_1$ reduces uncertainty by about $0.189$ bits, while $X_2$ does
not reduce uncertainty.

This measures the reduction in uncertainty about $Y$ when observing
$X_i$.

---

### 2.4 Mutual Information (MI)

$$
MI(X_i; Y) = \sum_{x,y} P(x,y)\log \frac{P(x,y)}{P(x)P(y)}
$$

Manual calculation for $X_1$:

From the table:

$$
P(X_1=1)=P(X_1=0)=\tfrac{1}{2}, \quad P(Y=malware)=P(Y=benign)=\tfrac{1}{2}
$$

$$
P(1,mal)=\tfrac{3}{8},\; P(1,ben)=\tfrac{1}{8},\; P(0,mal)=\tfrac{1}{8},\; P(0,ben)=\tfrac{3}{8}
$$

So:

$$
\begin{aligned}
MI(X_1;Y) =
&\; \tfrac{3}{8}\log_2\!\left(\frac{\tfrac{3}{8}}{\tfrac{1}{2}\cdot\tfrac{1}{2}}\right)
+\tfrac{1}{8}\log_2\!\left(\frac{\tfrac{1}{8}}{\tfrac{1}{2}\cdot\tfrac{1}{2}}\right)
+\tfrac{1}{8}\log_2\!\left(\frac{\tfrac{1}{8}}{\tfrac{1}{2}\cdot\tfrac{1}{2}}\right)
+\tfrac{3}{8}\log_2\!\left(\frac{\tfrac{3}{8}}{\tfrac{1}{2}\cdot\tfrac{1}{2}}\right) \approx 0.189
\end{aligned}
$$

Manual calculation for $X_2$:

From the table:

$$
P(X_2=1)=P(X_2=0)=\tfrac{1}{2}, \quad P(Y=malware)=P(Y=benign)=\tfrac{1}{2}
$$

All joint probabilities are equal:

$$
P(1,mal)=P(1,ben)=P(0,mal)=P(0,ben)=\tfrac{1}{4}
$$

So:

$$
\begin{aligned}
MI(X_2;Y) =
&\; \tfrac{1}{4}\log_2\!\left(\frac{\tfrac{1}{4}}{\tfrac{1}{2}\cdot\tfrac{1}{2}}\right)
+ \tfrac{1}{4}\log_2\!\left(\frac{\tfrac{1}{4}}{\tfrac{1}{2}\cdot\tfrac{1}{2}}\right)
+ \tfrac{1}{4}\log_2\!\left(\frac{\tfrac{1}{4}}{\tfrac{1}{2}\cdot\tfrac{1}{2}}\right)
+ \tfrac{1}{4}\log_2\!\left(\frac{\tfrac{1}{4}}{\tfrac{1}{2}\cdot\tfrac{1}{2}}\right) = 0
\end{aligned}
$$

Therefore:

$$
MI(X_1;Y) \approx 0.189, \quad MI(X_2;Y)=0
$$

Since the features and the class are **discrete** in this example, the two measures are
equivalent in our setting: $IG(X_i) = MI(X_i;Y)$.

Recommended reading: [Information Gain and Mutual Information for Machine Learning](https://www.geeksforgeeks.org/machine-learning/information-gain-and-mutual-information-for-machine-learning/)

---

## 3. Step 2: Selection of Top $m$ Features

Rank all features according to:

$$
Score(X_i) = I(X_i; Y)
$$

Select:

$$
S = \{X_1, X_2, \dots, X_m\}
$$

where $m \ll F$, with $F$ being the total number of features.

---

## 4. Step 3: Redundancy-Aware Feature Selection

After selecting the top $m$ features, refine the subset by accounting for redundancy.

Two methods are considered:

---

## 4.1 mRMR (Minimum Redundancy Maximum Relevance)

### Objective

Select features that are:

- Highly relevant to $Y$
- Minimally redundant with respect to each other

---

### Scoring Function

Given a selected set $S$, the score for a candidate feature $X_i \notin S$ is:

$$
Score_{mRMR}(X_i) = I(X_i; Y) - \frac{1}{|S|} \sum_{X_j \in S} I(X_i; X_j)
$$

---

### Interpretation

- First term: relevance to the target
- Second term: redundancy penalty

---

### Selection Strategy

Greedy:

1. Initialize:
  $$
  S = \{ \arg\max_X I(X;Y) \}
  $$

2. Iteratively add:

$$
X^* = \arg\max_{X_i \notin S} Score_{mRMR}(X_i)
$$

---

## 4.2 JMI (Joint Mutual Information)

### Objective

Capture interactions between features with respect to the class.

---

### Scoring Function

$$
Score_{JMI}(X_i) = \sum_{X_j \in S} I(X_i, X_j; Y)
$$

---

### Joint Mutual Information Definition

$$
I(X_i, X_j; Y) = H(Y) - H(Y \mid X_i, X_j)
$$

---

### Interpretation

- Measures how much $X_i$ contributes jointly with already selected
  features
- Captures pairwise interactions

---

### Selection Strategy

Greedy, similar to mRMR:

$$
X^* = \arg\max_{X_i \notin S} \sum_{X_j \in S} I(X_i, X_j; Y)
$$

---

## 5. Step 4: Optional Exploration of Feature Interactions

### Question: Before or after mRMR/JMI?

👉 **Answer: After**

---

### Rationale

- Before: computationally infeasible due to combinatorial explosion
- After: restricted to a manageable subset

---

### 5.1 Pairwise Conditional Entropy

$$
H(Y \mid X_i, X_j) = \sum_{x_i,x_j} P(x_i,x_j)\, H(Y \mid x_i,x_j)
$$

---

### 5.2 Information Gain for Pairs

$$
IG(X_i, X_j) = H(Y) - H(Y \mid X_i, X_j)
$$

---

### 5.3 Triplets

$$
H(Y \mid X_i, X_j, X_k)
$$

$$
IG(X_i, X_j, X_k) = H(Y) - H(Y \mid X_i, X_j, X_k)
$$

---

### Practical Strategy

- Compute only within reduced set $S$
- Optionally restrict to top-k features after mRMR/JMI
- Prioritize:
  - highest pairwise gains
  - strongest interaction effects

---

## 6. Step 5: Output Feature Set

Final selected set:

$$
S^* \subseteq S
$$

constructed using:

- Univariate filtering
- Redundancy-aware refinement
- Optional interaction analysis

---

## 7. Step 6: Integration with ILP

The reduced feature set $S^*$ is used as input for:

- Predicate construction
- Background knowledge encoding
- Rule induction using ILP systems (e.g. Popper)

No further specification is required at this stage.

---

## 8. Computational Considerations

### Univariate Stage

$$
O(N \cdot F)
$$

### mRMR / JMI

$$
O(m^2)
$$

### Pairwise Exploration

$$
O(m^2)
$$

### Triplets

$$
O(m^3)
$$

---

## 9. Key Insight

This pipeline approximates:

$$
H(Y \mid X_1, \dots, X_k)
$$

without computing the full joint distribution, by:

- selecting informative features
- controlling redundancy
- exploring limited interactions

---

## 10. Summary

Pipeline:

1. Compute $I(X_i; Y)$
2. Select top $m$
3. Apply mRMR or JMI
4. Optionally evaluate pairs/triples
5. Produce reduced feature set
6. Feed into ILP

This approach balances:

- scalability
- statistical grounding
- interpretability
- compatibility with logical models
