# Entropy-Based Feature Selection Pipeline for Malware Detection

## 1. Entropy Evaluation

For each feature $X_i$, compute its relevance with respect to the class variable $Y$.

### 1.1 Entropy of the Class

$$H(Y) = - \sum_{y \in Y} P(y)\log_2 P(y)$$

#### Example 1: Balanced Classes (50% / 50%)

$$H(Y) = - \left(0.5 \log_2(0.5) + 0.5 \log_2(0.5)\right) = 1$$

This shows why we use $\log_2$: entropy is measured in bits.

#### Example 2: Rare Malware (5% / 95%)

$$H(Y) = - \left(0.05 \log_2(0.05) + 0.95 \log_2(0.95)\right) \approx 0.286$$

### 1.2 Conditional Entropy

$$H(Y \mid X_i) = \sum_{x \in X_i} P(x) H(Y \mid X_i = x)$$

$$H(Y \mid X_i = x) = - \sum_{y \in Y} P(y \mid x)\log_2 P(y \mid x)$$

### 1.3 Manual Conditional Entropy Example

| X1 | X2 | Y       |
|:--:|:--:|---------|
| 1  | 1  | malware |
| 1  | 1  | malware |
| 1  | 0  | malware |
| 1  | 0  | benign  |
| 0  | 1  | benign  |
| 0  | 0  | benign  |
| 0  | 1  | benign  |
| 0  | 0  | malware |

**Class counts:** 4 malware, 4 benign

#### For $X_1$

$$H(Y \mid X_1 = 1) = - \left(\frac{3}{4}\log_2\frac{3}{4} + \frac{1}{4}\log_2\frac{1}{4}\right) \approx 0.811$$

$$H(Y \mid X_1 = 0) = - \left(\frac{1}{4}\log_2\frac{1}{4} + \frac{3}{4}\log_2\frac{3}{4}\right) \approx 0.811$$

$$H(Y \mid X_1) = 0.5 \cdot 0.811 + 0.5 \cdot 0.811 = 0.811$$

#### For $X_2$

$$H(Y \mid X_2 = 1) = 1$$

$$H(Y \mid X_2 = 0) = 1$$

$$H(Y \mid X_2) = 1$$

In this example, **X1** has lower conditional entropy than **X2**, so we can infer that **X1** is more **informative** about the class.

#### Computational Complexity.
For a binary feature, computing the conditional entropy requires counting the class frequencies for each feature value. This can be done in one pass through the dataset, which costs O(n) for a single feature, where n is the number of samples. Repeating this process for m features gives a total complexity of O(nm).

### 1.4 Conditional Entropy with Both Features

$$H(Y \mid X_1, X_2) = \sum_{x_1, x_2} P(x_1, x_2) H(Y \mid x_1, x_2)$$

$$H(Y \mid X_1, X_2) = 0.5$$

So, using both features together reduces class uncertainty to 0.5 bits.

#### Computational Complexity
While computing the conditional entropy for a single pair of features has cost O(n), evaluating all possible pairs requires m^2 computations, leading to an overall complexity of O(nm2 ).

#### Combinatorial Growth of Feature Interactions.
Extending the analysis to higher-order feature interactions leads to a rapid increase in computational complexity. For a subset of k features, the number of possible combinations is given by m^k , where m is the total number of features. For fixed k, this grows on the order of O(mk ). This combinatorial growth quickly makes exhaustive exploration of higher-order interactions impractical.

## 2. New Pipeline

To address the combinatorial growth of feature interactions while preserving informative structures in the data, we propose a multi-stage feature selection and reduction pipeline. The goal is to balance computational efficiency with the ability to capture both individual and joint contributions of features to class uncertainty.

### 2.1 Step 1: Univariate Filtering via Information Gain

We begin by computing the Information Gain (IG) of each feature with respect to the class variable. This step has complexity $O(nm)$ and allows us to rank all features according to their individual relevance. We retain the top $m'$ features (e.g., $m' = 1000$), significantly reducing the initial dimensionality.

### 2.2 Step 2: Feature Clustering

Next, we group the selected features into clusters using a similarity-based approach, such as k-nearest neighbors (KNN) in feature space. The objective is to identify subsets of features that exhibit similar behavior or redundancy patterns, enabling localized analysis of interactions.

### 2.3 Step 3: Intra-cluster Entropy Reduction

Within each cluster, we compute the conditional entropy $H(Y \mid X_j)$ for each feature and retain the top $k$ features that minimize the residual uncertainty of the class. Optionally, a threshold can be imposed relative to the base entropy $H(Y)$ to discard features that do not significantly reduce uncertainty. We then retain only the top $k$ features per cluster that achieve a reduction above a predefined threshold (e.g., $r_j > 0.7$). This step ensures that only highly informative features are preserved within each local group.

### 2.4 Step 4: Higher-order Interaction Refinement

If the resulting feature set is still too large for Inductive Logic Programming (ILP) processing, we can refine it by analyzing pairwise or triplet interactions within each cluster.

### 2.5 Step 5: ILP-based Relational Modeling

Finally, the reduced set of features and corresponding instances are provided as input to an ILP system.
