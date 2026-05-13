# Analysis Specification — Per-cluster Feature vs Method

Objetivo
--------
Documentar de forma precisa o que a análise por cluster deve produzir e como os cálculos são feitos. Este ficheiro é apenas especificação; não contém implementação.

Inputs (canónicos)
-------------------
- Per-cluster summary CSV (existente):
  - Path pattern: `reports/entropy_knn/analysis/per_cluster_feature_vs_method/cluster_{i}/top_feature_candidates.csv`
  - Columns esperadas (exemplo presente):
    - `feature` (string)
    - `method_count` (int)
    - `aggregated_score` (float)
    - `score_std` (float)
    - `rank_mean` (float)
    - `rank_std` (float)
    - `top_methods` (string, comma-separated)
    - `<method_name>` for cada método (e.g. `entropy_reduction_ratio`, `mutual_information`, `chi2_stat`, `f_stat`, `pearson_r`)

- Raw per-method score artifacts (optional alternative input):
  - JSON-first loader: `reports/entropy_knn/.../scores_{method}.json` or per-cluster JSON files. If JSON ausente, usar parquet fallback.

- (Opcional, futuro) `data/training_set.csv` — apenas necessário para STAGE 2 (validação), não para esta análise.

Assunções
---------
- Existem 5 métodos de scoring: `entropy_reduction_ratio`, `mutual_information`, `chi2_stat`, `f_stat`, `pearson_r`.
- Scores maiores = mais importantes (maior relevância).
- Cada método pode devolver score nulo/NaN para algumas features — tratar como ausência.

Parâmetros default
------------------
- `normalization`: `'minmax'` (opções: `'minmax'`, `'zscore'`, `'none'`)
- `top_k` (por método): `20`
- `consensus_top_n` (final candidates por cluster): `15`
- `annotation_threshold`: `method_count > 0` (apenas anotar features com pelo menos um método)
- `colormap`: `'viridis'`
- `heatmap_max_annotate`: `20` (anotar células apenas quando número de linhas <= 20)

Definições e Fórmulas
---------------------
1. Normalização (por método):
   - MinMax: x' = (x - min) / (max - min). If max==min => if all equal then x'=0.5 (or fallback to `'none'` and keep original scores), plus log a warning.
   - Z-score: x' = (x - mean) / std. If std==0 => fallback to 0.
   - None: keep original values.

2. Top-K por método (construção de conjuntos):
   - Para cada método m, ordenar features por `score_m` descendente e tomar os `top_k` primeiros. Este conjunto é `topk_set_m`.
   - Se um método não tem `top_k` features (fewer rows), o `topk_set` é o conjunto disponível.

3. `method_count` (por feature f):
   - method_count(f) = |{ m : f ∈ topk_set_m }|
   - Valor inteiro entre 0 e M (M = número de métodos com scores válidos).

4. `aggregated_score` (por feature f):
   - Método principal: média simples das pontuações normalizadas across all methods that provide a score for f.
   - Formula: aggregated_score(f) = mean_m( normalized_score_m(f) ), ignorando NaNs.
   - Alternativa (config): soma ponderada se escolher weights later.

5. `score_spread` (por feature f):
   - Desvio padrão das pontuações normalizadas across methods that provide a score for f (sample std, ddof=0).

6. `rank_mean` / `rank_std` (opcional):
   - Se ranks forem fornecidos por método, usar rank 1 = melhor. rank_mean = mean ranks; rank_std = std ranks.
   - Usados apenas como tie-breakers e para inspeção.

Ordenação / Seleção de candidatos
---------------------------------
- Critério principal para ordenar features:
  1. `method_count` desc (maior concordância entre métodos)
  2. `aggregated_score` desc (entre features com mesmo method_count)
  3. `score_spread` asc (preferir features com menor variabilidade entre métodos)
  4. `rank_mean` asc (menor média de rank é melhor)
- Seleção final: escolher os `consensus_top_n` primeiros segundo esta ordenação e gravar em `top_feature_candidates.csv` (conservar todas as colunas de suporte).

Tie-breakers e regras adicionais
--------------------------------
- Empates exactos permanecem ordenados por `feature` (alfabético) para estabilidade.
- Se `aggregated_score` não puder ser calculado (p. ex. todas as entradas NaN), colocar `aggregated_score=NaN` e ordenar abaixo de todas as features com valor numérico.
- Se `method_count`=0 para todas as features, selecionar as top-N por `aggregated_score` (ou por `rank_mean` se disponível).

Method-priority tie-breaker
---------------------------
O professor pediu primazia por métodos em casos de empate. Recomenda-se a seguinte ordem de prioridade (maior → menor):

1. `pearson_r`
2. `f_stat`
3. `chi2_stat`
4. `mutual_information`
5. `entropy_reduction_ratio`

Duas estratégias de aplicação (configurável):

- Lexicográfica (default): para cada feature construir um vetor binário de presença em `top_k` na ordem acima (ex.: `[pearson, f, chi2, mi, enr]`) e comparar esses vetores lexicograficamente; o vector com `1` mais à esquerda vence.
- Ponderada: atribuir pesos por método e calcular um `aggregated_score_weighted` = sum_m( weight_m * normalized_score_m ). Pesos sugeridos: `pearson_r=1.0`, `f_stat=0.9`, `chi2_stat=0.8`, `mutual_information=0.6`, `entropy_reduction_ratio=0.5`.

Regra operacional: por omissão usar Lexicográfica. Aplicar o tie-breaker escolhido quando `method_count` e `aggregated_score` empatarem.

Outputs esperados (por cluster)
-------------------------------
- `feature_method_summary.csv` — tabela completa com colunas: `feature`, `method_count`, `aggregated_score`, `score_std`, `rank_mean`, `rank_std`, `top_methods`, `entropy_reduction_ratio`, `mutual_information`, `chi2_stat`, `f_stat`, `pearson_r`.
- `top_feature_candidates.csv` — subset dos candidatos finais (`consensus_top_n`), com as mesmas colunas.
- `visualizations/heatmap.png` — heatmap feature × method (scores normalizados), colormap `viridis`, anotar valores quando linhas ≤ `heatmap_max_annotate`.
- `visualizations/method_bars.png` — barras horizontais: por método, barras dos `aggregated_score` ou score por método para as top features. Tamanho e rótulos legíveis.
- `visualizations/agreement.png` — barras das contagens de `method_count` entre candidatos.
- `visualizations/spread_vs_score.png` — scatter `aggregated_score` (x) vs `score_spread` (y), tamanho de ponto proporcional a `method_count`; anotar pontos com `method_count>0` e/ou com `aggregated_score` acima do percentil 90.

Formato de ficheiros e convenções
--------------------------------
- CSVs: `utf-8`, `,` separator, header, quoting minimal.
- PNGs: 300 DPI, figsize default 10×6 (ajustável por parâmetro), transparent background opcional.
- Todos os outputs gravar em: `reports/entropy_knn/analysis/per_cluster_feature_vs_method/cluster_{i}/` or subdir `visualizations/`.

Regras de logging e validação
-----------------------------
- Validar que pelo menos um método tem scores válidos; se nenhum, log de erro e skip do cluster.
- Logar warnings para: constantes por método (max==min), métodos com >50% NaNs, menos de `top_k` features disponíveis.
- Salvar um pequeno `run_metadata.json` em cada cluster output contendo parâmetros usados (`top_k`, `normalization`, timestamp, source files used).

Visual testing / checks
-----------------------
- Para uma amostra (ex.: cluster_0), validar manualmente:
  - O `feature_method_summary.csv` contém colunas por método com valores entre 0 e 1 (se `minmax`), ou z-scores se `zscore`.
  - `method_count` das features top correspondem à presença em top-Ks específicos.
  - Heatmap mostra comportamentos esperados (alto contraste para features com high agreement).

Edge cases
----------
- Poucas features total (< `consensus_top_n`): devolver todas e marcar esse cluster com `run_metadata['note'] = 'few_features'`.
- Métodos com direções opostas (se algum método tiver score invertido): assumir convenção "maior é melhor"; se detectado o contrário (metodo doc), aplicar transformação (1 - x) ou normalização apropriada. Documentar na metadata.

Parâmetros CLI (proposta)
-------------------------
- `--cluster-dir` path to cluster outputs (default pattern above)
- `--cluster-id` specific id or `--all`
- `--top-k` int (default 20)
- `--consensus-top-n` int (default 15)
- `--normalization` ['minmax','zscore','none']
- `--out-dir` override output directory
- `--annotate-threshold` int (default 0)

Testes unitários a incluir (após aprovação)
------------------------------------------
- Normalization tests: minmax, zscore, none; constant-array handling.
- method_count test: construção de topk sets e contagem.
- aggregated_score/spread computation with NaNs.
- Ordering/tie-breaker behaviour test with synthetic inputs.

Próximos passos (não implementar sem aprovação)
------------------------------------------------
1. Rever esta spec e aprovar/editar defaults.
2. Depois de aprovado: implementar módulos em `src/analysis/entropy_knn_analysis/` conforme `REORGANIZATION_PLAN.md`.
3. Escrever testes unitários e um runner `--dry-run`.

Histórico de versão
-------------------
- v0.1 — Spec inicial com defaults: `normalization=minmax`, `top_k=20`, `consensus_top_n=15`.
