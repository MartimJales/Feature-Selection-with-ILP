# TODO - Pipeline Balanced + Consensus + PADTAI

Lista de problemas e melhorias identificados para tratar incrementalmente.

## Prioridade Alta - Correcao da Analise

- [x] Corrigir possivel desalinhamento entre clusters balanced e dados usados no ILP.
- [ ] Corrigir filtro de clusters com malware, que atualmente procura `padtai_input.csv` antes de ele existir.
- [x] Garantir que o PADTAI usa as amostras corretas do cluster balanced, nao indices aplicados ao dataset original.
- [x] Guardar/propagar metadados suficientes para reconstruir exatamente quais amostras entram em cada cluster.
- [ ] Confirmar se clusters sem malware devem ser ignorados, reportados separadamente, ou usados como negativos.

## Prioridade Media - Performance

- [x] Evitar recarregar `extracted_features.parquet` e `training_set.csv` para cada cluster.
- [x] Adicionar cache/reuso dos datasets durante a fase ILP.
- [ ] Adicionar mecanismo de resume/skip para clusters ja concluidos.
- [ ] Avaliar paralelizacao controlada do PADTAI por cluster.
- [ ] Rever timeout de 1200s por cluster e definir politica para falhas/timeouts.

## Prioridade Media - Qualidade das Regras PADTAI

- [ ] Rever `--sample-size 100`, porque pode ser pouco representativo para clusters com ate 500 amostras.
- [ ] Tornar o sampling do PADTAI reprodutivel com seed fixa.
- [ ] Avaliar se `--solver rc2` deve continuar fixo ou se `nuwls` e melhor.
- [ ] Rever `--grounded none`, porque pode estar a desativar operadores grounded em vez de usar o default.
- [ ] Guardar metricas estruturadas por regra: coverage, recall, precision, cluster, features usadas.

## Prioridade Media - Interpretabilidade

- [ ] Guardar o mapeamento entre nomes originais das features e nomes sanitizados para Prolog.
- [ ] Detetar colisoes na sanitizacao de nomes de features.
- [ ] Melhorar extracao de regras do stdout/stderr do PADTAI.
- [ ] Criar resumo final por cluster com `n_rules`, malware/goodware, features, status e metricas.

## Prioridade Baixa - Reprodutibilidade e Operacao

- [ ] Remover ou tornar opcional o `git pull` automatico no script overnight.
- [ ] Parametrizar valores hardcoded no bash script.
- [ ] Guardar config completa de cada execucao.
- [ ] Consolidar logs e outputs para facilitar auditoria.
- [ ] Reduzir ou configurar granularidade das notificacoes Discord.
- [ ] Documentar explicitamente que os clusters KNN podem sobrepor amostras.

## Notas da Sessao

- [x] Preservado o shuffle do dataset balanced, mas guardado o mapeamento `original_indices`.
- [x] Adicionado `original_sample_indices` aos JSONs dos clusters quando existe mapeamento original.
- [x] Adicionado caminho ILP `run_ilp_cluster_from_data(...)` para usar `features_df` e `labels` ja carregados em memoria.
- [x] Atualizada a pipeline balanced completa para reutilizar `pipeline.bundle.X` e `pipeline.bundle.y` durante a fase ILP.
- [ ] Ainda falta corrigir o filtro de clusters com malware, solver/sample-size PADTAI, resume, paralelizacao e melhorias de interpretabilidade.
