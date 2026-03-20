# PADTAI pipeline (malware binary)

Estrutura mínima para integrar PADTAI no projeto principal.

## Pastas
- `data/ilp/`: datasets CSV prontos para PADTAI (ex.: top200_ig_malware.csv)
- `reports/padtai/`: regras encontradas + métricas resumidas
- `logs/padtai/`: logs de execução
- `src/padtai/`: scripts utilitários de preparação e análise

## Convenções propostas
1. Gerar tabela com:
   - colunas = top-200 IG
   - última coluna = `label`
2. Executar PADTAI em modo sem aritmética:
   - `--grounded none`
   - `--intcols none`
3. Fazer 3 corridas com seeds/ordens diferentes e consolidar regras estáveis.

## Próximo passo
Criar script de preparação da tabela top-200 em `src/padtai/` e um script runner para chamadas repetidas ao PADTAI.


## ⚠️ Warning: setuptools compatibility issue
If you encounter an error related to `setuptools` when running the PADTAI pipeline, please ensure that you have the correct version of `setuptools` installed. 81
