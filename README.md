# Financial Growth Case 05 — Customer Segmentation

Case técnico end-to-end de segmentação acionável de clientes em serviços financeiros, desenvolvido para apoiar CRM, cross-sell, retenção e estratégia de relacionamento. A solução compara uma baseline RFM com métodos não supervisionados, congela uma taxonomia em um trimestre de desenvolvimento e mede sua estabilidade e utilidade de negócio nos trimestres seguintes.

## Objetivo

Construir segmentos que sejam simultaneamente distintos, estáveis, interpretáveis e úteis para decisões. O projeto separa deliberadamente duas etapas:

1. formação dos segmentos usando apenas atributos disponíveis no instante do snapshot;
2. validação externa posterior com receita, margem, churn e adoção de produto nos 90 dias seguintes.

Os outcomes futuros nunca participam do preprocessamento, da escolha do algoritmo, do número de clusters ou da atribuição dos segmentos.

## Dados e desenho temporal

- 900.000 registros `customer_id × snapshot_quarter`;
- 300.000 clientes únicos presentes em 2026Q1, 2026Q2 e 2026Q3;
- 52 variáveis de comportamento, relacionamento, produto e valor;
- 2026Q1: desenvolvimento e congelamento da taxonomia;
- 2026Q2: validação temporal e diagnóstico de estabilidade;
- 2026Q3: holdout temporal final;
- outcomes futuros: usados somente após o congelamento, para validação externa.

## Metodologia

- auditoria de qualidade, missing, sentinelas, escala, caudas, outliers, redundância e correlação;
- baseline RFM baseada em regras congeladas em 2026Q1;
- preprocessamento aprendido somente em 2026Q1: tratamento semântico de missing, indicadores de ausência, winsorization, `log1p` e escala robusta;
- benchmark pragmático de K-Means, Gaussian Mixture, clustering hierárquico em amostra e HDBSCAN quando disponível;
- seleção considerando métricas internas, tamanho mínimo, interpretabilidade e capacidade operacional;
- aplicação do pipeline congelado em 2026Q2 e 2026Q3;
- estabilidade por participação, retenção de rótulo, matriz de migração e drift de centroides;
- validação externa por receita, margem, churn e adoção futura;
- tradução dos perfis em hipóteses de ação, guardrails e plano de teste.

## Estrutura

```text
data/raw/          base pública e dicionário de dados
notebooks/         notebook executado de ponta a ponta
outputs/figures/   gráficos analíticos
outputs/tables/    perfis, estabilidade, validação e ações
outputs/metrics/   benchmark e registro de congelamento
outputs/models/    preprocessador, modelo e taxonomia congelados
presentation/      apresentação executiva de cinco slides
src/               funções reutilizáveis e pipeline determinístico
tests/             testes das regras críticas
```

## Execução

No Git Bash, a partir da raiz do repositório:

```bash
py -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.run_case
pytest -q
deactivate
```

O script regenera os artefatos analíticos e o notebook. O CSV raw é versionado com Git LFS.

## Princípio de decisão

O melhor clustering não é necessariamente aquele com maior Silhouette. A solução final deve equilibrar separação matemática, estabilidade temporal, interpretabilidade, diferenciação futura e capacidade de gerar ações de negócio.

Os segmentos identificam padrões e apoiam hipóteses de ação; eles não provam que uma campanha, oferta ou abordagem causará melhora de receita, retenção ou adoção. Essas hipóteses devem ser validadas por experimentos controlados.

