# CASE 5 - Customer Segmentation

## 1. Contexto do desafio

Uma instituição financeira digital deseja compreender melhor sua base para desenhar estratégias diferenciadas de comunicação, produto, cross-sell, retenção e relacionamento.

A primeira pergunta obrigatória não é:

> Qual algoritmo de clustering devemos usar?

A primeira pergunta é:

> Qual decisão os segmentos precisam suportar?

Neste case, a segmentação deve ser útil para:

- campanhas e CRM;
- cross-sell;
- retenção;
- relacionamento;
- priorização de oportunidades;
- diferenciação de estratégia por perfil de cliente.

O objetivo não é produzir clusters matematicamente bonitos. É construir uma taxonomia de clientes interpretável, estável e acionável.

## 2. Unidade de análise

`customer_id x snapshot_quarter`

A base possui 300.000 clientes únicos acompanhados em três snapshots trimestrais, totalizando 900.000 linhas.

Cada cliente aparece uma vez por trimestre.

Essa estrutura permite avaliar não apenas a separação dos clusters, mas também sua estabilidade e as migrações de clientes ao longo do tempo.

## 3. Base fornecida

Arquivo único:

`data/raw/case_05_customer_segmentation_raw.csv`

Características:

- 300.000 clientes únicos;
- 3 snapshots trimestrais;
- 900.000 customer-quarter snapshots;
- variáveis de RFM, saldo, renda, cartão, Pix, merchant, produtos, investimentos, digital, CRM, receita, margem e tenure;
- outcomes futuros disponíveis exclusivamente para validação externa;
- problemas deliberados de escala, outliers, redundância, correlação, missing, não esfericidade e drift.

Nenhuma base de cluster pronta é fornecida.

## 4. Ausência de target supervisionado

Customer Segmentation é um problema não supervisionado.

Não existe target de treinamento.

Por isso, conceitos como ROC-AUC, Calibration ou OOT supervisionado não se aplicam diretamente à formação dos clusters.

A validação deve combinar:

1. qualidade geométrica;
2. estabilidade;
3. interpretabilidade;
4. actionability;
5. diferenciação futura de negócio.

## 5. Outcomes futuros para validação externa

A base contém:

- `revenue_next_90d`
- `margin_next_90d`
- `churn_next_90d`
- `product_adoption_next_90d`

Essas variáveis NÃO podem ser utilizadas:

- na preparação das features de clustering;
- na escolha inicial da geometria dos clusters;
- no ajuste de PCA;
- no treinamento dos algoritmos.

Elas devem permanecer isoladas até a etapa de validação externa.

O objetivo é perguntar depois:

> Os segmentos encontrados hoje apresentam comportamentos futuros diferentes e relevantes para o negócio?

Isso é validação externa, não treinamento supervisionado disfarçado.

## 6. Famílias de features

### RFM

- recency_days
- frequency_90d
- monetary_90d
- avg_ticket_90d
- txn_active_days_90d

### Conta e saldo

- avg_balance_90d
- max_balance_90d
- balance_volatility_90d
- income_proxy
- salary_inflow_flag

### Cartão e crédito

- has_credit_card
- credit_limit
- credit_utilization
- card_spend_90d

### Pix

- pix_count_90d
- pix_value_90d

### Merchant

- merchant_flag
- merchant_txn_count_90d
- merchant_gmv_90d

### Produtos

- has_loan
- loan_balance
- has_investment
- investment_balance
- has_insurance
- has_pos
- number_of_products

### Digital e CRM

- app_sessions_90d
- app_active_days_90d
- days_since_last_login
- push_open_rate_90d
- email_open_rate_90d
- campaigns_received_90d
- campaign_clicks_90d
- campaign_click_rate_90d

### Valor econômico atual

- revenue_90d
- margin_90d

### Relacionamento

- customer_tenure_months
- relationship_tier

O dicionário completo está em `case_05_data_dictionary.csv`.

## 7. Problemas deliberadamente presentes

A base foi construída para exigir decisões metodológicas.

Existem:

- escalas muito diferentes;
- variáveis monetárias com cauda longa;
- outliers extremos;
- zeros legítimos;
- missing estrutural;
- missing crescente em variáveis digitais;
- código sentinela;
- features redundantes;
- correlação elevada;
- variáveis categóricas;
- clusters parcialmente não esféricos;
- um nicho pequeno de alto valor;
- sobreposição entre perfis;
- migração de clientes entre snapshots;
- mudança temporal gradual de composição.

Também existem features deliberadamente redundantes:

- `total_financial_assets`
- `transactional_intensity`
- `engagement_index_raw`

A solução deve decidir se elas ajudam, duplicam informação ou distorcem distância.

## 8. Decisão central sobre o snapshot de desenvolvimento

O primeiro snapshot deve funcionar como development snapshot da taxonomia.

Recomendação:

- desenvolver a solução principalmente em 2026Q1;
- congelar o pipeline de preprocessing e a lógica de segmentação;
- aplicar a taxonomia aos snapshots 2026Q2 e 2026Q3;
- medir estabilidade, migração e drift.

Isso evita criar uma nova taxonomia totalmente diferente a cada trimestre.

## 9. Baseline obrigatória - RFM rule-based

Antes de Machine Learning, construir uma segmentação simples de negócio baseada em RFM.

Exemplo conceitual:

- Champions;
- Loyal;
- Promising;
- At Risk;
- Hibernating;

ou outra taxonomia bem justificada.

Não copiar nomes de segmentos automaticamente. Definir regras e nomes coerentes com o contexto financeiro.

A baseline deve demonstrar o valor de uma solução simples e interpretável.

## 10. Preprocessing

Avaliar:

- tratamento de missing;
- tratamento de sentinelas;
- winsorization;
- log1p para distribuições positivas e muito assimétricas;
- RobustScaler;
- StandardScaler;
- encoding das categóricas quando necessário;
- remoção de features redundantes;
- seleção baseada em racional de negócio;
- PCA como opção, não obrigação.

### PCA

PCA deve ser usado apenas se melhorar:

- estabilidade;
- eficiência computacional;
- visualização;
- tratamento de multicolinearidade.

Não usar PCA automaticamente se a perda de interpretabilidade for maior que o benefício.

## 11. Algoritmos

### Baseline

- RFM rule-based segmentation

### Challengers

- K-Means
- Gaussian Mixture Model
- Hierarchical Clustering
- HDBSCAN, se disponível

## 12. Ajuste pragmático de escala

A base possui 300 mil clientes por snapshot.

Alguns algoritmos não escalam da mesma forma.

### K-Means

Pode ser executado em grande escala.

MiniBatchKMeans pode ser considerado para eficiência.

### Gaussian Mixture

Pode ser executado com seleção cuidadosa do número de componentes e covariance type.

### Hierarchical Clustering

Clustering hierárquico aglomerativo tradicional possui custo elevado de memória e tempo para 300 mil observações.

Não tentar executá-lo ingenuamente na base completa.

Utilizar amostra representativa e documentar essa decisão.

### HDBSCAN

Também pode ser computacionalmente caro em alta dimensão e grandes volumes.

Avaliar amostra representativa, dimensionalidade e disponibilidade da biblioteca.

O objetivo é demonstrar escolha metodológica, não produzir uma execução impraticável.

## 13. Número de clusters

Para K-Means e GMM, testar uma faixa racional, por exemplo:

`k = 3 a 10`

Avaliar:

- Silhouette;
- Davies-Bouldin;
- Calinski-Harabasz;
- tamanho dos clusters;
- interpretabilidade;
- estabilidade;
- actionability.

Não selecionar k apenas pelo maior Silhouette.

## 14. Métricas internas

Avaliar:

- Silhouette Score;
- Davies-Bouldin Index;
- Calinski-Harabasz Index;
- cluster sizes;
- distância entre centroides;
- distribuição das features por cluster.

Métricas internas são diagnósticos geométricos, não objetivos de negócio.

## 15. Perfil dos clusters

Para cada cluster produzir:

- tamanho;
- participação na população;
- RFM;
- saldo;
- renda;
- utilização de cartão;
- Pix;
- produtos;
- investimentos;
- engajamento digital;
- CRM;
- receita;
- margem;
- tenure.

Transformar números em personas analíticas acionáveis.

Exemplos de descrições:

- alto valor e alto engajamento;
- digitais com potencial de cross-sell;
- merchants de alto volume;
- alto uso de crédito;
- baixo engajamento e risco de perda.

Os nomes devem emergir dos resultados.

## 16. Estabilidade temporal

A solução final deve ser aplicada aos três snapshots.

Avaliar:

- tamanho de cada segmento por trimestre;
- centroid drift;
- feature drift;
- transição de clientes entre segmentos;
- matriz de migração;
- proporção de clientes que permanecem no mesmo segmento;
- Adjusted Rand Index ou métricas similares quando metodologicamente aplicáveis;
- estabilidade das personas.

### Label alignment

Clusters não possuem identidade nominal intrínseca.

Se modelos forem ajustados separadamente por trimestre, alinhar os labels antes de comparar.

Uma alternativa preferível para a taxonomia operacional é:

- ajustar o pipeline em Q1;
- congelar;
- aplicar a mesma taxonomia em Q2 e Q3 quando o algoritmo permitir.

## 17. Validação externa

Somente depois de congelar a segmentação, utilizar:

- revenue_next_90d;
- margin_next_90d;
- churn_next_90d;
- product_adoption_next_90d.

Perguntas:

- segmentos diferem em receita futura?
- segmentos diferem em margem futura?
- segmentos diferem em churn?
- segmentos diferem em adoção de produto?
- diferenças possuem magnitude relevante?

Usar:

- tabelas;
- intervalos de confiança;
- effect sizes;
- testes globais;
- post-hoc quando pertinente.

Não transformar validação externa em caça a p-values.

## 18. Actionability Score

Criar uma avaliação qualitativa ou semiquantitativa das soluções considerando:

- interpretabilidade;
- estabilidade;
- diferenciação de negócio;
- facilidade de operacionalização;
- cobertura da população;
- tamanho mínimo útil;
- clareza das ações possíveis.

Não criar um score arbitrário sem explicar critérios.

A finalidade é comparar soluções em termos operacionais, não esconder julgamento atrás de um número.

## 19. Estratégias por segmento

Depois da seleção da solução final, propor ações exemplificativas.

Possíveis eixos:

### CRM

- frequência e canal de comunicação;
- intensidade de campanhas;
- conteúdo.

### Cross-sell

- produtos com maior aderência por perfil.

### Retenção

- grupos com maior churn futuro ou deterioração.

### Relationship

- clientes de alto valor;
- merchants;
- investidores;
- digitais;
- clientes de baixa atividade.

Essas recomendações devem ser hipóteses acionáveis, não causalidade comprovada.

## 20. Segmentação versus personalização

A conclusão deve reconhecer que segmentação é uma ferramenta de simplificação.

Para decisões individuais de alta granularidade, propensity, uplift ou recommendation podem ser mais apropriados.

Segmentos são particularmente úteis para:

- estratégia;
- comunicação;
- planejamento;
- reporting;
- desenho de campanhas;
- definição de políticas diferenciadas.

## 21. Monitoramento

Monitorar trimestralmente:

- tamanho dos segmentos;
- drift de features;
- centroid drift;
- migração;
- estabilidade;
- receita e margem;
- churn;
- adoção;
- eficácia das estratégias aplicadas.

Definir quando recalibrar ou reconstruir a taxonomia.

## 22. Conclusão obrigatória

Finalizar explicitamente com:

> O melhor clustering não é necessariamente aquele com maior Silhouette. A solução final deve equilibrar separação matemática, estabilidade temporal, interpretabilidade, diferenciação futura e capacidade de gerar ações de negócio.

---

# MASTER PROMPT PARA ambiente de desenvolvimento - CASE 5

Você atuará como Senior Data Scientist responsável por implementar um case técnico completo de Customer Segmentation aplicado a Growth em Financial Services.

O projeto deve ser desenvolvido como se fosse uma entrega de desafio técnico para uma posição Senior Data Scientist.

## PRINCÍPIO CENTRAL

A solução deve priorizar:

1. decisão de negócio suportada pela segmentação;
2. entendimento dos dados;
3. qualidade e preprocessing;
4. geometria dos dados;
5. interpretabilidade;
6. estabilidade temporal;
7. actionability;
8. validação externa;
9. eficiência computacional;
10. comunicação.

Evite overengineering.

A primeira versão deve ser integralmente compreensível em UM notebook Jupyter executável de ponta a ponta.

## REPOSITÓRIO

Criar:

`financial-growth-case-05-customer-segmentation`

Estrutura mínima:

```text
README.md
requirements.txt

data/
    raw/

notebooks/
    01_end_to_end_case.ipynb

outputs/
    figures/
    tables/
    metrics/
    models/

presentation/

src/
    somente funções reutilizáveis que aumentem clareza
```

## ARQUIVO DE ENTRADA

Utilizar exclusivamente:

`data/raw/case_05_customer_segmentation_raw.csv`

O arquivo contém 300 mil clientes acompanhados em três snapshots trimestrais.

Não criar arquivos de Train/Test supervisionados.

## OUTCOMES FUTUROS PROIBIDOS NA CLUSTERIZAÇÃO

Não utilizar em preprocessing, PCA, distance calculation, fit ou seleção inicial de clusters:

- revenue_next_90d
- margin_next_90d
- churn_next_90d
- product_adoption_next_90d

Essas colunas são exclusivamente para validação externa depois da segmentação congelada.

## SNAPSHOT DE DESENVOLVIMENTO

Usar prioritariamente 2026Q1 para desenvolver a taxonomia.

Depois:

- congelar o preprocessing;
- congelar feature set;
- congelar modelo/regra;
- aplicar em 2026Q2;
- aplicar em 2026Q3;
- medir estabilidade e migração.

Quando um algoritmo não possuir predict natural, documentar estratégia de aplicação e comparação temporal.

## ESTRUTURA DO NOTEBOOK

1. Contexto e decisões de negócio
2. Unidade de análise
3. Estrutura longitudinal
4. Bibliotecas e seeds
5. Carga da base raw
6. Data Quality Report
7. Tipagem
8. Missing
9. Sentinelas
10. Cardinalidade
11. Outliers
12. Distribuições
13. Skewness e caudas longas
14. Correlação
15. Redundância
16. Drift entre snapshots
17. Definição das features permitidas
18. Exclusão dos outcomes futuros
19. Baseline RFM
20. Perfil da baseline
21. Estratégias de winsorization
22. Log transformation
23. Scaling
24. StandardScaler versus RobustScaler
25. PCA como experimento opcional
26. K-Means
27. MiniBatchKMeans se útil
28. Gaussian Mixture
29. Hierarchical em amostra representativa
30. HDBSCAN em configuração computacionalmente viável
31. Seleção de número de clusters
32. Silhouette
33. Davies-Bouldin
34. Calinski-Harabasz
35. Cluster sizes
36. Cluster profiles
37. Interpretabilidade
38. Actionability
39. Seleção da solução candidata
40. Congelamento da taxonomia
41. Aplicação em Q2
42. Aplicação em Q3
43. Cluster size drift
44. Centroid drift
45. Migration matrix
46. Customer transitions
47. Stability metrics
48. Validação externa com revenue_next_90d
49. Validação externa com margin_next_90d
50. Validação externa com churn_next_90d
51. Validação externa com product_adoption_next_90d
52. Estratégias por segmento
53. Limitações
54. Monitoramento
55. Recalibration criteria
56. Próximos passos
57. Conclusão executiva

## BASELINE RFM

Construir uma segmentação rule-based usando:

- recency_days;
- frequency_90d;
- monetary_90d.

Criar scores ou faixas interpretáveis.

Definir regras transparentes.

Não assumir que RFM será inferior aos métodos não supervisionados.

Comparar actionability e estabilidade.

## DATA QUALITY

Investigar:

- missing;
- missing estrutural;
- sentinelas;
- zeros legítimos;
- heavy tails;
- outliers;
- correlação;
- features redundantes;
- drift temporal.

## PREPROCESSING

Comparar racionalmente:

- winsorization;
- log1p;
- RobustScaler;
- StandardScaler.

Não aplicar transformação indiscriminadamente.

Preservar interpretabilidade quando possível.

## PCA

Tratar como experimento.

Avaliar explained variance.

Usar PCA se houver benefício claro em:

- desempenho;
- visualização;
- estabilidade;
- redução de redundância.

Não tornar PCA obrigatório.

## K-MEANS

Testar faixa razoável de k.

Avaliar:

- inertia;
- Silhouette;
- Davies-Bouldin;
- Calinski-Harabasz;
- cluster size;
- business profile.

Não selecionar pelo elbow isoladamente.

## GMM

Avaliar diferentes números de componentes.

Considerar:

- covariance type;
- BIC;
- AIC;
- posterior probabilities;
- overlap entre segmentos.

A soft assignment do GMM pode ser informativa para clientes próximos de fronteiras.

## HIERARCHICAL

Não executar Agglomerative Clustering ingenuamente sobre 300 mil clientes.

Utilizar amostra representativa.

Documentar:

- tamanho da amostra;
- linkage;
- distance metric;
- dendrogram quando viável.

Usar principalmente como ferramenta de compreensão da estrutura.

## HDBSCAN

Se disponível, testar em amostra ou representação de dimensionalidade compatível.

Avaliar:

- min_cluster_size;
- noise share;
- estabilidade;
- clusters pequenos.

Não considerar todo ruído automaticamente como problema.

## CLUSTERS NÃO ESFÉRICOS

Comparar como K-Means, GMM e HDBSCAN lidam com geometrias diferentes.

Não assumir que K-Means é superior apenas por escala.

## SMALL NICHE SEGMENT

Existe estrutura de nicho pequeno na população.

Avaliar se algum método consegue identificá-la sem fragmentar excessivamente a base.

Perguntar:

- o nicho é economicamente relevante?
- é estável?
- é acionável?
- possui tamanho suficiente para estratégia?

## PERFIL DOS SEGMENTOS

Criar tabela de profiling contendo medianas/médias adequadas para:

- RFM;
- balance;
- income;
- card;
- Pix;
- merchant;
- products;
- investment;
- app;
- CRM;
- revenue;
- margin;
- tenure.

Normalizar contra a população quando útil.

## STABILITY

Depois de selecionar a solução:

Aplicar em Q2 e Q3.

Construir:

- cluster share por quarter;
- migration matrix;
- retention in segment;
- centroid drift;
- feature drift.

Usar ARI/NMI apenas quando a comparação de labels for metodologicamente válida.

## LABEL ALIGNMENT

Clusters são identificados por números arbitrários.

Nunca comparar Cluster 0 de dois modelos diferentes apenas pelo número.

Se houver refit independente, alinhar clusters por centroid similarity ou Hungarian matching antes de comparar.

Preferir, quando operacionalmente possível, taxonomia congelada em Q1 aplicada aos snapshots seguintes.

## VALIDAÇÃO EXTERNA

Somente depois de congelar a segmentação, abrir os outcomes futuros.

Comparar por cluster:

- mean/median revenue_next_90d;
- mean/median margin_next_90d;
- churn_next_90d rate;
- product_adoption_next_90d rate.

Calcular:

- diferenças absolutas;
- effect sizes;
- intervalos de confiança;
- testes globais quando pertinentes.

Não selecionar clusters exclusivamente pelo menor p-value.

## ACTIONABILITY

Para cada segmento responder:

1. Quem são?
2. Qual comportamento os diferencia?
3. Qual valor geram?
4. Qual risco apresentam?
5. Qual oportunidade existe?
6. Que tipo de ação de CRM, cross-sell ou retenção é plausível?
7. A ação é operacionalmente executável?

## SELEÇÃO FINAL

A solução escolhida deve equilibrar:

- mathematical separation;
- cluster size;
- temporal stability;
- interpretability;
- future business differentiation;
- actionability;
- computational cost.

Não escolher automaticamente a maior Silhouette.

## APRESENTAÇÃO

Estruturar no máximo cinco slides:

1. Business Problem, Decision & Data
2. Data Geometry & Preprocessing
3. Segmentation Benchmark & Profiles
4. Stability & Future Business Validation
5. Actionable Segments, Strategy & Next Steps

## CONCLUSÃO OBRIGATÓRIA

Finalizar explicitamente:

"O melhor clustering não é necessariamente aquele com maior Silhouette. A solução final deve equilibrar separação matemática, estabilidade temporal, interpretabilidade, diferenciação futura e capacidade de gerar ações de negócio."

