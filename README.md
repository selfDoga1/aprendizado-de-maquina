# Predição do Desempenho Acadêmico de Estudantes

Este projeto tem como objetivo prever a nota final dos estudantes (G3) com base em características pessoais, socioeconômicas, comportamentais e histórico escolar. Foram aplicadas e comparadas diversas técnicas de regressão, com foco em performance preditiva e estabilidade.

## 📊 Modelos Utilizados

- **Ridge Regression**  
  Modelo de regressão linear com regularização L2, que reduz o impacto de variáveis colineares, penalizando grandes coeficientes, mas sem zerá-los.

- **Lasso Regression**  
  Também é um modelo linear, mas com regularização L1, capaz de eliminar variáveis irrelevantes ao zerar seus coeficientes, promovendo seleção automática de atributos.

- **Elastic Net**  
  Combina os efeitos do Ridge e do Lasso (L1 + L2), equilibrando a regularização e a seleção de variáveis. É útil quando há muitas variáveis correlacionadas.

- **Árvore de Decisão**  
  Modelo baseado em divisões sucessivas dos dados em ramos, criando uma estrutura semelhante a uma árvore. Simples de interpretar, mas propenso a overfitting.

- **Random Forest**  
  Conjunto de várias árvores de decisão (modelo de ensemble), onde cada árvore é treinada com uma amostra aleatória dos dados. Resulta em maior robustez e generalização.

- **Support Vector Regression (SVR)**  
  Modelo que busca encontrar uma função que se mantenha dentro de uma margem de erro (`epsilon`) e usa o conceito de margens e vetores de suporte. É eficaz para dados não lineares e de alta dimensão.


## ⚙️ Metodologia

- **Pré-processamento**: Utilização de `ColumnTransformer` para codificação de variáveis categóricas com `OneHotEncoder` e normalização de dados quando necessário.
- **Validação**: Repeated K-Fold Cross-Validation (5 folds, 30 repetições).
- **Otimização de Hiperparâmetros**: `RandomizedSearchCV`.

### 🔧 Hiperparâmetros Otimizados

Abaixo estão os hiperparâmetros otimizados para cada modelo, juntamente com uma breve explicação sobre sua função:

#### 🔹 Elastic Net
- `alpha`: Controla a intensidade da regularização (quanto maior, mais penalização).
- `l1_ratio`: Define a proporção entre L1 (Lasso) e L2 (Ridge); 0 = só Ridge, 1 = só Lasso.

#### 🔹 Lasso Regression
- `alpha`: Parâmetro de regularização que controla o quanto os coeficientes são reduzidos. Pode zerar variáveis irrelevantes.

#### 🔹 Ridge Regression
- `alpha`: Também controla a regularização, mas sem zerar variáveis; reduz o impacto de colinearidade.

#### 🔹 Random Forest
- `n_estimators`: Número de árvores na floresta. Mais árvores geralmente aumentam a estabilidade.
- `max_depth`: Profundidade máxima de cada árvore. Controla o nível de detalhamento.
- `min_samples_split`: Número mínimo de amostras para dividir um nó.
- `min_samples_leaf`: Número mínimo de amostras em uma folha.
- `max_features`: Número de atributos considerados em cada divisão. Afeta diversidade entre árvores.

#### 🔹 Decision Tree
- `max_depth`: Profundidade máxima da árvore.
- `min_samples_split`: Mínimo de amostras para que um nó seja dividido.
- `min_samples_leaf`: Mínimo de amostras em uma folha.
- `max_features`: Número de atributos avaliados por divisão (None = todos).

#### 🔹 SVR (Support Vector Regression)
- `C`: Penalidade por erro. Valores altos tentam acertar mais, mas podem overfitar.
- `epsilon`: Margem de tolerância ao erro. Erros dentro dessa faixa não são penalizados.
- `kernel`: Tipo de função usada para projetar os dados (ex: linear, rbf).
- `degree`: Grau do polinômio, se `kernel='poly'`.
- `gamma`: Define o alcance da influência de um único exemplo. Usado em kernels não-lineares.


## 🏆 Comparativo de modelos

| Modelo           | R² Médio | Desvio Padrão | Melhor R² |
|------------------|----------|----------------|------------|
| Elastic Net      | 0.9485   | 0.0009         | 0.9504     |
| Lasso Regression | 0.9485   | 0.0009         | 0.9504     |
| Ridge Regression | 0.9459   | 0.0014         | 0.9494     |
| Random Forest    | 0.9366   | 0.0022         | 0.9405     |
| Decision Tree    | 0.9083   | 0.0070         | 0.9237     |
| SVR              | 0.8445   | 0.0032         | 0.8537     |

➡️ **Elastic Net** e **Lasso Regression** apresentaram os melhores desempenhos, com excelente estabilidade (menores desvios padrão).

## 📁 Dataset

- **Fonte**: [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)
- **Variável Alvo**: Nota final (G3)
- **Características**: Demográficas, socioeconômicas, comportamentais e desempenho escolar anterior.

## 📄 Sobre os Dados

O conjunto de dados contém informações sobre estudantes do ensino médio em Portugal, matriculados nos cursos de Matemática e Língua Portuguesa.

### 🔢 Atributos

| Atributo     | Descrição                                 | Valores Possíveis                                   | Tipo       | Representação pós pre-processamento                                          |
| ------------ | ----------------------------------------- | --------------------------------------------------- | ---------- |-----------------------------------------------------------------------------|
| `school`     | Escola                                    | `GP`, `MS`                                          | Categórico | `MS` (1 se MS, 0 se GP)                                                     |
| `sex`        | Sexo                                      | `F`, `M`                                            | Categórico | `M` (1 se masculino, 0 se feminino)                                         |
| `age`        | Idade                                     | 15 a 22                                             | Numérico   | Mantido como está                                                           |
| `address`    | Tipo de endereço                          | `U`, `R`                                            | Categórico | `R` (1 se rural, 0 se urbano)                                               |
| `famsize`    | Tamanho da família                        | `LE3`, `GT3`                                        | Categórico | `GT3` (1 se >3 membros, 0 se ≤3)                                            |
| `Pstatus`    | Estado civil dos pais                     | `T`, `A`                                            | Categórico | `A` (1 se separados, 0 se juntos)                                           |
| `Medu`       | Escolaridade da mãe                       | 0 a 4                                               | Numérico   | Mantido como está                                                           |
| `Fedu`       | Escolaridade do pai                       | 0 a 4                                               | Numérico   | Mantido como está                                                           |
| `Mjob`       | Profissão da mãe                          | `teacher`, `health`, `services`, `at_home`, `other` | Categórico | 4 colunas (one-hot sem `at_home`): `health`, `other`, `services`, `teacher` |
| `Fjob`       | Profissão do pai                          | `teacher`, `health`, `services`, `at_home`, `other` | Categórico | 4 colunas (one-hot sem `at_home`)                                           |
| `reason`     | Motivo da escolha da escola               | `home`, `reputation`, `course`, `other`             | Categórico | 3 colunas (one-hot sem `course`): `home`, `other`, `reputation`             |
| `guardian`   | Responsável legal                         | `mother`, `father`, `other`                         | Categórico | 2 colunas (one-hot sem `mother`): `father`, `other`                         |
| `traveltime` | Tempo de deslocamento casa–escola         | 1 a 4                                               | Numérico   | Mantido como está                                                           |
| `studytime`  | Tempo semanal de estudo                   | 1 a 4                                               | Numérico   | Mantido como está                                                           |
| `failures`   | Nº de reprovações anteriores              | 0 a 4                                               | Numérico   | Mantido como está                                                           |
| `schoolsup`  | Apoio educacional extra                   | `yes`, `no`                                         | Categórico | `yes` (1 se sim, 0 se não)                                                  |
| `famsup`     | Apoio educacional da família              | `yes`, `no`                                         | Categórico | `yes` (1 se sim, 0 se não)                                                  |
| `paid`       | Aulas particulares pagas                  | `yes`, `no`                                         | Categórico | `yes` (1 se sim, 0 se não)                                                  |
| `activities` | Participa de atividades extracurriculares | `yes`, `no`                                         | Categórico | `yes` (1 se sim, 0 se não)                                                  |
| `nursery`    | Frequentou pré-escola                     | `yes`, `no`                                         | Categórico | `yes` (1 se sim, 0 se não)                                                  |
| `higher`     | Deseja cursar o ensino superior           | `yes`, `no`                                         | Categórico | `yes` (1 se sim, 0 se não)                                                  |
| `internet`   | Acesso à internet em casa                 | `yes`, `no`                                         | Categórico | `yes` (1 se sim, 0 se não)                                                  |
| `romantic`   | Está em relacionamento amoroso            | `yes`, `no`                                         | Categórico | `yes` (1 se sim, 0 se não)                                                  |
| `famrel`     | Relação familiar                          | 1 a 5                                               | Numérico   | Mantido como está                                                           |
| `freetime`   | Tempo livre após a escola                 | 1 a 5                                               | Numérico   | Mantido como está                                                           |
| `goout`      | Frequência de saídas com amigos           | 1 a 5                                               | Numérico   | Mantido como está                                                           |
| `Dalc`       | Consumo de álcool durante a semana        | 1 a 5                                               | Numérico   | Mantido como está                                                           |
| `Walc`       | Consumo de álcool no fim de semana        | 1 a 5                                               | Numérico   | Mantido como está                                                           |
| `health`     | Estado de saúde atual                     | 1 a 5                                               | Numérico   | Mantido como está                                                           |
| `absences`   | Faltas escolares                          | 0 a 93                                              | Numérico   | Mantido como está                                                           |
| `G1`, `G2`   | Notas dos períodos anteriores             | 0 a 20                                              | Numérico   | Mantido como está                                                           |
| `G3`         | Nota final (alvo)                         | 0 a 20                                              | Numérico   | Mantido como está                                                           |
                                                              |


### 🧪 Dados Fictícios Utilizados nas Predições

| Atributo    | Aluno 1       | Aluno 2       | Aluno 3       |
|-------------|---------------|---------------|---------------|
| school      | GP            | MS            | GP            |
| sex         | F             | M             | F             |
| age         | 17            | 18            | 16            |
| address     | U             | R             | U             |
| famsize     | GT3           | LE3           | GT3           |
| Pstatus     | T             | A             | T             |
| Medu        | 3             | 2             | 4             |
| Fedu        | 2             | 3             | 1             |
| Mjob        | health        | teacher       | at_home       |
| Fjob        | services      | other         | teacher       |
| reason      | course        | home          | reputation    |
| guardian    | mother        | father        | mother        |
| traveltime  | 1             | 2             | 3             |
| studytime   | 2             | 3             | 1             |
| failures    | 0             | 1             | 2             |
| schoolsup   | no            | yes           | no            |
| famsup      | yes           | no            | yes           |
| paid        | no            | yes           | no            |
| activities  | yes           | no            | yes           |
| nursery     | yes           | yes           | no            |
| higher      | yes           | yes           | no            |
| internet    | yes           | no            | yes           |
| romantic    | no            | yes           | no            |
| famrel      | 4             | 3             | 5             |
| freetime    | 3             | 2             | 4             |
| goout       | 3             | 4             | 2             |
| Dalc        | 1             | 2             | 1             |
| Walc        | 2             | 3             | 1             |
| health      | 5             | 3             | 4             |
| absences    | 4             | 10            | 2             |
| G1          | 12            | 10            | 15            |
| G2          | 14            | 11            | 16            |

## 📈 Resultados das Predições

Foram realizadas predições para três estudantes fictícios utilizando os modelos treinados. Abaixo estão os resultados previstos para a nota final (G3):

| Modelo           | Aluno   | G3 Previsto |
|------------------|---------|-------------|
| SVR              | Aluno 1 | 14          |
| SVR              | Aluno 2 | 11          |
| SVR              | Aluno 3 | 16          |
| Elastic Net      | Aluno 1 | 12          |
| Elastic Net      | Aluno 2 | 10          |
| Elastic Net      | Aluno 3 | 15          |
| Decision Tree    | Aluno 1 | 12          |
| Decision Tree    | Aluno 2 | 10          |
| Decision Tree    | Aluno 3 | 15          |
| Random Forest    | Aluno 1 | 12          |
| Random Forest    | Aluno 2 | 10          |
| Random Forest    | Aluno 3 | 15          |
| Ridge Regression | Aluno 1 | 12          |
| Ridge Regression | Aluno 2 | 10          |
| Ridge Regression | Aluno 3 | 15          |
| Lasso Regression | Aluno 1 | 12          |
| Lasso Regression | Aluno 2 | 10          |
| Lasso Regression | Aluno 3 | 15          |


## 📚 Referências

### 🧠 Artigos e Fontes Online

- Cao, Y., & Wang, X. (2021). *Support Vector Regression*. Disponível em: [https://www.ncbi.nlm.nih.gov/books/NBK583961](https://www.ncbi.nlm.nih.gov/books/NBK583961)
- IBM Corporation (2023). *Ridge Regression*. Disponível em: [https://www.ibm.com/br-pt/think/topics/ridge-regression](https://www.ibm.com/br-pt/think/topics/ridge-regression)
- IBM Corporation (2024). *Lasso Regression*. Disponível em: [https://www.ibm.com/br-pt/think/topics/lasso-regression](https://www.ibm.com/br-pt/think/topics/lasso-regression)
- Corporate Finance Institute (2023). *Elastic Net*. Disponível em: [https://corporatefinanceinstitute.com/resources/data-science/elastic-net](https://corporatefinanceinstitute.com/resources/data-science/elastic-net)
- IBM Corporation (2021). *Decision Trees*. Disponível em: [https://www.ibm.com/br-pt/think/topics/decision-trees](https://www.ibm.com/br-pt/think/topics/decision-trees)
- IBM Corporation (n.d.). *Random Forest*. Disponível em: [https://www.ibm.com/br-pt/think/topics/random-forest](https://www.ibm.com/br-pt/think/topics/random-forest)

## 👨‍💻 Autores

- Douglas C. Bezerra – [douglas.costab@ufrpe.br](mailto:douglas.costab@ufrpe.br)  
- Valdemir É. A. Silva – [valdemir.everton@ufrpe.br](mailto:valdemir.everton@ufrpe.br)  
Projeto desenvolvido no Bacharelado em Sistemas de Informação da UFRPE – Unidade Acadêmica de Serra Talhada.

## 🔗 Repositório

[https://github.com/selfDoga1/aprendizado-de-maquina](https://github.com/selfDoga1/aprendizado-de-maquina)
