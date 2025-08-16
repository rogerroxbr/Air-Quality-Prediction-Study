# **PRD: Estudo de Previsão de Qualidade do Ar para Publicação Científica (Versão Revisada)**

## **1. Objetivo**

Desenvolver e validar rigorosamente um sistema de previsão de qualidade do ar, com foco em interpretabilidade, robustez e generalização, visando a publicação em um journal científico de alto impacto. Os objetivos específicos são:

1. Comparar metodicamente técnicas de seleção de features.
2. Identificar o algoritmo de regressão de melhor performance através de uma competição ampla com otimização de hiperparâmetros.
3. Validar a robustez temporal dos modelos usando validação cruzada específica para séries temporais.
4. Avaliar a capacidade de generalização dos modelos em um dataset de outro continente.
5. Quantificar a contribuição de diferentes fontes de informação (features) através de um estudo de ablação.
6. Interpretar as predições do modelo final usando técnicas de XAI (Explainable AI).
7. Estimar a incerteza das previsões do modelo com métodos estatisticamente robustos.

## **2. Fontes de Dados**

1. **Dataset Primário (Treino/Teste):** PRSA2017\_Data\_20130301-20170228/ (Dados de Beijing).
2. **Dataset Secundário (Teste de Generalização):** air+quality/AirQualityUCI.csv (Dados da Itália).

## **3. Metodologia**

O projeto seguirá um fluxo de trabalho metodológico rigoroso, garantindo a reprodutibilidade e a validade científica dos resultados.

### **3.1. Preparação e Harmonização dos Dados**

*   **Tarefa:** Consolidação do dataset PRSA e limpeza inicial de ambos os datasets (tratamento de valores ausentes, tipos de dados, outliers).
*   **Tarefa:** Realizar uma análise de compatibilidade e harmonização entre os datasets de Beijing e da Itália. Isso inclui mapear features equivalentes, comparar distribuições estatísticas e definir uma estratégia de normalização comum para o teste de generalização.

### **3.2. Análise Preliminar e Estratégia de Validação**

*   **Tarefa:** Análise de alto nível para entender a estrutura dos dados (sazonalidade, tendências, autocorrelação).
*   **Tarefa:** Definição da estratégia de validação cruzada temporal (ex: TimeSeriesSplit do Scikit-learn) que será usada consistentemente em todas as etapas de modelagem para evitar vazamento de dados (data leakage).

### **3.3. Divisão dos Dados (Train/Test Split)**

*   **Tarefa:** Divisão do dataset primário (PRSA) em conjuntos de treino e teste, respeitando estritamente a ordem cronológica. O conjunto de teste será "cego" e usado apenas uma vez na avaliação final.

### **3.4. Análise Exploratória (no Conjunto de Treino)**

*   **Tarefa:** EDA detalhada no conjunto de treino para guiar a engenharia de features. Incluirá análise de correlação, decomposição de séries temporais e visualização de relações entre as variáveis.

### **3.5. Competição de Seleção de Features**

*   **Tarefa:** Gerar e avaliar subconjuntos de features usando uma variedade de abordagens:
    1.  **Métodos de Filtro:** Ex: Coeficiente de Correlação de Pearson, Mutual Information.
    2.  **Métodos Wrapper:** Ex: Recursive Feature Elimination (RFE).
    3.  **Métodos Embarcados (Embedded):** Ex: Lasso (L1 Regularization).
    4.  **Métodos Híbridos:** Ex: BorutaShap.
*   **Processo de Avaliação:** A performance de cada subconjunto de features será avaliada usando um **modelo baseline único e eficiente (LightGBM com hiperparâmetros padrão)**. A avaliação será conduzida dentro do framework de validação cruzada temporal. Esta abordagem é computacionalmente tratável e metodologicamente sólida, usando um modelo forte como um proxy para identificar o conjunto de features mais informativo. O subconjunto vencedor será utilizado na etapa seguinte.

### **3.6. Competição de Modelos e Otimização**

*   **Tarefa:** Usando o conjunto de features vencedor, treinar e otimizar um portfólio de algoritmos de regressão. A avaliação será feita contra benchmarks robustos.
*   **Benchmarks:**
    1.  **Modelo de Persistência (Naive):** Prediz que o valor futuro será igual ao valor atual.
    2.  **Modelo Estatístico Clássico:** SARIMA (Seasonal AutoRegressive Integrated Moving Average) para estabelecer uma linha de base robusta.
*   **Lista de Algoritmos:** ElasticNet, RandomForest, Gradient Boosting, XGBoost, LightGBM, CatBoost, ExtraTreesRegressor, SVR.
*   **Processo:** Otimização Bayesiana para os hiperparâmetros de cada algoritmo, executada dentro do loop de validação cruzada temporal no conjunto de treino.
*   **Métricas de Avaliação:** A performance será medida usando um conjunto abrangente de métricas: **MSE (Mean Squared Error), RMSE (Root Mean Squared Error), MAE (Mean Absolute Error) e R² (R-squared)**.
*   **Análise de Custo Computacional:** O **tempo de treinamento** de cada modelo durante a otimização será medido e reportado para avaliar a eficiência computacional.

### **3.7. Avaliação Final e Estudo de Ablação**

*   **Tarefa:** Selecionar o modelo campeão (melhor performance média na validação cruzada), avaliá-lo no conjunto de teste (PRSA) e conduzir um estudo de ablação.
*   **Estudo de Ablação:** Remover sistematicamente grupos de features pré-definidos para quantificar a importância de cada tipo de informação. Os grupos serão definidos semanticamente (ex: features temporais, meteorológicas, poluentes defasados, outros poluentes).

### **3.8. Análise de Interpretabilidade (XAI)**

*   **Tarefa:** Aplicar o SHAP (SHapley Additive exPlanations) no modelo final para explicar suas predições.
*   **Análises:** Gerar summary plots (importância global), dependence plots (efeito de cada feature), force plots (explicação de predições individuais) e **interaction plots** para descobrir relações não-lineares entre as variáveis.

### **3.9. Quantificação de Incerteza**

*   **Tarefa:** Estimar a incerteza das previsões do modelo final para gerar intervalos de predição.
*   **Técnicas a serem exploradas:** Regressão Quantílica (nativa em LightGBM/XGBoost), Bootstrapping ou **Previsão Conformal (Conformal Prediction)**, um método moderno que oferece garantias estatísticas rigorosas.

### **3.10. Teste de Generalização Cross-Dataset**

*   **Tarefa:** Avaliar a robustez e a capacidade de generalização do modelo campeão no dataset AirQualityUCI (Itália), que possui um domínio de dados diferente.
*   **Cenários de Avaliação:**
    1.  **Zero-Shot:** Aplicar o modelo treinado em Beijing diretamente nos dados da Itália (após harmonização) para medir a generalização pura.
    2.  **Fine-Tuning (Opcional):** Continuar o treinamento do modelo campeão com uma pequena parte dos dados da Itália para medir a adaptabilidade.
    3.  **Treinamento do Zero:** Treinar o mesmo modelo, com os mesmos hiperparâmetros, apenas com os dados da Itália para criar uma linha de base de performance local.

### **3.11. Análise Estatística dos Resultados**

*   **Tarefa:** Realizar uma análise de significância estatística (ex: teste t pareado) para comparar os resultados de performance dos melhores modelos. O objetivo é determinar se a diferença de performance entre o modelo campeão e o segundo colocado é estatisticamente significativa ou apenas resultado da variabilidade dos dados na validação cruzada.

## **4. Entregáveis**

*   **Jupyter Notebook (analise\_qualidade\_ar.ipynb):** Notebook completo e reprodutível com todo o código, análises, comparações e conclusões.
*   **Arquivo de dados consolidado (PRSA\_consolidado.csv).**
*   **Pré-Artigo / Relatório de Resultados:** Um documento formatado (ex: Markdown ou PDF) com os resultados, gráficos, tabelas e análises, estruturado de forma a facilitar a adaptação para a submissão de um artigo científico.
