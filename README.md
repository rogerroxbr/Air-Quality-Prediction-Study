# Air-Quality-Prediction-Study

## Estudo Abrangente de Previsão da Qualidade do Ar para Publicação Científica

Este repositório contém o código e os resultados de um estudo rigoroso focado no desenvolvimento e validação de um sistema de previsão da qualidade do ar. O projeto visa a publicação em um journal científico de alto impacto, priorizando interpretabilidade, robustez e capacidade de generalização.

## Objetivos do Estudo

*   Comparar metodicamente técnicas de seleção de features.
*   Identificar o algoritmo de regressão de melhor performance através de uma competição ampla com otimização de hiperparâmetros.
*   Validar a robustez temporal dos modelos usando validação cruzada específica para séries temporais.
*   Avaliar a capacidade de generalização dos modelos em um dataset de outro continente.
*   Quantificar a contribuição de diferentes fontes de informação (features) através de um estudo de ablação.
*   Interpretar as predições do modelo final usando técnicas de XAI (Explainable AI).
*   Estimar a incerteza das previsões do modelo com métodos estatisticamente robustos.
*   Medir o custo computacional e realizar análise estatística dos resultados.

## Funcionalidades Implementadas

*   **Processamento de Dados:** Consolidação, limpeza e harmonização de datasets de qualidade do ar de Beijing (China) e da Itália.
*   **Análise Exploratória de Dados (EDA):** Análise aprofundada do conjunto de treino com visualizações geradas.
*   **Seleção de Features:** Competição entre métodos de filtro (Correlação, Mutual Information) e wrapper/embarcados (RFECV, LassoCV).
*   **Competição e Otimização de Modelos:** Treinamento e otimização Bayesiana de um portfólio de algoritmos de regressão (LightGBM, XGBoost, RandomForest, CatBoost) contra benchmarks (Persistência, SARIMA).
*   **Avaliação Final e Estudo de Ablação:** Avaliação do modelo campeão em um conjunto de teste cego e análise da contribuição de grupos de features.
*   **Interpretabilidade (XAI):** Análise SHAP para explicar as predições do modelo.
*   **Quantificação de Incerteza:** Geração de intervalos de predição usando Regressão Quantílica.
*   **Teste de Generalização:** Avaliação da arquitetura do modelo em um dataset de domínio diferente (Itália, com alvo `C6H6(GT)`).
*   **Análise Estatística:** Comparação da significância estatística entre os melhores modelos.

## Resultados Principais

*   **Modelo Campeão:** LightGBM, com excelente performance e eficiência computacional.
*   **Performance Final (Teste):** MSE de ~208.25 e R² de ~0.9658.
*   **Significância Estatística:** A diferença de performance entre o LightGBM e o XGBoost foi considerada estatisticamente significante.
*   **Generalização:** A arquitetura do modelo demonstrou capacidade de generalização razoável para um novo problema e domínio de dados (R² de ~0.53 no dataset italiano).

## Estrutura do Repositório

```
. (raiz do projeto)
├── .gitignore
├── PRD.md
├── README.md
├── consolidate_prsa_data.py
├── 01_data_cleaning.py
├── 02_clean_airquality_uci.py
├── 03_temporal_split.py
├── 04_eda_training_set.py
├── 05_time_series_cv.py
├── 06_feature_selection_filter.py
├── 07_feature_selection_wrapper.py
├── 08_model_training_optimization.py
├── 09_model_evaluation_ablation.py
├── 10_shap_analysis.py
├── 11_uncertainty_quantification.py
├── 12_generalization_test.py
├── 13_statistical_analysis.py
├── report.md
├── report.tex
├── air+quality/ (dados brutos da Itália)
├── PRSA2017_Data_20130301-20170228/ (dados brutos de Beijing)
├── eda_plots/ (gráficos gerados pela EDA)
├── evaluation_plots/ (gráficos de avaliação final)
├── shap_plots/ (gráficos SHAP)
├── uncertainty_plots/ (gráficos de incerteza)
└── ... (outros arquivos gerados e ignorados pelo .gitignore)
```

## Como Começar

### Pré-requisitos

*   Python 3.8+
*   `uv` (recomendado para gerenciamento de pacotes)

### Instalação

1.  Clone o repositório:
    ```bash
    git clone https://github.com/SEU_USUARIO/Air-Quality-Prediction-Study.git
    cd Air-Quality-Prediction-Study
    ```
2.  Crie e ative um ambiente virtual:
    ```bash
    python -m venv .venv
    # No Windows
    .venv\Scripts\activate
    # No macOS/Linux
    source .venv/bin/activate
    ```
3.  Instale as dependências:
    ```bash
    uv pip install -r requirements.txt # (Você precisará gerar este arquivo)
    # Ou instale manualmente as principais bibliotecas:
    # uv pip install pandas numpy scikit-learn lightgbm xgboost catboost scikit-optimize statsmodels shap scipy
    ```

### Execução da Pipeline

Para reproduzir todas as análises e gerar os relatórios e gráficos, execute os scripts Python em sequência:

```bash
# Certifique-se de que o ambiente virtual está ativado
python consolidate_prsa_data.py
python 01_data_cleaning.py
python 02_clean_airquality_uci.py
python 03_temporal_split.py
python 04_eda_training_set.py
python 05_time_series_cv.py
python 06_feature_selection_filter.py
python 07_feature_selection_wrapper.py
python 08_model_training_optimization.py
python 09_model_evaluation_ablation.py
python 10_shap_analysis.py
python 11_uncertainty_quantification.py
python 12_generalization_test.py
python 13_statistical_analysis.py
```

## Relatórios

Os resultados completos do estudo, incluindo tabelas, gráficos e discussões, estão disponíveis nos seguintes arquivos:

*   `report.md` (Markdown)
*   `report.tex` (LaTeX, para compilação em PDF)

## Licença

[Adicione sua licença aqui, ex: MIT License]
