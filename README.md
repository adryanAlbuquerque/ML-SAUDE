# Machine Learning Aplicado à Saúde - Câncer de Mama

Este projeto demonstra o uso de técnicas de **aprendizado supervisionado e não supervisionado** aplicadas a um **dataset real de saúde**.

## Objetivo
Analisar dados médicos, tratar, modelar e interpretar resultados de forma acessível, utilizando aprendizado de máquina.

## Dataset
- Fonte: `sklearn.datasets.load_breast_cancer`
- Dados derivados de imagens microscópicas de tumores de mama.
- Cada registro representa um exame, com características como:
  - Raio médio dos núcleos celulares
  - Textura média
  - Área média
  - Suavidade
- Diagnóstico: 0 = Maligno, 1 = Benigno

## Etapas
1. Coleta e visualização dos dados  
2. Análise exploratória  
3. Filtro interativo de raio médio  
4. Tratamento (padronização)  
5. Modelo supervisionado (Regressão Logística)  
6. Modelo não supervisionado (K-Means)  
7. Interpretação dos resultados

## Tecnologias
- Python
- Streamlit
- scikit-learn
- Pandas
- Matplotlib
- Seaborn

## Como executar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
