import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="Análise de Saúde - COVID-19", layout="wide")

@st.cache_data
def carregar_dados():
    dados = {
        "estado": ["SP", "RJ", "MG", "BA", "RS", "PR", "SC", "PE", "CE", "PA",
                   "GO", "AM", "ES", "PB", "RN", "MT", "DF", "MS", "MA", "PI",
                   "SE", "AL", "TO", "RO", "RR", "AP", "AC"],
        "casos": [1200000, 800000, 750000, 650000, 620000, 580000, 500000, 470000, 450000, 420000,
                  400000, 390000, 380000, 350000, 340000, 320000, 310000, 300000, 290000, 280000,
                  250000, 240000, 230000, 220000, 200000, 180000, 160000],
        "obitos": [45000, 30000, 25000, 22000, 21000, 20000, 17000, 15000, 14000, 13000,
                   12000, 11000, 10000, 9000, 8500, 8000, 7800, 7600, 7500, 7400,
                   7000, 6800, 6600, 6400, 6000, 5800, 5600]
    }
    return pd.DataFrame(dados)

dados = carregar_dados()

st.title("📊 Machine Learning Aplicado à Saúde")
st.subheader("Predição de Óbitos por COVID-19 com Regressão Linear")
st.markdown("""
Este projeto demonstra a aplicação de **aprendizado supervisionado** na área da saúde, utilizando dados simplificados de COVID-19 no Brasil.
""")

aba = st.sidebar.radio("Selecione a Seção", ["Visão Geral", "Análise Exploratória", "Modelagem Supervisionada"])

if aba == "Visão Geral":
    st.header("Visão Geral dos Dados")
    st.dataframe(dados, use_container_width=True)

    st.markdown("""
    Os dados simulam o total de casos e óbitos de COVID-19 por estado brasileiro.  
    Essa base serve para demonstrar o processo de coleta, tratamento e modelagem preditiva.
    """)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=dados, x="estado", y="casos", color="skyblue", label="Casos")
    sns.barplot(data=dados, x="estado", y="obitos", color="salmon", label="Óbitos")
    ax.legend()
    ax.set_title("Casos e Óbitos por Estado")
    ax.set_xlabel("Estado")
    ax.set_ylabel("Quantidade")
    st.pyplot(fig)
    plt.close(fig)

elif aba == "Análise Exploratória":
    st.header("Análise Exploratória dos Dados")

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=dados, x="casos", y="obitos", s=100)
    ax1.set_title("Relação entre Casos e Óbitos")
    st.pyplot(fig1)
    plt.close(fig1)

    correlacao = dados["casos"].corr(dados["obitos"])
    st.metric("Correlação entre Casos e Óbitos", f"{correlacao:.2f}")

    st.markdown("""
    **Interpretação:**  
    Observa-se uma correlação positiva alta entre casos e óbitos, indicando que quanto mais casos confirmados, maior tende a ser o número de óbitos.
    """)

elif aba == "Modelagem Supervisionada":
    st.header("Predição de Óbitos com Regressão Linear")
    st.markdown("""
    Esta seção aplica um modelo de **Regressão Linear** para estimar o número de óbitos a partir da quantidade de casos confirmados.  
    O objetivo é demonstrar uma técnica simples de aprendizado supervisionado.
    """)

    X = dados[["casos"]].values
    y = dados["obitos"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    col1, col2, col3 = st.columns(3)
    col1.metric("R²", f"{r2:.2f}")
    col2.metric("MAE", f"{mae:.0f}")
    col3.metric("RMSE", f"{rmse:.0f}")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.regplot(x=y_test, y=y_pred, line_kws={"color": "gray"}, scatter_kws={"s": 80})
    ax2.set_xlabel("Óbitos Reais")
    ax2.set_ylabel("Óbitos Preditos")
    ax2.set_title("Desempenho da Regressão Linear")
    st.pyplot(fig2)
    plt.close(fig2)

    st.markdown("""
    **Interpretação:**  
    O modelo conseguiu explicar parte da relação linear entre casos e óbitos.  
    Embora simples, ele demonstra como técnicas de aprendizado supervisionado podem ser aplicadas em dados de saúde para fins de estimativa e análise preditiva.
    """)

st.markdown("---")
st.caption("Desenvolvido para a disciplina de *Machine Learning Aplicado à Saúde*.")
